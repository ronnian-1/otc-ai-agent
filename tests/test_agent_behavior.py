"""Agent behavior boundary tests for Prompt-guided tool selection.

These tests use a mock tool-calling LLM to verify the Agent routing boundary:
product lookup should call search_product, product comparison should call
compare_product, and non-product platform questions should not call product tools.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.agent.agent import SAFETY_FALLBACK_MESSAGE, STATE_CONTEXT_PREFIX, build_agent, run_agent_turn
from app.agent.reply import CLOSING_REPLY, REPLY_CONTEXT_PREFIX, plan_reply
from app.agent.state import SessionState
from app.safety.guard import check_ai_output, check_user_input


class PromptAwareFakeModel(FakeMessagesListChatModel):
    """Mock LLM that emits tool calls based on the latest user message."""

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "PromptAwareFakeModel":
        """Return self so LangChain Agent can bind tools during tests."""
        _ = tools, tool_choice, kwargs
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any) -> Any:
        """Generate deterministic tool decisions for behavior-boundary tests."""
        user_text = _latest_user_text(messages)

        if "S001" in user_text and "S002" in user_text:
            self.responses = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "compare_product",
                            "args": {"sku_ids": "S001,S002"},
                            "id": "mock_compare_call",
                        }
                    ],
                ),
                AIMessage(content="这是两个商品的结构化对比回答。"),
            ]
        elif "S001" in user_text:
            self.responses = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_product",
                            "args": {"query": "S001"},
                            "id": "mock_search_call",
                        }
                    ],
                ),
                AIMessage(content="S001 的价格信息已查询。"),
            ]
        else:
            self.responses = [AIMessage(content="保税仓发货时效通常以平台规则为准。")]

        return super()._generate(messages, stop=stop, **kwargs)


def _latest_user_text(messages: list[BaseMessage]) -> str:
    """Return the latest human message content from Agent input."""
    for message in reversed(messages):
        if message.type == "human":
            return str(message.content)
    return ""


def _run_agent(user_input: str) -> list[BaseMessage]:
    """Run Agent with the behavior-test mock LLM."""
    runtime = build_agent(llm=PromptAwareFakeModel(responses=[]))
    result = runtime["agent"].invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )
    return result["messages"]


def _tool_calls(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Collect tool calls emitted by AI messages."""
    calls: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            calls.extend(message.tool_calls)
    return calls


def test_agent_selects_search_product_for_sku_price_question() -> None:
    """S001 price question should route to search_product with query=S001."""
    messages = _run_agent("S001多少钱？")
    calls = _tool_calls(messages)

    assert calls
    assert calls[0]["name"] == "search_product"
    assert calls[0]["args"] == {"query": "S001"}


def test_agent_selects_compare_product_for_two_sku_difference_question() -> None:
    """Two-SKU difference question should route to compare_product."""
    messages = _run_agent("S001和S002有什么区别？")
    calls = _tool_calls(messages)

    assert calls
    assert calls[0]["name"] == "compare_product"
    assert calls[0]["args"] == {"sku_ids": "S001,S002"}


def test_agent_does_not_call_product_tools_for_platform_question() -> None:
    """Platform-rule questions should be answered without product tool calls."""
    messages = _run_agent("保税仓多久发货？")
    calls = _tool_calls(messages)
    final_message = messages[-1]

    assert calls == []
    assert final_message.content


class PolicyReadingFakeModel(FakeMessagesListChatModel):
    """Interpret the actual reply plan, not user intent or prompt prose.

    Output assertions validate this deterministic fake + real local tool chain.
    They cannot establish a real LLM's compliance or subjective conversation tone.
    """

    observed_plans: list[dict[str, Any]] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    long_first_answer: bool = False
    unsafe_once: str | None = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: list[BaseMessage], **kwargs):
        def snapshot(prefix):
            content = next(str(m.content) for m in reversed(messages)
                           if m.type == "system" and str(m.content).startswith(prefix))
            return json.loads(content.split("\n")[1])

        plan, state = snapshot(REPLY_CONTEXT_PREFIX), snapshot(STATE_CONTEXT_PREFIX)
        self.observed_plans.append(plan)
        if self.unsafe_once:
            reply = AIMessage(content=self.unsafe_once)
            self.unsafe_once = None
        elif isinstance(messages[-1], ToolMessage):
            result = json.loads(messages[-1].content)
            self.tool_outputs.append(result)
            if result.get("error") or result.get("_tool_evidence", {}).get("status") == "error":
                text = "当前无法确认所问商品资料。"
            elif messages[-1].name == "compare_product":
                fields = plan["fields"] or ["price", "spec"]
                text = "；".join(
                    sku + "：" + "、".join(
                        str(result["basic_difference"].get(f, {}).get(side, "当前资料未提供"))
                        for f in fields)
                    for sku, side in zip(result["sku_ids"], ("sku_a", "sku_b")))
            elif not result.get("results"):
                text = "当前无法确认该商品的价格或规格。"
            else:
                row = result["results"][0]
                text = row["sku_id"] + "：" + "、".join(
                    str(row.get(f) or "当前资料未提供") for f in plan["fields"])
                if self.long_first_answer and not plan["repeated"]:
                    text += "。" + row["name"] + "，" + row["product_desc"] + "。可以继续了解商品。"
            if plan["correction"]:
                text = "已改为" + plan["skus"][0] + "。" + text
            reply = AIMessage(content=text)
        elif plan["requires_tool_evidence"]:
            comparing = plan["intent"] == "comparison"
            name = "compare_product" if comparing else "search_product"
            args = ({"sku_ids": ",".join(plan["skus"])} if comparing
                    else {"query": plan["skus"][0]})
            reply = AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": f"policy_call_{len(self.observed_plans)}",
            }])
        elif plan["intent"] == "budget_concern":
            prefix = (f"预算上限仍按{state['budget_max']:g}元。" if state["budget_max"] is not None
                      else "可以先不急着决定。")
            reply = AIMessage(content=prefix + "要不要调整预算范围，或比较其他商品？")
        elif plan["intent"] == "correction":
            reply = AIMessage(content="已改为" + plan["skus"][0] + "。")
        elif plan["intent"] == "hesitation":
            reply = AIMessage(content="可以慢慢考虑，有想核对的地方再告诉我。")
        else:
            reply = AIMessage(content="你更想了解商品的哪方面？")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def policy_runtime(**kwargs):
    model = PolicyReadingFakeModel(responses=[], **kwargs)
    return build_agent(llm=model), model


@pytest.mark.parametrize(("query", "field", "value", "unrelated"), [
    ("S001多少钱", "price", "176 元", "48 粒/瓶"),
    ("S002规格", "spec", "22g/支", "84 元"),
])
def test_field_reply_uses_actual_local_tool_value_without_full_intro(query, field, value, unrelated):
    runtime, model = policy_runtime()
    answer = run_agent_turn(runtime, query)
    assert value in answer and unrelated not in answer
    assert "购买" not in answer and "适用人群" not in answer
    assert runtime["last_reply_plan"]["intent"] == "field_query"
    assert runtime["last_reply_plan"]["fields"] == (field,)
    assert model.tool_outputs[-1]["_tool_evidence"]["status"] == "success"
    assert runtime["memory"].get_last_turn()["output_check"]["safe"] is True


def test_comparison_focus_followup_and_references_keep_actual_order():
    runtime, model = policy_runtime()
    answer = run_agent_turn(runtime, "比较S002和S001的价格")
    assert answer == "S002：84 元；S001：176 元"
    assert model.tool_outputs[-1]["sku_ids"] == ["S002", "S001"]
    assert "22g" not in answer and "48 粒" not in answer
    assert run_agent_turn(runtime, "只看规格") == "S002：22g/支；S001：48 粒/瓶"
    assert runtime["last_reply_plan"]["fields"] == ("spec",)
    assert runtime["memory"].get_state().comparison_skus == ("S002", "S001")
    assert run_agent_turn(runtime, "前者多少钱") == "S002：84 元"
    assert run_agent_turn(runtime, "后者的规格") == "S001：48 粒/瓶"


@pytest.mark.parametrize("concern", ["有点贵", "超预算了", "超出我的预算", "这个价格有点贵", "S001价格太贵"])
def test_budget_concern_does_not_change_budget_select_alternative_or_query_tools(concern):
    runtime, model = policy_runtime()
    run_agent_turn(runtime, "S001多少钱，预算200元以内")
    before = len(model.tool_outputs)
    answer = run_agent_turn(runtime, concern)
    state = runtime["memory"].get_state()
    assert state.budget_max == 200 and state.current_sku == "S001"
    assert runtime["last_reply_plan"]["intent"] == "budget_concern"
    assert not runtime["last_reply_plan"]["requires_tool_evidence"]
    assert len(model.tool_outputs) == before
    assert "200" in answer and "S002" not in answer and "优惠" not in answer
    assert "要不要" in answer  # Offer a choice, do not execute it automatically.


def test_concern_after_comparison_does_not_silently_repeat_comparison():
    runtime, model = policy_runtime()
    run_agent_turn(runtime, "比较S001和S002价格")
    before = len(model.tool_outputs)
    run_agent_turn(runtime, "太贵了")
    assert runtime["last_reply_plan"]["intent"] == "budget_concern"
    assert runtime["memory"].get_state().budget_max is None
    assert len(model.tool_outputs) == before


def test_price_concern_is_not_a_repeated_query_but_can_include_an_explicit_question():
    runtime, model = policy_runtime()
    run_agent_turn(runtime, "S001多少钱")
    run_agent_turn(runtime, "这个价格有点贵")
    assert runtime["last_reply_plan"]["intent"] == "budget_concern"
    assert run_agent_turn(runtime, "它多少钱") == "S001：176 元"
    assert runtime["last_reply_plan"]["repeated"] is False
    assert run_agent_turn(runtime, "有点贵，S002多少钱") == "S002：84 元"
    assert runtime["last_reply_plan"]["intent"] == "field_query"
    assert runtime["last_reply_plan"]["budget_concern"] is True


def test_correction_answers_new_sku_field_once_and_preserves_old_evidence_ownership():
    runtime, model = policy_runtime()
    run_agent_turn(runtime, "S001多少钱")
    answer = run_agent_turn(runtime, "不是S001，是S002，多少钱")
    assert answer == "已改为S002。S002：84 元"
    assert "176" not in answer and "抱歉" not in answer
    assert runtime["last_reply_plan"]["correction"] is True
    assert model.tool_outputs[-1]["query"] == "S002"
    assert runtime["memory"].get_state().current_sku == "S002"
    assert run_agent_turn(runtime, "我说的是S001") == "已改为S001。"
    assert runtime["last_reply_plan"]["requires_tool_evidence"] is False


def test_repeat_becomes_direct_and_reuses_verified_result_without_new_facts():
    runtime, model = policy_runtime(long_first_answer=True)
    first = run_agent_turn(runtime, "S001多少钱")
    second = run_agent_turn(runtime, "再说一遍价格")
    assert "虚构的口服营养类展示样本，用于演示片剂包装与商品资料核对" in first
    assert second == "S001：176 元" and len(second) < len(first)
    assert runtime["last_reply_plan"]["repeated"] is True
    assert model.tool_outputs[-1]["_tool_evidence"]["reused"] is True
    assert len(runtime["memory"].get_history()) == 4
    assert all(REPLY_CONTEXT_PREFIX not in m["content"] for m in runtime["memory"].get_history())


@pytest.mark.parametrize("closing", ["知道了", "不用了", "先这样", "好的，谢谢", "知道了，不用了，谢谢"])
def test_explicit_closing_has_no_model_tool_or_sales_and_is_still_reviewed(closing):
    runtime, model = policy_runtime()
    run_agent_turn(runtime, "S001多少钱")
    before = len(model.observed_plans)
    reviews = []

    def review(text):
        reviews.append(text)
        return check_ai_output(text)

    assert run_agent_turn(runtime, closing, output_guard=review) == CLOSING_REPLY
    assert reviews == [CLOSING_REPLY]
    assert len(model.observed_plans) == before
    assert runtime.get("last_result") is None
    assert runtime["last_reply_plan"]["intent"] == "closing"
    assert runtime["memory"].get_history()[-1]["content"] == CLOSING_REPLY
    assert run_agent_turn(runtime, "S002多少钱") == "S002：84 元"  # User may resume.


@pytest.mark.parametrize("text", ["我再想想", "还没决定", "知道了，S001多少钱", "不用介绍了，S001多少钱"])
def test_hesitation_or_a_new_question_is_not_a_closing(text):
    runtime, model = policy_runtime()
    run_agent_turn(runtime, text)
    assert runtime["last_reply_plan"]["intent"] != "closing"
    assert model.observed_plans


def test_unknown_product_and_ambiguous_reference_cannot_reuse_old_identity():
    runtime, model = policy_runtime()
    run_agent_turn(runtime, "S001多少钱")
    answer = run_agent_turn(runtime, "S999多少钱")
    assert "无法确认" in answer and "176" not in answer
    assert model.tool_outputs[-1]["results"] == []
    run_agent_turn(runtime, "比较S001和S002价格")
    before = len(model.observed_plans)
    answer = run_agent_turn(runtime, "它多少钱")
    assert "哪个商品" in answer or "确认" in answer
    assert runtime["last_reply_plan"]["intent"] == "clarification"
    assert len(model.observed_plans) == before
    assert runtime["memory"].get_last_turn()["output_check"]["safe"] is True


def test_missing_field_is_not_guessed_from_intro_or_another_sku(monkeypatch):
    search_module = importlib.import_module("app.tools.sku_search")
    original = search_module.load_sku_data

    def without_spec():
        return [{k: v for k, v in row.items() if k != "spec"} for row in original()]

    monkeypatch.setattr(search_module, "load_sku_data", without_spec)
    runtime, model = policy_runtime()
    answer = run_agent_turn(runtime, "S002规格")
    assert "当前资料未提供" in answer and "22g" not in answer
    assert model.tool_outputs[-1]["_tool_evidence"]["status"] == "partial"


@pytest.mark.parametrize("query", [
    "S001多少钱", "S002规格", "比较S002和S001价格", "有点贵", "不是S001，是S002，多少钱", "我再想想",
])
def test_generated_replies_in_every_strategy_still_hit_p11_review(query):
    runtime, model = policy_runtime(unsafe_once="不可用于疾病治疗，但它肯定有效")
    assert run_agent_turn(runtime, query) == SAFETY_FALLBACK_MESSAGE
    turn = runtime["memory"].get_last_turn()
    assert turn["output_check"]["safe"] is False
    assert "absolute_efficacy" in turn["output_check"]["risk_types"]
    assert all("肯定有效" not in m["content"] for m in runtime["memory"].get_history())
    assert runtime["memory"].facts.get_query("search_product", {"query": "S001"}) is None


def test_repeat_and_server_closing_cannot_bypass_output_rejection():
    runtime, model = policy_runtime()
    run_agent_turn(runtime, "S001多少钱")
    model.unsafe_once = "S001肯定有效"
    assert run_agent_turn(runtime, "它多少钱") == SAFETY_FALLBACK_MESSAGE
    assert runtime["last_reply_plan"]["repeated"] is True
    assert run_agent_turn(runtime, "不用了", output_guard=lambda text: {
        "safe": False, "filtered_text": "", "reason": "test rejection",
    }) == SAFETY_FALLBACK_MESSAGE
    assert "不是商品事实" in runtime["memory"].get_history()[-1]["content"]


def test_input_safety_precedes_closing_or_intent_planning_and_clears_stale_plan():
    runtime, model = policy_runtime()
    run_agent_turn(runtime, "S001多少钱")
    before = len(model.observed_plans)
    reply = run_agent_turn(runtime, "不用了，S002能治好皮炎吗")
    assert "请咨询医生或药师" in reply
    assert len(model.observed_plans) == before
    assert "last_reply_plan" not in runtime
    assert runtime["memory"].get_state().current_sku == "S002"


def test_unknown_intent_and_repetition_use_evidence_without_inventing_a_goal():
    state = SessionState(current_sku="S001", query_field="price")
    old = SessionState(current_sku="S002", query_field="price")
    last = {"user_input": "S002多少钱", "input_check": {"risk": False}, "output_check": {"safe": True}}
    plan = plan_reply("S001多少钱", state, old, last)
    assert not plan.repeated and plan.requires_tool_evidence
    ambiguous = plan_reply("随便聊聊", SessionState(), SessionState(), None)
    assert ambiguous.intent == "general" and ambiguous.basis == "no_deterministic_intent"
    assert ambiguous.skus == () and ambiguous.fields == ()
