"""State -> Agent -> real local tools, with a deterministic snapshot-reading LLM.

This fake deliberately does not resolve identities from history. It verifies
integration and actual tool arguments, not a real model's instruction following.
"""

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.agent.agent import STATE_CONTEXT_PREFIX, build_agent, run_agent_turn
from app.agent.state import SessionState


class StateToolFakeModel(FakeMessagesListChatModel):
    snapshots: list[dict[str, Any]] = Field(default_factory=list)
    unsafe_reply: str | None = None

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "StateToolFakeModel":
        return self

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        snapshot_text = next(
            str(message.content)
            for message in reversed(messages)
            if message.type == "system" and str(message.content).startswith(STATE_CONTEXT_PREFIX)
        )
        state = json.loads(snapshot_text.split("\n")[1])
        self.snapshots.append(state)
        if self.unsafe_reply is not None:
            reply = AIMessage(content=self.unsafe_reply)
            self.unsafe_reply = None
        elif isinstance(messages[-1], ToolMessage):
            result = json.loads(messages[-1].content)
            if messages[-1].name == "compare_product":
                reply = AIMessage(content="已比较：" + "、".join(result["sku_ids"]))
            elif result.get("results"):
                product = result["results"][0]
                reply = AIMessage(content=f"{product['sku_id']}：{product[state['query_field']]}")
            else:
                reply = AIMessage(content="没有查到该商品资料。")
        else:
            if state["current_sku"] and state["query_field"]:
                name, args = "search_product", {"query": state["current_sku"]}
            elif state["comparison_skus"] and state["current_sku"] is None:
                name, args = "compare_product", {"sku_ids": ",".join(state["comparison_skus"])}
            else:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="已记录本轮需求。"))])
            reply = AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": f"state_call_{len(self.snapshots)}",
            }])
        return ChatResult(generations=[ChatGeneration(message=reply)])


def new_runtime() -> tuple[dict[str, Any], StateToolFakeModel]:
    model = StateToolFakeModel(responses=[])
    return build_agent(llm=model), model


def last_tool_args(runtime: dict[str, Any]) -> dict[str, Any]:
    calls = [call for message in runtime["last_result"]["messages"]
             if isinstance(message, AIMessage) for call in message.tool_calls]
    return calls[-1]["args"]


def test_main_flow_switch_correction_and_reference_drive_actual_tools() -> None:
    runtime, model = new_runtime()
    assert run_agent_turn(runtime, "S001多少钱") == "S001：176 元"
    run_agent_turn(runtime, "改看S002")
    assert run_agent_turn(runtime, "它多少钱") == "S002：84 元"
    assert last_tool_args(runtime) == {"query": "S002"}
    run_agent_turn(runtime, "不是S002，是S001")
    assert run_agent_turn(runtime, "它的规格") == "S001：48 粒/瓶"
    assert last_tool_args(runtime) == {"query": "S001"}
    assert model.snapshots[-1]["sources"]["current_sku"]["basis"] == "current_product_reference"
    # Transient snapshots are not appended into normal conversation history.
    assert len(runtime["memory"].get_history()) == 10
    assert all(STATE_CONTEXT_PREFIX not in m["content"] for m in runtime["memory"].get_history())


def test_comparison_reversal_drives_reference_queries_in_actual_order() -> None:
    runtime, _ = new_runtime()
    for first, second, first_spec, second_price in [
        ("S001", "S002", "48 粒/瓶", "84 元"),
        ("S002", "S001", "22g/支", "176 元"),
    ]:
        run_agent_turn(runtime, f"比较{first}和{second}")
        assert last_tool_args(runtime) == {"sku_ids": f"{first},{second}"}
        assert run_agent_turn(runtime, "前者的规格") == f"{first}：{first_spec}"
        assert last_tool_args(runtime) == {"query": first}
        assert run_agent_turn(runtime, "后者价格") == f"{second}：{second_price}"
        assert last_tool_args(runtime) == {"query": second}
    run_agent_turn(runtime, "比较顺序反过来")
    assert last_tool_args(runtime) == {"sku_ids": "S001,S002"}
    assert run_agent_turn(runtime, "前者规格") == "S001：48 粒/瓶"


def test_ambiguity_clarifies_without_model_or_tool_then_accepts_selection() -> None:
    runtime, model = new_runtime()
    run_agent_turn(runtime, "S001多少钱")
    run_agent_turn(runtime, "比较S001和S002")
    before = len(model.snapshots)
    answer = run_agent_turn(runtime, "它多少钱")
    assert "确认" in answer or "哪个商品" in answer
    assert len(model.snapshots) == before
    assert runtime.get("last_result") is None
    assert runtime["memory"].get_state().current_sku is None
    assert len(runtime["memory"].get_history()) == 6
    assert run_agent_turn(runtime, "S002多少钱") == "S002：84 元"


def test_unknown_named_product_does_not_query_previous_sku() -> None:
    runtime, model = new_runtime()
    run_agent_turn(runtime, "S001多少钱")
    before = len(model.snapshots)
    assert "未找到" in run_agent_turn(runtime, "不存在的维生素多少钱")
    assert runtime["memory"].get_state().current_sku is None
    assert len(model.snapshots) == before
    assert runtime["last_result"]["messages"][-1].name == "search_product"
    assert json.loads(runtime["last_result"]["messages"][-1].content)["results"] == []


def test_current_budget_not_old_history_is_sent_to_agent() -> None:
    runtime, model = new_runtime()
    for text, expected in [
        ("预算100元以内", 100), ("改成200元以内", 200), ("预算不限", None),
    ]:
        run_agent_turn(runtime, text)
        assert model.snapshots[-1]["budget_max"] == expected
    run_agent_turn(runtime, "S001多少钱")
    assert model.snapshots[-1]["budget_max"] is None
    assert "预算100元以内" in [m["content"] for m in runtime["memory"].get_history()]
    assert model.snapshots[-1]["sources"]["budget_max"]["basis"] == "user_retraction"


def test_new_need_and_reset_do_not_resurrect_product_from_history() -> None:
    runtime, model = new_runtime()
    run_agent_turn(runtime, "S001多少钱，预算100元以内")
    run_agent_turn(runtime, "换个话题，想买维生素")
    assert model.snapshots[-1]["current_sku"] is None
    assert model.snapshots[-1]["budget_max"] is None
    assert model.snapshots[-1]["goal"] == "维生素"
    before = len(model.snapshots)
    assert "哪个商品" in run_agent_turn(runtime, "它多少钱")
    assert len(model.snapshots) == before
    runtime["memory"].clear()
    assert runtime["memory"].get_state() == SessionState()
    assert "哪个商品" in run_agent_turn(runtime, "前者多少钱")
    assert len(model.snapshots) == before


def test_interleaved_sessions_and_reset_are_independent() -> None:
    first, _ = new_runtime()
    second, _ = new_runtime()
    run_agent_turn(first, "S001多少钱，预算100元以内")
    run_agent_turn(second, "S002多少钱，预算200元以内")
    assert run_agent_turn(first, "它多少钱") == "S001：176 元"
    assert run_agent_turn(second, "它多少钱") == "S002：84 元"
    first["memory"].clear()
    assert first["memory"].get_state() == SessionState()
    assert run_agent_turn(second, "它多少钱") == "S002：84 元"
    assert second["memory"].get_state().budget_max == 200


def test_rejected_input_and_candidate_cannot_become_preferences() -> None:
    runtime, model = new_runtime()
    run_agent_turn(runtime, "S001多少钱，目标是日常养护，预算200元以内")
    before = len(model.snapshots)
    run_agent_turn(runtime, "改看S002，目标是治好皮炎")
    assert len(model.snapshots) == before
    state = runtime["memory"].get_state()
    assert state.current_sku == "S002"
    assert state.goal == "日常养护"
    assert "治好皮炎" not in json.dumps(state.model_context(), ensure_ascii=False)
    assert run_agent_turn(runtime, "它多少钱") == "S002：84 元"
    unsafe_candidate = "S001 治疗皮炎，目标是治愈，预算1元"
    model.unsafe_reply = unsafe_candidate
    run_agent_turn(runtime, "改看S001")
    assert runtime["memory"].get_last_turn()["output_check"]["safe"] is False
    state = runtime["memory"].get_state()
    assert state.current_sku == "S001"
    assert state.budget_max == 200
    assert state.goal == "日常养护"
    assert run_agent_turn(runtime, "它多少钱") == "S001：176 元"
    assert all(unsafe_candidate not in m["content"] for m in runtime["memory"].get_history())
