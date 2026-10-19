"""Tests for Agent and Safety Guard integration.

These tests verify the enterprise flow of input guard -> Agent/Tool -> output
guard using mocked LLMs. No real model API, Tool changes, or data changes are
required.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.agent.agent import SAFETY_FALLBACK_MESSAGE, build_agent, run_agent_turn
from app.safety.guard import check_ai_output, check_user_input
from app.tools.data_loader import load_sku_data


@pytest.mark.parametrize("blocked_at", ["input", "candidate"])
def test_fixed_refusal_has_separate_final_review_without_promoting_it_to_facts(blocked_at):
    from app.safety.guard import DISCLAIMER_MESSAGE

    danger = "根据你的皮炎情况，S002每天半管。"
    model = ToolCallingFakeModel(responses=_search_replies("S002", danger))
    runtime = build_agent(llm=model)
    question = "我有皮炎，S002每天半管可以吗？" if blocked_at == "input" else "S002规格"
    answer = run_agent_turn(runtime, question)
    turn = runtime["memory"].get_last_turn()
    blocked_check = turn["input_check"] if blocked_at == "input" else turn["output_check"]
    review = blocked_check["refusal_review"]
    assert review["safe"] is True
    assert review["decision"] == "allow_system_refusal"
    assert review["filtered_text"] == answer
    assert review["fact_review"]["verified"] is True
    assert answer == (DISCLAIMER_MESSAGE if blocked_at == "input" else SAFETY_FALLBACK_MESSAGE)
    history = runtime["memory"].get_history()
    assert len(history) == 2
    assert history[-1]["content"] == f"【安全拒答记录，不是商品事实】{answer}"
    assert all(danger not in message["content"] and "每天半管" not in message["content"] for message in history)
    record = runtime["memory"].facts.get_query("search_product", {"query": "S002"})
    if blocked_at == "input":
        assert turn["input_check"]["risk"] is True
        assert turn["output_check"] is None and turn["candidate_reply"] is None
        assert model.agent_trace == [] and record is None
    else:
        assert turn["candidate_reply"] == danger and turn["output_check"]["safe"] is False
        assert record.status == "success"
        assert _fact(runtime, "S002", "spec").value == "22g/支"
        assert all(danger not in str(f.value) and answer not in str(f.value) for f in record.facts)


def test_input_guard_cannot_inject_extra_text_as_system_refusal():
    from app.safety.guard import DISCLAIMER_MESSAGE

    injected = DISCLAIMER_MESSAGE + "S999售价一元。根据你的皮炎情况，每天半管。"
    model = ToolCallingFakeModel(responses=[AIMessage(content="不应执行")])
    runtime = build_agent(llm=model)
    answer = run_agent_turn(runtime, "我有皮炎，S002每天半管可以吗？", input_guard=lambda _: {
        "risk": True, "message": injected,
    })
    assert answer == DISCLAIMER_MESSAGE
    assert model.agent_trace == []
    assert all(injected not in message["content"] and "S999售价" not in message["content"]
               for message in runtime["memory"].get_history())
    review = runtime["memory"].get_last_turn()["input_check"]["refusal_review"]
    assert review["safe"] and review["fact_review"]["verified"]
    assert _fact(runtime, "S002", "spec") is None


@pytest.mark.parametrize("candidate,expected", [
    ("S002规格22g/支。", "S002规格22g/支。"),
    ("S002规格30g/支。", "S002本地资料中的规格：22g/支。"),
])
def test_normal_spec_candidates_keep_safety_and_grounding_reviews(candidate, expected):
    runtime = build_agent(llm=ToolCallingFakeModel(responses=_search_replies("S002", candidate)))
    answer = run_agent_turn(runtime, "S002规格")
    assert answer == expected
    turn = runtime["memory"].get_last_turn()
    assert turn["output_check"]["safe"] is True
    assert "refusal_review" not in turn["output_check"]
    assert runtime["memory"].get_history()[-1]["content"] == expected
    assert _fact(runtime, "S002", "spec").value == "22g/支"
    if candidate != expected:
        assert turn["output_check"]["replacement_fact_review"]["verified"]
        assert turn["output_check"]["replacement_safety_review"]["safe"]


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Mock LLM that supports LangChain tool binding and records execution."""

    agent_trace: list[str] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ToolCallingFakeModel":
        """Return self so LangChain Agent can bind tools during tests."""
        _ = tools, tool_choice, kwargs
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any) -> Any:
        """Record model execution before returning the next mocked response."""
        self.agent_trace.append("agent")
        return super()._generate(messages, stop=stop, **kwargs)


class HistoryPriceFakeModel(ToolCallingFakeModel):
    """Exercise history handoff, not real-model understanding or copy quality."""

    observed_messages: list[list[BaseMessage]] = Field(default_factory=list)
    unsafe_reply: str | None = None

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any) -> Any:
        self.observed_messages.append(list(messages))
        if self.unsafe_reply is not None:
            reply = AIMessage(content=self.unsafe_reply)
            self.unsafe_reply = None
        elif isinstance(messages[-1], ToolMessage):
            product = json.loads(messages[-1].content)["results"][0]
            reply = AIMessage(content=f"{product['sku_id']} 的价格是 {product['price']}。")
        else:
            # Resolve only from human context; assistant claims are not SKU evidence.
            sku_id = next(
                match[-1]
                for message in reversed(messages)
                if message.type == "human"
                and (match := re.findall(r"S\d{3,}", str(message.content)))
            )
            reply = AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_product", "args": {"query": sku_id},
                    "id": f"history_search_{len(self.observed_messages)}",
                }],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    """Collect ToolMessages from an Agent result."""
    return [message for message in messages if isinstance(message, ToolMessage)]


def test_normal_question_runs_agent_and_calls_tool() -> None:
    """Normal product consultation should run Agent and call search_product."""
    llm = ToolCallingFakeModel(
        responses=[
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
            AIMessage(content="S001 的价格是 176 元。"),
        ]
    )
    runtime = build_agent(llm=llm)

    answer = run_agent_turn(runtime, "S001多少钱？")
    result = runtime["last_result"]
    tool_messages = _tool_messages(result["messages"])
    tool_output = json.loads(tool_messages[0].content)

    assert answer == "S001 的价格是 176 元。"
    assert tool_messages
    assert tool_messages[0].name == "search_product"
    assert tool_output["results"][0]["sku_id"] == "S001"


def test_high_risk_user_input_is_blocked_before_agent_runs() -> None:
    """High-risk user input should be blocked before Agent or Tool execution."""
    trace: list[str] = []
    llm = ToolCallingFakeModel(responses=[AIMessage(content="不应该执行")], agent_trace=trace)
    runtime = build_agent(llm=llm)

    answer = run_agent_turn(runtime, "这个药能治好我的皮炎吗？")

    assert "请咨询医生或药师" in answer
    assert trace == []
    history = runtime["memory"].get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert "治好我的皮炎" not in history[0]["content"]
    assert history[1]["role"] == "assistant"
    assert answer in history[1]["content"]
    assert "不是商品事实" in history[1]["content"]
    turn = runtime["memory"].get_last_turn()
    assert turn["user_input"] == "这个药能治好我的皮炎吗？"
    assert turn["input_check"]["risk"] is True
    assert turn["candidate_reply"] is None
    assert turn["output_check"] is None
    assert turn["displayed_reply"] == answer


def test_unsafe_ai_output_is_blocked() -> None:
    """Unsafe AI output should not be returned directly to the user."""
    unsafe_text = "这个产品可以治疗皮炎，保证有效"
    llm = ToolCallingFakeModel(responses=[AIMessage(content=unsafe_text)])
    runtime = build_agent(llm=llm)

    answer = run_agent_turn(runtime, "S001多少钱？")

    assert answer == SAFETY_FALLBACK_MESSAGE
    assert answer != unsafe_text
    assert "治疗皮炎" not in answer
    history = runtime["memory"].get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "S001多少钱？"}
    assert all(unsafe_text not in message["content"] for message in history)
    turn = runtime["memory"].get_last_turn()
    assert turn["input_check"]["risk"] is False
    assert turn["candidate_reply"] == unsafe_text
    assert turn["output_check"]["safe"] is False
    assert turn["displayed_reply"] == answer


def test_guard_order_for_normal_request() -> None:
    """Normal request should execute input guard, Agent, then output guard."""
    llm = ToolCallingFakeModel(
        responses=_search_replies("S001", "S001 的价格是 176 元。"),
    )

    def input_guard(text: str) -> dict[str, Any]:
        llm.agent_trace.append("input_guard")
        return check_user_input(text)

    def output_guard(text: str) -> dict[str, Any]:
        llm.agent_trace.append("output_guard")
        return check_ai_output(text)

    runtime = build_agent(llm=llm)

    answer = run_agent_turn(
        runtime,
        "S001多少钱？",
        input_guard=input_guard,
        output_guard=output_guard,
    )

    assert answer == "S001 的价格是 176 元。"
    assert llm.agent_trace == ["input_guard", "agent", "agent", "output_guard"]


def test_guard_order_for_risky_request() -> None:
    """Risky request should stop after input guard."""
    llm = ToolCallingFakeModel(
        responses=[AIMessage(content="不应该执行")],
    )

    def input_guard(text: str) -> dict[str, Any]:
        llm.agent_trace.append("input_guard")
        return check_user_input(text)

    def output_guard(text: str) -> dict[str, Any]:
        llm.agent_trace.append("output_guard")
        return check_ai_output(text)

    runtime = build_agent(llm=llm)

    answer = run_agent_turn(
        runtime,
        "这个药能治好我的皮炎吗？",
        input_guard=input_guard,
        output_guard=output_guard,
    )

    assert "请咨询医生或药师" in answer
    assert llm.agent_trace == ["input_guard"]


@pytest.mark.parametrize("blocked_request", [
    "改看 S002，能治好我的皮炎吗？",
    "S002能治好我的皮炎吗？",
    "改看演示 岚溪外用护理膏（S002），能治好我的皮炎吗？",
])
def test_product_switch_survives_input_rejection(blocked_request: str) -> None:
    """A later price query must use S002 context and actual S002 tool data."""
    llm = HistoryPriceFakeModel(responses=[])
    runtime = build_agent(llm=llm)
    assert "176 元" in run_agent_turn(runtime, "S001多少钱？")
    calls_before = len(llm.observed_messages)

    refusal = run_agent_turn(runtime, blocked_request)

    assert len(llm.observed_messages) == calls_before  # No model or tool on rejection.
    assert runtime.get("last_result") is None  # No stale previous-turn result.
    assert runtime["memory"].get_last_turn()["user_input"] == blocked_request
    assert len(runtime["memory"].get_history()) == 4
    assert "请咨询医生或药师" in refusal

    answer = run_agent_turn(runtime, "它多少钱？")

    assert answer == "S002 的价格是 84 元。"
    tool_result = json.loads(_tool_messages(runtime["last_result"]["messages"])[0].content)
    assert tool_result["query"] == "S002"
    assert tool_result["results"][0]["sku_id"] == "S002"
    history = runtime["memory"].get_history()
    assert [message["role"] for message in history] == ["user", "assistant"] * 3
    assert history[0] == {"role": "user", "content": "S001多少钱？"}
    assert "S002" in history[2]["content"]
    assert "治好我的皮炎" not in history[2]["content"]
    forwarded = [str(message.content) for message in llm.observed_messages[-2]]
    assert "它多少钱？" in forwarded
    assert all("治好我的皮炎" not in content for content in forwarded)
    assert sum("S001多少钱？" == content for content in forwarded) == 1


def test_product_switch_survives_output_rejection_without_candidate_leak() -> None:
    llm = HistoryPriceFakeModel(responses=[])
    runtime = build_agent(llm=llm)
    run_agent_turn(runtime, "S001多少钱？")
    unsafe_text = "S001 可以治疗皮炎，保证有效，价格只要 1 元"
    llm.unsafe_reply = unsafe_text

    assert run_agent_turn(runtime, "改看 S002，先介绍一下") == SAFETY_FALLBACK_MESSAGE
    turn = runtime["memory"].get_last_turn()
    assert turn["candidate_reply"] == unsafe_text
    assert turn["output_check"]["safe"] is False
    assert turn["displayed_reply"] == SAFETY_FALLBACK_MESSAGE
    assert len(runtime["memory"].get_history()) == 4

    assert run_agent_turn(runtime, "它多少钱？") == "S002 的价格是 84 元。"
    forwarded = [str(message.content) for message in llm.observed_messages[-2]]
    assert unsafe_text not in forwarded
    assert all("价格只要 1 元" not in content for content in forwarded)
    assert len(runtime["memory"].get_history()) == 6
    assert all(unsafe_text not in message["content"] for message in runtime["memory"].get_history())


def test_repeated_input_rejections_append_exactly_one_pair_each() -> None:
    llm = ToolCallingFakeModel(responses=[AIMessage(content="不会调用")])
    runtime = build_agent(llm=llm)
    for size in (2, 4):
        run_agent_turn(runtime, "改看 S002，能治好皮炎吗？")
        assert len(runtime["memory"].get_history()) == size
    assert llm.agent_trace == []


@pytest.mark.parametrize(("blocked_request", "retained"), [
    ("改看演示 岚溪外用护理膏，能治好皮炎吗？", "改看演示 岚溪外用护理膏"),
    ("预算改成 100 元，S002 能治好皮炎吗？", "预算改成 100 元"),
    ("它能治好皮炎吗？", "它"),
])
def test_rejected_input_keeps_only_context_not_medical_request(
    blocked_request: str, retained: str,
) -> None:
    llm = ToolCallingFakeModel(responses=[AIMessage(content="不会调用")])
    runtime = build_agent(llm=llm)
    run_agent_turn(runtime, blocked_request)
    history = runtime["memory"].get_history()
    assert retained in history[0]["content"]
    assert "治好皮炎" not in history[0]["content"]
    assert "未核验，不是商品事实" in history[0]["content"]
    assert runtime["memory"].get_last_turn()["user_input"] == blocked_request
    assert llm.agent_trace == []


def test_only_filtered_output_is_saved_as_assistant_history() -> None:
    candidate = "S002 的价格是 84 元。额外说明待过滤。"
    filtered = "S002 的价格是 84 元。"
    llm = ToolCallingFakeModel(responses=_search_replies("S002", candidate))
    runtime = build_agent(llm=llm)

    answer = run_agent_turn(
        runtime, "S002多少钱？",
        output_guard=lambda text: {"safe": True, "filtered_text": filtered, "reason": ""},
    )

    assert answer == filtered
    assert runtime["memory"].get_history() == [
        {"role": "user", "content": "S002多少钱？"},
        {"role": "assistant", "content": filtered},
    ]
    turn = runtime["memory"].get_last_turn()
    assert turn["candidate_reply"] == candidate
    assert turn["displayed_reply"] == filtered
    assert turn["output_check"]["safe"] is True


def _search_replies(sku: str, candidate: str) -> list[AIMessage]:
    return [
        AIMessage(content="", tool_calls=[{
            "name": "search_product", "args": {"query": sku}, "id": f"safety_{sku}",
        }]),
        AIMessage(content=candidate),
    ]


def _fact(runtime: dict[str, Any], sku: str, field: str) -> Any:
    return runtime["memory"].facts.get_fact(sku, field, tool="search_product", args={"query": sku})


def test_actual_s002_taboo_is_displayed_without_losing_evidence() -> None:
    taboo = next(p["taboo"] for p in load_sku_data() if p["sku_id"] == "S002")
    candidate = "S002禁忌：" + taboo + "。"
    model = ToolCallingFakeModel(responses=_search_replies("S002", candidate))
    runtime = build_agent(llm=model)

    assert run_agent_turn(runtime, "S002有哪些禁忌？") == candidate
    memory = runtime["memory"]
    assert memory.get_last_turn()["output_check"]["safe"] is True
    assert memory.get_last_turn()["output_check"]["decision"] == "allow"
    assert memory.get_history()[-1]["content"] == candidate
    assert _fact(runtime, "S002", "taboo").value == taboo
    assert memory.get_state().current_sku == "S002"


@pytest.mark.parametrize("danger", [
    "肯定有效。",
    "虽然不能保证根治，但它肯定能治好皮炎。",
    "根据你的皮炎情况，每天涂三次。",
])
def test_new_output_risks_preserve_state_actual_facts_and_safe_history(danger: str) -> None:
    candidate = "S002禁忌：不可用于疾病治疗。S002价格1元。" + danger
    model = ToolCallingFakeModel(responses=[
        *_search_replies("S001", "S001价格176元。"),
        *_search_replies("S002", candidate),
        *_search_replies("S002", "S002价格84元。"),
    ])
    runtime = build_agent(llm=model)
    run_agent_turn(runtime, "S001多少钱，预算200元以内，目标是日常养护")
    assert run_agent_turn(runtime, "改看S002，介绍一下") == SAFETY_FALLBACK_MESSAGE
    memory = runtime["memory"]
    turn = memory.get_last_turn()
    assert turn["candidate_reply"] == candidate
    assert turn["output_check"]["safe"] is False
    assert turn["displayed_reply"] == SAFETY_FALLBACK_MESSAGE
    assert len(memory.get_history()) == 4
    assert "不是商品事实" in memory.get_history()[-1]["content"]
    assert all(candidate not in m["content"] for m in memory.get_history())
    state = memory.get_state()
    assert state.current_sku == "S002"
    assert state.budget_max == 200
    assert state.goal == "日常养护"
    assert _fact(runtime, "S001", "price").value == "176 元"
    assert _fact(runtime, "S002", "price").value == "84 元"
    assert "治疗" in _fact(runtime, "S002", "taboo").value  # Actual limitation, not the claim.
    assert not any(f.field in ("dosage", "efficacy") for f in memory.facts.get_query(
        "search_product", {"query": "S002"},
    ).facts)
    assert run_agent_turn(runtime, "它多少钱？") == "S002价格84元。"
    assert len(memory.get_history()) == 6
    assert _fact(runtime, "S002", "price").value == "84 元"


@pytest.mark.parametrize("dose_request", [
    "S002我有皮炎每天涂几次？",
    "改看S002，每天涂几次？",
    "改看S002，我怀孕了该怎么调整用量？",
])
def test_new_dosing_input_block_keeps_switch_without_updating_preferences_or_facts(dose_request: str) -> None:
    model = HistoryPriceFakeModel(responses=[])
    runtime = build_agent(llm=model)
    run_agent_turn(runtime, "S001多少钱，预算200元以内，目标是日常养护")
    before = len(model.observed_messages)
    reply = run_agent_turn(runtime, dose_request)
    assert "请咨询医生或药师" in reply
    assert len(model.observed_messages) == before
    assert runtime.get("last_result") is None
    memory = runtime["memory"]
    turn = memory.get_last_turn()
    assert turn["input_check"]["risk_type"] == "personal_medication_advice"
    assert turn["candidate_reply"] is None and turn["output_check"] is None
    assert memory.get_state().current_sku == "S002"
    assert memory.get_state().budget_max == 200
    assert memory.get_state().goal == "日常养护"
    assert _fact(runtime, "S002", "price") is None
    assert len(memory.get_history()) == 4
    assert "涂几次" not in memory.get_history()[2]["content"]
    assert "调整用量" not in memory.get_history()[2]["content"]
    assert run_agent_turn(runtime, "它多少钱？") == "S002 的价格是 84 元。"
    assert len(memory.get_history()) == 6


@pytest.fixture
def synthetic_label_source(tmp_path, monkeypatch):
    """Small temporary tool dataset, NOT a dosage claim about real S001."""
    from app.tools import data_loader

    data = load_sku_data()
    for row in data:
        if row["sku_id"] == "S001":
            row["product_desc"] = "标签用法：每日2次，每次1片。"
    path = tmp_path / "sku.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(data_loader, "SKU_DATA_PATH", path)
    return path


def _label_runtime(source, candidate, *, query_sku="S001"):
    from app.agent.facts import SessionFacts

    model = ToolCallingFakeModel(responses=_search_replies(query_sku, candidate))
    runtime = build_agent(llm=model)
    runtime["memory"].facts = SessionFacts(source_path=source)
    return runtime


def test_verified_label_directions_flow_from_executed_tool_into_output_guard(synthetic_label_source):
    candidate = "S001标签用法：每日2次，每次1片。"
    runtime = _label_runtime(synthetic_label_source, candidate)
    assert run_agent_turn(runtime, "请转述S001标签用法") == candidate
    memory = runtime["memory"]
    assert memory.get_last_turn()["output_check"]["safe"] is True
    fact = _fact(runtime, "S001", "product_desc")
    assert fact.value == "标签用法：每日2次，每次1片。"
    assert fact.source_path == str(synthetic_label_source)
    assert fact.source_version == memory.facts.source_version()
    assert memory.get_history()[-1]["content"] == candidate


@pytest.mark.parametrize("candidate", [
    "S001标签用法：每日3次，每次1片。",  # Altered source value.
    "S001标签用法：每日2次，每次1片。你可以加倍。",  # Unsupported personalization.
    "S001标签用法：每日2次，每次1片。保证有效。",  # Source is not a claim waiver.
    "S002标签用法：每日2次，每次1片。",  # Evidence belongs to a different SKU.
    "Ｓ００２标签用法：每日2次，每次1片。",  # Width-normalized identity must also match.
    "S\u200b002标签用法：每日2次，每次1片。",  # Invisible separators cannot change ownership.
])
def test_verified_label_never_authorizes_changes_additions_or_another_sku(synthetic_label_source, candidate):
    runtime = _label_runtime(synthetic_label_source, candidate)
    assert run_agent_turn(runtime, "请转述S001标签用法") == SAFETY_FALLBACK_MESSAGE
    assert runtime["memory"].get_last_turn()["output_check"]["safe"] is False
    assert _fact(runtime, "S001", "product_desc").value == "标签用法：每日2次，每次1片。"
    assert _fact(runtime, "S002", "product_desc") is None
    assert candidate not in [m["content"] for m in runtime["memory"].get_history()]


def test_label_evidence_must_belong_to_current_selection(synthetic_label_source, monkeypatch):
    from app.tools.sku_search import search_product

    # A conflicting SKU must be rejected before the local tool function runs.
    calls = []
    original_search = search_product.func

    def observed_search(query):
        calls.append(query)
        return original_search(query)

    monkeypatch.setattr(search_product, "func", observed_search)
    candidate = "S002标签用法：每日2次，每次1片。"
    runtime = _label_runtime(synthetic_label_source, candidate, query_sku="S001")
    assert run_agent_turn(runtime, "请转述S002标签用法") == SAFETY_FALLBACK_MESSAGE
    assert calls == []
    assert _fact(runtime, "S001", "product_desc") is None
    assert _fact(runtime, "S002", "product_desc") is None
    message = next(m for m in runtime["last_result"]["messages"] if isinstance(m, ToolMessage))
    check = json.loads(message.content)["_task_check"]
    assert message.status == "error" and check["decision"] == "replan"
    assert check["reason"] == "sku_conflict"
    assert check["proposed_args"] == {"query": "S001"} and check["execution_args"] is None
    record = runtime["memory"].facts.get_query("search_product", {"query": "S001"})
    assert record.status == "error" and not record.facts
    assert all(candidate not in m["content"] for m in runtime["memory"].get_history())


def test_claimed_label_without_executed_evidence_requires_verification():
    # The real project data contains no label doses; model attribution isn't proof.
    candidate = "S001标签用法：每日2次，每次1片。"
    runtime = build_agent(llm=ToolCallingFakeModel(responses=[AIMessage(content=candidate)]))
    assert run_agent_turn(runtime, "请转述S001标签用法") == SAFETY_FALLBACK_MESSAGE
    decision = runtime["memory"].get_last_turn()["output_check"]
    assert decision["decision"] == "clarify"
    assert decision["risk_types"] == ["unverified_dosage"]
    assert _fact(runtime, "S001", "product_desc") is None


def test_label_safety_evidence_is_session_local_and_does_not_survive_reset(synthetic_label_source):
    candidate = "S001标签用法：每日2次，每次1片。"
    first = _label_runtime(synthetic_label_source, candidate)
    assert run_agent_turn(first, "请转述S001标签用法") == candidate
    second = build_agent(llm=ToolCallingFakeModel(responses=[AIMessage(content=candidate)]))
    assert run_agent_turn(second, "请转述S001标签用法") == SAFETY_FALLBACK_MESSAGE
    assert _fact(second, "S001", "product_desc") is None
    first["memory"].clear()
    after_reset = build_agent(
        llm=ToolCallingFakeModel(responses=[AIMessage(content=candidate)]), memory=first["memory"],
    )
    assert run_agent_turn(after_reset, "请转述S001标签用法") == SAFETY_FALLBACK_MESSAGE
    assert _fact(after_reset, "S001", "product_desc") is None


@pytest.mark.parametrize("dose", [
    "每天涂半支", "每周用一支", "每天半管", "每天一管",
    "用半管", "涂一管", "每次半支", "一天两次",
])
def test_counted_dose_input_and_candidate_never_reach_normal_history(dose):
    model = HistoryPriceFakeModel(responses=[])
    runtime = build_agent(llm=model)
    run_agent_turn(runtime, "S001多少钱？")
    before = len(model.observed_messages)
    question = "我有皮炎，S002" + dose + "可以吗？"
    answer = run_agent_turn(runtime, question)
    assert "请咨询医生或药师" in answer
    assert len(model.observed_messages) == before
    assert runtime["memory"].get_last_turn()["candidate_reply"] is None
    assert _fact(runtime, "S002", "price") is None
    assert runtime["memory"].get_state().current_sku == "S002"
    assert all(dose not in m["content"] for m in runtime["memory"].get_history())
    assert run_agent_turn(runtime, "它多少钱？") == "S002 的价格是 84 元。"
    assert len(runtime["memory"].get_history()) == 6

    candidate = "根据你的皮炎情况，" + dose + "。"
    rt = build_agent(llm=ToolCallingFakeModel(responses=_search_replies("S002", candidate)))
    assert run_agent_turn(rt, "S002规格") == SAFETY_FALLBACK_MESSAGE
    assert rt["memory"].get_last_turn()["candidate_reply"] == candidate
    assert all(candidate not in m["content"] for m in rt["memory"].get_history())
    assert _fact(rt, "S002", "price").value == "84 元"


def test_half_tube_candidate_is_blocked_after_actual_tool_query():
    candidate = "根据你的皮炎情况，S002每天半管。"
    rt = build_agent(llm=ToolCallingFakeModel(responses=_search_replies("S002", candidate)))
    answer = run_agent_turn(rt, "S002规格")
    memory = rt["memory"]
    turn = memory.get_last_turn()
    assert answer == SAFETY_FALLBACK_MESSAGE and "每天半管" not in answer
    assert turn["candidate_reply"] == candidate
    assert turn["displayed_reply"] == answer and turn["output_check"]["safe"] is False
    assert "personalized_or_treatment_dosage" in turn["output_check"]["risk_types"]
    assert all(candidate not in m["content"] for m in memory.get_history())
    assert memory.get_history()[-1]["content"] == f"【安全拒答记录，不是商品事实】{answer}"
    record = memory.facts.get_query("search_product", {"query": "S002"})
    assert record.status == "success"
    assert _fact(rt, "S002", "spec").value == "22g/支"
    assert all(candidate not in str(f.value) and answer not in str(f.value) for f in record.facts)


def test_half_tube_in_replacement_is_independently_reviewed_before_history():
    candidate = "S002规格30g/支。"
    injected = "根据你的皮炎情况，S002每天半管。"
    rt = build_agent(llm=ToolCallingFakeModel(responses=_search_replies("S002", candidate)))
    reviews = []

    def filter_output(text):
        reviews.append(text)
        return {"safe": True, "filtered_text": text if len(reviews) == 1 else injected}

    answer = run_agent_turn(rt, "S002规格", output_guard=filter_output)
    turn = rt["memory"].get_last_turn()
    assert len(reviews) == 2 and "22g/支" in reviews[1]
    assert answer == SAFETY_FALLBACK_MESSAGE
    assert turn["candidate_reply"] == candidate and turn["displayed_reply"] == answer
    assert turn["output_check"]["replacement_safety_review"]["safe"] is False
    assert "replacement_fact_review" in turn["output_check"]
    assert turn["output_check"]["safe"] is False
    history = rt["memory"].get_history()
    assert all(candidate not in m["content"] and injected not in m["content"] for m in history)
    assert history[-1]["content"] == f"【安全拒答记录，不是商品事实】{answer}"
    record = rt["memory"].facts.get_query("search_product", {"query": "S002"})
    assert _fact(rt, "S002", "spec").value == "22g/支"
    assert all(injected not in str(f.value) and answer not in str(f.value) for f in record.facts)


@pytest.mark.parametrize("dose", ["每周用半支", "每天半管", "每次用一管"])
def test_counted_label_from_actual_temporary_tool_source(tmp_path, monkeypatch, dose):
    from app.tools import data_loader

    data = load_sku_data()
    for row in data:
        if row["sku_id"] == "S001":
            row["product_desc"] = f"标签用法：{dose}。"
    source = tmp_path / "label.json"
    source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(data_loader, "SKU_DATA_PATH", source)
    candidate = f"S001标签用法：{dose}。"
    rt = _label_runtime(source, candidate)
    assert run_agent_turn(rt, "请转述S001标签用法") == candidate
    assert rt["memory"].get_history()[-1]["content"] == candidate
