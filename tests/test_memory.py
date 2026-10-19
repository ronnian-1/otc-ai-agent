"""Tests for lightweight session memory and Agent memory integration.

These tests verify basic in-process conversation state management without
calling a real LLM, adding long-term memory, or integrating Safety Guard.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import Field

from app.agent.agent import build_agent, run_agent_turn
from app.agent.memory import SessionMemory
from app.agent.state import SessionState


class MemoryAwareFakeModel(FakeMessagesListChatModel):
    """Mock LLM that records messages passed by the Agent."""

    observed_message_contents: list[list[str]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "MemoryAwareFakeModel":
        """Return self so LangChain Agent can bind tools during tests."""
        _ = tools, tool_choice, kwargs
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any) -> Any:
        """Capture message contents before returning the next fake response."""
        self.observed_message_contents.append([str(message.content) for message in messages])
        return super()._generate(messages, stop=stop, **kwargs)


def test_session_memory_adds_and_returns_history() -> None:
    """SessionMemory should store user and assistant messages."""
    memory = SessionMemory()

    memory.add_message("user", "S001多少钱？")
    memory.add_message("assistant", "价格是176元")

    assert memory.get_history() == [
        {"role": "user", "content": "S001多少钱？"},
        {"role": "assistant", "content": "价格是176元"},
    ]


def test_agent_reads_memory_on_second_turn() -> None:
    """Second Agent turn should include previous user and assistant messages."""
    memory = SessionMemory()
    llm = MemoryAwareFakeModel(
        responses=[
            AIMessage(content="", tool_calls=[{
                "name": "search_product", "args": {"query": "S001"}, "id": "memory_price",
            }]),
            AIMessage(content="价格是176元"),
            AIMessage(content="", tool_calls=[{
                "name": "search_product", "args": {"query": "S001"}, "id": "memory_crowd",
            }]),
            AIMessage(content="它适合独立虚构场景"),
        ]
    )
    runtime = build_agent(llm=llm, memory=memory)

    first_answer = run_agent_turn(runtime, "S001多少钱？")
    second_answer = run_agent_turn(runtime, "它适合谁？")

    assert first_answer == "价格是176元"
    assert "S001本地资料中的适用人群：经常熬夜、轮班、希望记录作息变化的成年演示人群" in second_answer
    assert len(memory.get_history()) == 4

    second_turn_messages = llm.observed_message_contents[-1]
    assert "S001多少钱？" in second_turn_messages
    assert "价格是176元" in second_turn_messages
    assert "它适合谁？" in second_turn_messages


def test_session_memory_clear_removes_history() -> None:
    """SessionMemory.clear should remove all messages."""
    memory = SessionMemory()
    memory.add_message("user", "S001多少钱？")
    memory.add_message("assistant", "价格是176元")

    memory.clear()

    assert memory.get_history() == []


def test_latest_turn_diagnostics_are_separate_copied_and_cleared() -> None:
    """Diagnostic raw/candidate text must not become model context or survive clear."""
    memory = SessionMemory()
    input_check = {"risk": True, "risk_type": "medical_consultation"}
    memory.add_turn(
        user_input="原始风险请求",
        input_check=input_check,
        candidate_reply=None,
        output_check=None,
        displayed_reply="请咨询医生或药师",
        history_user_input="用户改看 S002；仅为未核验表述",
    )
    input_check["risk"] = False
    record = memory.get_last_turn()
    assert record["input_check"]["risk"] is True
    record["input_check"]["risk"] = False
    assert memory.get_last_turn()["input_check"]["risk"] is True
    assert all("原始风险请求" not in message["content"] for message in memory.get_history())
    memory.clear()
    assert memory.get_history() == []
    assert memory.get_last_turn() is None


def _query_responses(sku: str, candidate: str) -> list[AIMessage]:
    return [AIMessage(content="", tool_calls=[{
        "name": "search_product", "args": {"query": sku}, "id": "reset_" + sku,
    }]), AIMessage(content=candidate)]


def _assert_session_cleared(runtime, persistent, queried_skus):
    # Compare every runtime key against its pre-session inventory, not just one
    # trace field. Dependencies/configuration must retain their object identity.
    assert set(runtime) == set(persistent)
    for key, value in persistent.items():
        assert runtime[key] is value
    memory = runtime["memory"]
    assert memory.get_history() == []
    assert memory.get_last_turn() is None
    assert memory.get_state() == SessionState()
    for sku in queried_skus:
        assert memory.facts.get_query("search_product", {"query": sku}) is None
        assert memory.facts.valid_query_facts("search_product", {"query": sku}) == ()


def test_memory_clear_removes_all_runtime_diagnostics_before_next_turn():
    model = MemoryAwareFakeModel(responses=[
        *_query_responses("S001", "S001价格176元。"),
        *_query_responses("S002", "S002价格84元。"),
    ])
    runtime = build_agent(llm=model)
    persistent = dict(runtime)
    run_agent_turn(runtime, "S001多少钱，预算200元以内")
    before = {key: value for key, value in runtime.items() if key not in persistent}
    assert set(before) == {"last_result", "last_reply_plan", "last_fact_check"}
    assert all(before.values())
    assert any(isinstance(m, ToolMessage) for m in before["last_result"]["messages"])
    assert runtime["memory"].facts.get_fact(
        "S001", "price", tool="search_product", args={"query": "S001"},
    ).value == "176 元"

    runtime["memory"].clear()
    _assert_session_cleared(runtime, persistent, ["S001"])
    observed = len(model.observed_message_contents)
    answer = run_agent_turn(runtime, "它多少钱")
    assert "确认" in answer and "SKU" in answer
    assert len(model.observed_message_contents) == observed
    assert runtime.get("last_result") is None
    assert run_agent_turn(runtime, "S002多少钱") == "S002价格84元。"
    forwarded = model.observed_message_contents[observed:]
    assert all("S001" not in text and "176" not in text for turn in forwarded for text in turn)
    assert runtime["memory"].get_state().budget_max is None
    assert runtime["memory"].facts.get_query("search_product", {"query": "S001"}) is None


@pytest.mark.parametrize("candidate", [
    "根据你的皮炎情况，每周用一支。价格只要1元。",
    "S002规格三十克/支。",
])
def test_clear_removes_rejected_candidate_from_all_owned_diagnostics(candidate):
    model = MemoryAwareFakeModel(responses=_query_responses("S002", candidate))
    runtime = build_agent(llm=model)
    persistent = dict(runtime)
    answer = run_agent_turn(runtime, "S002规格")
    memory = runtime["memory"]
    assert candidate not in answer
    assert all(candidate not in message["content"] for message in memory.get_history())
    assert memory.get_last_turn()["candidate_reply"] == candidate
    assert any(str(m.content) == candidate for m in runtime["last_result"]["messages"])
    record = memory.facts.get_query("search_product", {"query": "S002"})
    assert record is not None and candidate not in str(record)
    assert next(f.value for f in record.facts if f.field == "spec") == "22g/支"
    assert next(f.value for f in record.facts if f.field == "price") == "84 元"

    memory.clear()
    _assert_session_cleared(runtime, persistent, ["S002"])
    # The scripted response/observer belongs to the injected test model, which
    # must not be destroyed. No application-owned diagnostic remains reachable.
    assert not any(key.startswith("last_") for key in runtime)


def test_repeated_clear_preserves_runtime_and_new_query_executes_tools_again():
    model = MemoryAwareFakeModel(responses=[
        *_query_responses("S001", "S001价格176元。"),
        *_query_responses("S001", "S001价格176元。"),
    ])
    runtime = build_agent(llm=model)
    persistent = dict(runtime)
    runtime["memory"].clear()  # Ready runtime, no turn yet.
    _assert_session_cleared(runtime, persistent, [])
    run_agent_turn(runtime, "S001多少钱")
    old_trace = runtime["last_result"]
    runtime["memory"].clear()
    runtime["memory"].clear()
    _assert_session_cleared(runtime, persistent, ["S001"])

    assert run_agent_turn(runtime, "S001多少钱") == "S001价格176元。"
    assert runtime["last_result"] is not old_trace
    tool_message = next(m for m in runtime["last_result"]["messages"] if isinstance(m, ToolMessage))
    assert json.loads(tool_message.content)["_tool_evidence"]["reused"] is False
    assert len(runtime["memory"].get_history()) == 2
    runtime["memory"].clear()
    _assert_session_cleared(runtime, persistent, ["S001"])


def test_clearing_one_runtime_leaves_other_session_diagnostics_and_facts_intact():
    first = build_agent(llm=MemoryAwareFakeModel(responses=_query_responses("S001", "S001价格176元。")))
    second = build_agent(llm=MemoryAwareFakeModel(responses=[
        *_query_responses("S002", "S002价格84元。"),
        *_query_responses("S002", "S002价格84元。"),
    ]))
    first_persistent, second_persistent = dict(first), dict(second)
    assert first["memory"] is not second["memory"]
    run_agent_turn(first, "S001多少钱，预算100元以内")
    run_agent_turn(second, "S002多少钱，预算200元以内")
    second_diagnostics = {k: deepcopy(v) for k, v in second.items() if k not in second_persistent}
    second_history = second["memory"].get_history()
    second_state = second["memory"].get_state()
    second_turn = second["memory"].get_last_turn()
    second_fact = second["memory"].facts.get_query("search_product", {"query": "S002"})

    first["memory"].clear()
    _assert_session_cleared(first, first_persistent, ["S001"])
    assert {k: second[k] for k in second_diagnostics} == second_diagnostics
    assert second["memory"].get_history() == second_history
    assert second["memory"].get_state() == second_state
    assert second["memory"].get_last_turn() == second_turn
    assert second["memory"].facts.get_query("search_product", {"query": "S002"}) == second_fact
    assert run_agent_turn(second, "它多少钱") == "S002价格84元。"
