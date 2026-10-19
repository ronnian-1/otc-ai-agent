"""Portable execution regressions: scripted models and real public-turn control.

Atomic reservation is a supplemental local concurrency check; it is not a full
Agent replay. Blocking callbacks always have release paths and bounded waits.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from app.agent.agent import build_agent, run_agent_turn, search_product, compare_product
from app.agent.execution import TimeoutPolicy, _Budget, MAX_MODEL_CALLS, MAX_TOOL_CALLS
from app.agent.grounding import current_tool_facts


class SequenceModel(FakeMessagesListChatModel):
    observed: list = Field(default_factory=list)
    gate: object = None
    entered: object = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.observed.append(list(messages))
        if self.gate is not None:
            self.entered.set()
            assert self.gate.wait(4), "controlled model was not released"
        return super()._generate(messages, **kwargs)


def call(name, args, ident):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident}])


def lookup(sku, answer):
    return [call("search_product", {"query": sku}, "lookup_" + sku), AIMessage(content=answer)]


def reset(model, responses):
    model.responses = responses
    model.i = 0
    model.observed.clear()


def test_error_success_error_stops_next_model_and_isolates_only_touched_facts(monkeypatch):
    actual = []
    search = search_product.func

    def searched(query):
        actual.append("search_product")
        return search(query)

    def failed(sku_ids):
        actual.append("compare_product")
        raise RuntimeError("controlled local comparison failure")

    monkeypatch.setattr(search_product, "func", searched)
    monkeypatch.setattr(compare_product, "func", failed)
    model = SequenceModel(responses=lookup("S002", "S002价格84元。"))
    rt = build_agent(llm=model)
    assert "84" in run_agent_turn(rt, "S002多少钱？")
    unrelated = rt["memory"].facts.get_query("search_product", {"query": "S002"})
    actual.clear()
    reset(model, [call("compare_product", {"sku_ids": "S001,S002"}, "failure1"),
                  call("search_product", {"query": "S001"}, "success"),
                  call("compare_product", {"sku_ids": "S001,S002"}, "failure2"),
                  AIMessage(content="UNREACHABLE")])
    answer = run_agent_turn(rt, "比较S001和S002")
    execution = rt["last_result"]["execution"]
    assert actual == ["compare_product", "search_product", "compare_product"]
    assert execution["tool_errors"] == 2 and execution["invalid_calls"] == 0
    assert execution["reason"] == "tool_error_budget" and execution["model_calls"] == 3
    assert len(model.observed) == 3 and "已停止" in answer
    assert rt["memory"].facts.get_query("search_product", {"query": "S001"}) is None
    assert rt["memory"].facts.get_query("compare_product", {"sku_ids": "S001,S002"}) is None
    assert rt["memory"].facts.get_query("search_product", {"query": "S002"}) == unrelated
    assert not current_tool_facts(rt)
    assert all("UNREACHABLE" not in item["content"] for item in rt["memory"].get_history())
    reset(model, lookup("S001", "S001价格176元。"))
    assert "176" in run_agent_turn(rt, "重新查S001价格")
    assert not rt["last_result"]["execution"]["stopped"]


def test_tool_timeout_discards_late_result_and_recovers(monkeypatch):
    entered, release = Event(), Event()
    search = search_product.func

    def delayed(query):
        if query == "S001":
            entered.set()
            assert release.wait(4), "controlled tool was not released"
        return search(query)

    monkeypatch.setattr(search_product, "func", delayed)
    model = SequenceModel(responses=lookup("S001", "S001价格176元。"))
    rt = build_agent(llm=model, timeouts=TimeoutPolicy(tool_seconds=0.15, model_seconds=3, turn_seconds=5, tool_workers=1))
    try:
        answer = run_agent_turn(rt, "S001多少钱？")
        assert entered.is_set() and rt["last_result"]["execution"]["reason"] == "tool_timeout"
        assert len(model.observed) == 1
        history = rt["memory"].get_history()
        assert rt["memory"].facts.get_query("search_product", {"query": "S001"}) is None
        assert "176" not in answer
    finally:
        release.set()
        assert rt["execution_policy"].workers["tool"].idle.wait(3)
    assert rt["memory"].get_history() == history
    assert rt["memory"].facts.get_query("search_product", {"query": "S001"}) is None
    reset(model, lookup("S002", "S002价格84元。"))
    assert run_agent_turn(rt, "改看S002价格") == "S002价格84元。"
    assert not rt["last_result"]["execution"]["stopped"]


def test_model_timeout_retains_capacity_rejects_busy_and_discards_late_reply():
    entered, release = Event(), Event()
    model = SequenceModel(responses=[AIMessage(content="LATE_UNSUPPORTED_CLAIM")], gate=release, entered=entered)
    rt = build_agent(llm=model, timeouts=TimeoutPolicy(tool_seconds=2, model_seconds=0.15, turn_seconds=5))
    try:
        answer = run_agent_turn(rt, "S001多少钱？")
        assert entered.is_set() and rt["last_result"]["execution"]["reason"] == "model_timeout"
        assert "LATE_UNSUPPORTED_CLAIM" not in answer
        calls = len(model.observed)
        run_agent_turn(rt, "S002多少钱？")
        assert rt["last_result"]["execution"]["reason"] == "model_busy"
        assert len(model.observed) == calls
        history = rt["memory"].get_history()
    finally:
        release.set()
        assert rt["execution_policy"].workers["model"].idle.wait(3)
    assert rt["memory"].get_history() == history
    assert all("LATE_UNSUPPORTED_CLAIM" not in item["content"] for item in history)
    model.gate = model.entered = None
    reset(model, lookup("S002", "S002价格84元。"))
    assert run_agent_turn(rt, "重新查S002价格") == "S002价格84元。"


def test_same_runtime_concurrent_reentry_never_commits_another_turn():
    entered, release = Event(), Event()
    model = SequenceModel(responses=[AIMessage(content="这里只核对商品资料。")], gate=release, entered=entered)
    rt = build_agent(llm=model, timeouts=TimeoutPolicy(model_seconds=3, turn_seconds=5))
    errors, answers = [], []

    def first():
        try:
            answers.append(run_agent_turn(rt, "S001介绍一下"))
        except BaseException as exc:
            errors.append(type(exc).__name__)

    thread = Thread(target=first)
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="Another turn is active"):
            run_agent_turn(rt, "S002多少钱？")
        assert not rt["memory"].get_history()
    finally:
        release.set()
        thread.join(4)
    assert not thread.is_alive() and not errors and len(answers) == 1
    assert len(rt["memory"].get_history()) == 2
    assert rt["memory"].get_state().current_sku == "S001"


def test_repeated_model_requests_stop_at_model_budget():
    model = SequenceModel(responses=[call("search_product", {"query": "S001"}, "loop")])
    rt = build_agent(llm=model)
    answer = run_agent_turn(rt, "S001多少钱？")
    assert rt["last_result"]["execution"]["reason"] == "model_budget"
    assert len(model.observed) == MAX_MODEL_CALLS
    assert "已停止" in answer


def test_atomic_reservations_are_bounded_under_parallel_attempts():
    # Supplemental local budget test, not a public-entry Agent replay.
    budget = _Budget()
    with ThreadPoolExecutor(max_workers=6) as workers:
        admitted = list(workers.map(lambda i: budget.reserve_tool(
            {"name": "search_product", "args": {"query": "S001"}, "id": str(i)}), range(30)))
    assert sum(admitted) == MAX_TOOL_CALLS
    assert budget.summary()["tool_calls"] == MAX_TOOL_CALLS
    assert budget.summary()["reason"] == "tool_budget"
