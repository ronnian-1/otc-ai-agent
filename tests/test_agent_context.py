"""Main-flow integration: deterministic fake LLM + actual LangChain/local tools.

These tests exercise the execution middleware and reuse; they do not demonstrate
that an external model will reliably follow provenance instructions.
"""

import importlib
import json
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.agent.agent import STATE_CONTEXT_PREFIX, build_agent, run_agent_turn
from app.agent.facts import SessionFacts
from app.agent.state import SessionState


class EvidenceModel(FakeMessagesListChatModel):
    """Read explicit state; decide calls deterministically and echo actual fields."""

    payloads: list[dict[str, Any]] = Field(default_factory=list)
    observed_contexts: list[str] = Field(default_factory=list)
    reply_once: str | None = None
    query_override: str | None = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
        text = next(str(m.content) for m in reversed(messages)
                    if m.type == "system" and str(m.content).startswith(STATE_CONTEXT_PREFIX))
        self.observed_contexts.append(text)
        state = json.loads(text.split("\n")[1])
        if self.reply_once is not None:
            reply = AIMessage(content=self.reply_once)
            self.reply_once = None
        elif isinstance(messages[-1], ToolMessage):
            result = json.loads(messages[-1].content)
            self.payloads.append(result)
            if "error" in result:
                reply = AIMessage(content="未能查询商品资料。")
            elif messages[-1].name == "compare_product":
                reply = AIMessage(content="已比较：" + "、".join(result["sku_ids"]))
            elif result.get("results"):
                row = result["results"][0]
                reply = AIMessage(content=f"{row['sku_id']}：{row.get(state['query_field'], '字段缺失')}")
            else:
                reply = AIMessage(content="没有查到该商品资料。")
        else:
            if state["current_sku"] and state["query_field"]:
                name, args = "search_product", {"query": self.query_override or state["current_sku"]}
            elif state["comparison_skus"] and state["current_sku"] is None:
                name, args = "compare_product", {"sku_ids": ",".join(state["comparison_skus"])}
            else:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="已记录需求。"))])
            reply = AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": f"facts_call_{len(self.observed_contexts)}",
            }])
        return ChatResult(generations=[ChatGeneration(message=reply)])


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def runtime():
    model = EvidenceModel(responses=[])
    return build_agent(llm=model), model


def fact(rt, sku, field, *, tool="search_product", args=None):
    return rt["memory"].facts.get_fact(sku, field, tool=tool,
                                     args=args if args is not None else {"query": sku})


def watch_search(monkeypatch):
    module = importlib.import_module("app.tools.sku_search")
    original = module.load_sku_data
    calls = []

    def load():
        calls.append(True)
        return original()

    monkeypatch.setattr(module, "load_sku_data", load)
    return module, original, calls


def test_switching_sku_reads_its_own_fields_then_reuses_only_matching_query(monkeypatch):
    _, _, calls = watch_search(monkeypatch)
    rt, model = runtime()
    assert run_agent_turn(rt, "S001多少钱") == "S001：176 元"
    assert run_agent_turn(rt, "S002多少钱") == "S002：84 元"
    assert run_agent_turn(rt, "它的规格") == "S002：22g/支"
    assert len(calls) == 2  # Third turn reads verified S002 results via middleware.
    assert model.payloads[-1]["_tool_evidence"]["reused"] is True
    assert fact(rt, "S001", "price").value == "176 元"
    assert fact(rt, "S002", "price").value == "84 元"
    assert fact(rt, "S002", "spec").value == "22g/支"
    assert fact(rt, "S002", "price", args={"query": "S001"}) is None
    assert fact(rt, "S001", "spec", args={"query": "S002"}) is None
    assert len(rt["memory"].get_history()) == 6
    assert all("_tool_evidence" not in m["content"] for m in rt["memory"].get_history())


def test_reuse_preserves_observation_time_and_pairs_new_tool_call_id(monkeypatch):
    _, _, calls = watch_search(monkeypatch)
    rt, model = runtime()
    run_agent_turn(rt, "S001价格")
    original = model.payloads[-1]["_tool_evidence"]
    run_agent_turn(rt, "它多少钱")
    evidence = model.payloads[-1]["_tool_evidence"]
    assert len(calls) == 1
    assert evidence["reused"] and not original["reused"]
    assert evidence["observed_at"] == original["observed_at"]
    assert evidence["expires_at"] == original["expires_at"]
    messages = rt["last_result"]["messages"]
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    call = next(m.tool_calls[0] for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    assert tool_message.tool_call_id == call["id"]


def test_unknown_sku_and_tool_exception_do_not_inherit_previous_product(monkeypatch):
    module, _, _ = watch_search(monkeypatch)
    rt, model = runtime()
    run_agent_turn(rt, "S001价格")
    assert "没有查到" in run_agent_turn(rt, "S999价格")
    assert fact(rt, "S999", "price") is None
    assert fact(rt, "S999", "spec") is None
    assert rt["memory"].facts.get_query("search_product", {"query": "S999"}).status == "empty"

    def broken():
        raise RuntimeError("local test failure")

    monkeypatch.setattr(module, "load_sku_data", broken)
    # A new tool failure is not replaced by an old successful response.
    run_agent_turn(rt, "S001价格，重新查询")
    assert "error" in model.payloads[-1]
    assert fact(rt, "S001", "price") is None
    assert rt["memory"].facts.get_query("search_product", {"query": "S001"}).status == "error"


def test_missing_taboo_field_is_not_invented_or_reused_as_complete(monkeypatch):
    module, original, calls = watch_search(monkeypatch)

    def incomplete():
        calls.append(True)
        return [{k: v for k, v in row.items() if k != "taboo"} for row in original()]

    monkeypatch.setattr(module, "load_sku_data", incomplete)
    rt, model = runtime()
    run_agent_turn(rt, "S001价格")
    assert fact(rt, "S001", "price").value == "176 元"
    assert fact(rt, "S001", "taboo") is None
    assert model.payloads[-1]["_tool_evidence"]["status"] == "partial"
    assert {"sku": "S001", "field": "taboo"} in model.payloads[-1]["_tool_evidence"]["missing_fields"]
    run_agent_turn(rt, "它的禁忌")
    assert len(calls) == 2
    assert not model.payloads[-1]["_tool_evidence"]["reused"]
    assert fact(rt, "S001", "taboo") is None


def test_fuzzy_search_is_scoped_and_always_requeried(monkeypatch):
    _, _, calls = watch_search(monkeypatch)
    rt, model = runtime()
    model.query_override = "星屿"
    run_agent_turn(rt, "S001价格")
    run_agent_turn(rt, "它多少钱")
    assert len(calls) == 2
    scoped = fact(rt, "S001", "price", args={"query": "星屿"})
    assert scoped.value == "176 元"
    assert fact(rt, "S001", "price") is None
    assert fact(rt, "S001", "price", args={"query": ""}) is None
    assert fact(rt, "S002", "price", args={"query": "星屿"}) is None
    assert not model.payloads[-1]["_tool_evidence"]["reused"]


def test_comparison_records_real_order_and_field_ownership(monkeypatch):
    module = importlib.import_module("app.tools.compare")
    original = module.load_sku_data
    calls = []

    def load():
        calls.append(True)
        return original()

    monkeypatch.setattr(module, "load_sku_data", load)
    rt, model = runtime()
    for first, second in (("S002", "S001"), ("S001", "S002")):
        run_agent_turn(rt, f"比较{first}和{second}")
        assert model.payloads[-1]["sku_ids"] == [first, second]
        args = {"sku_ids": f"{first},{second}"}
        assert fact(rt, "S001", "price", tool="compare_product", args=args).value == "176 元"
        assert fact(rt, "S002", "price", tool="compare_product", args=args).value == "84 元"
        assert fact(rt, "S002", "spec", tool="compare_product", args=args).value == "22g/支"
        taboo = fact(rt, "S002", "taboo", tool="compare_product", args=args)
        assert taboo.value == "演示限制：仅供外用类别展示，出现皮肤不适时请向专业人士求助；不得用于疾病治疗"
        assert "summary" not in [f.field for f in rt["memory"].facts.get_query("compare_product", args).facts]
    run_agent_turn(rt, "比较S001和S002")
    assert len(calls) == 3  # No comparison-response cache or inferred reordering.


def test_model_price_user_claim_and_rejected_candidate_are_not_tool_facts():
    rt, model = runtime()
    model.reply_once = "S001价格是1元"
    answer = run_agent_turn(rt, "S001价格，我看到是2元")
    assert "无法确认" in answer and "1元" not in answer  # P1-3 also protects display.
    assert "1元" not in rt["memory"].get_history()[-1]["content"]
    assert fact(rt, "S001", "price") is None
    assert rt["memory"].facts.get_query("search_product", {"query": "S001"}) is None
    model.reply_once = "S001价格是3元，能治疗皮炎"
    run_agent_turn(rt, "S001多少钱")
    assert rt["memory"].get_last_turn()["output_check"]["safe"] is False
    assert fact(rt, "S001", "price") is None
    assert all("价格是3元" not in m["content"] for m in rt["memory"].get_history())
    assert run_agent_turn(rt, "它多少钱") == "S001：176 元"
    assert fact(rt, "S001", "price").value == "176 元"


def test_rejected_input_keeps_switch_but_writes_no_new_facts_or_medical_claims():
    rt, model = runtime()
    run_agent_turn(rt, "S001多少钱")
    before = len(model.payloads)
    run_agent_turn(rt, "改看S002，能治疗皮炎吗")
    assert len(model.payloads) == before
    assert fact(rt, "S002", "price") is None
    assert rt["memory"].get_state().current_sku == "S002"
    assert run_agent_turn(rt, "它多少钱") == "S002：84 元"
    assert fact(rt, "S002", "price").value == "84 元"
    assert len(rt["memory"].get_history()) == 6


def test_session_isolation_and_clear_remove_history_state_and_facts(monkeypatch):
    _, _, calls = watch_search(monkeypatch)
    first, _ = runtime()
    second, _ = runtime()
    run_agent_turn(first, "S001价格")
    assert fact(second, "S001", "price") is None
    run_agent_turn(second, "S001价格")
    assert len(calls) == 2
    first["memory"].clear()
    assert first["memory"].get_history() == []
    assert first["memory"].get_last_turn() is None
    assert first["memory"].get_state() == SessionState()
    assert fact(first, "S001", "price") is None
    assert first["memory"].facts.get_query("search_product", {"query": "S001"}) is None
    assert fact(second, "S001", "price").value == "176 元"
    run_agent_turn(first, "S001价格")
    assert len(calls) == 3  # Middleware still refers to the cleared session store.


@pytest.mark.parametrize("field", ["库存", "发货方式", "实时价格", "最新价格", "价格，重新查询"])
def test_live_sensitive_or_explicit_freshness_requests_execute_again(monkeypatch, field):
    _, _, calls = watch_search(monkeypatch)
    rt, model = runtime()
    run_agent_turn(rt, "S001价格")
    run_agent_turn(rt, f"S001{field}")
    assert len(calls) == 2
    evidence = model.payloads[-1]["_tool_evidence"]
    assert not evidence["reused"]
    assert evidence["source_kind"] == "local_json_snapshot"
    assert "不证明实时库存" in evidence["limitation"]
    assert "不能证明实时库存" in model.observed_contexts[-1]


def test_changed_local_source_and_expiry_force_execution_without_touching_project_data(tmp_path, monkeypatch):
    module, original, calls = watch_search(monkeypatch)
    source = tmp_path / "sku.json"
    source.write_text(json.dumps(original(), ensure_ascii=False), encoding="utf-8")

    def load():
        calls.append(True)
        return json.loads(source.read_text(encoding="utf-8"))

    monkeypatch.setattr(module, "load_sku_data", load)
    rt, model = runtime()
    ticks = [1000.0]
    rt["memory"].facts = SessionFacts(source_path=source, clock=lambda: ticks[0])
    run_agent_turn(rt, "S001价格")
    data = json.loads(source.read_text(encoding="utf-8"))
    data[0]["price"] = "199 元"
    source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert fact(rt, "S001", "price") is None  # Hash changed before next execution.
    assert run_agent_turn(rt, "它多少钱") == "S001：199 元"
    ticks[0] = 1301
    assert run_agent_turn(rt, "它多少钱") == "S001：199 元"
    assert len(calls) == 3
    assert not model.payloads[-1]["_tool_evidence"]["reused"]


def test_actual_validation_error_message_does_not_write_fact():
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "search_product", "args": {}, "id": "invalid_args"}]),
        AIMessage(content="未能查询资料。"),
    ])
    rt = build_agent(llm=model)
    run_agent_turn(rt, "S001价格")
    record = rt["memory"].facts.get_query("search_product", {})
    assert record.status == "error" and not record.facts
    assert fact(rt, "S001", "price") is None


@pytest.mark.parametrize("bad_result", [
    "S001价格是伪造数据（非 JSON）",
    {"query": "S001", "results": [{"sku_id": "S999", "price": "错误归属价格"}]},
    {"query": "S001", "results": "异常结构"},
])
def test_abnormal_executed_result_is_not_forwarded_as_successful_evidence(monkeypatch, bad_result):
    module = importlib.import_module("app.tools.sku_search")
    monkeypatch.setattr(module.search_product, "func", lambda query: bad_result)
    rt, model = runtime()
    run_agent_turn(rt, "S001价格")
    assert fact(rt, "S001", "price") is None
    assert fact(rt, "S999", "price", args={"query": "S001"}) is None
    record = rt["memory"].facts.get_query("search_product", {"query": "S001"})
    assert record.status == "error" and not record.facts
    assert model.payloads[-1]["results"] == []
    assert model.payloads[-1]["_tool_evidence"]["status"] == "error"
    tool_message = next(m for m in rt["last_result"]["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    assert "伪造数据" not in tool_message.content
    assert "错误归属价格" not in tool_message.content
