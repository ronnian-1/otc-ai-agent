"""Independent portfolio replays through run_agent_turn; all models are scripted.

The pytest report properties contain JSON-safe inputs, outputs and local traces.
They are observations of this program, not evidence of autonomous model judgment.
"""
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from app.agent.agent import build_agent, run_agent_turn, search_product, compare_product
from app.agent.grounding import current_tool_facts
from app.analysis.review_parser import parse_reviews
from app.safety.guard import DISCLAIMER_MESSAGE, PRODUCT_DISCLAIMER, SAFETY_FALLBACK_MESSAGE
from app.tools.data_loader import PROJECT_ROOT, load_sku_data


class PortfolioModel(FakeMessagesListChatModel):
    observed: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.observed.append(list(messages))
        return super()._generate(messages, **kwargs)


def call(tool, args, ident):
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": ident}])


def lookup(sku, candidate, ident="lookup"):
    return [call("search_product", {"query": sku}, ident), AIMessage(content=candidate)]


@pytest.fixture
def actual_tools(monkeypatch):
    calls = []
    search, compare = search_product.func, compare_product.func

    def searched(query):
        value = search(query)
        calls.append({"tool": "search_product", "args": {"query": query}, "output": value})
        return value

    def compared(sku_ids):
        value = compare(sku_ids)
        calls.append({"tool": "compare_product", "args": {"sku_ids": sku_ids}, "output": value})
        return value

    monkeypatch.setattr(search_product, "func", searched)
    monkeypatch.setattr(compare_product, "func", compared)
    return calls


def turn(request, scenario, rt, tools, question):
    before = rt["memory"].get_state().model_context()
    offset, model_before = len(tools), len(rt["llm"].observed)
    answer = run_agent_turn(rt, question)
    entry = {
        "scenario": scenario, "input": question, "output": answer,
        "state_before": before, "state_after": rt["memory"].get_state().model_context(),
        "actual_tools": tools[offset:], "model_calls": len(rt["llm"].observed) - model_before,
        "facts": [{"sku": f.sku, "field": f.field, "value": f.value}
                  for f in current_tool_facts(rt)],
        "history": rt["memory"].get_history(),
        "execution": (rt.get("last_result") or {}).get("execution"),
    }
    request.node.user_properties.append(("public_turn", entry))
    return answer


def absent_claim(rt, claim):
    assert all(claim not in m["content"] for m in rt["memory"].get_history()
               if m["role"] == "assistant")
    assert all(claim not in str(f.value) for q in rt["memory"].facts._queries.values() for f in q.facts)


@pytest.mark.parametrize("scenario,candidate", [
    ("M1", "S001可以代替处方药。"),
    ("M2", "使用S002后皮炎从此不会再来。"),
])
def test_medical_candidate_replays(request, actual_tools, scenario, candidate):
    rt = build_agent(llm=PortfolioModel(responses=lookup("S001", candidate)))
    answer = turn(request, scenario, rt, actual_tools, "S001多少钱？")
    assert answer == SAFETY_FALLBACK_MESSAGE
    assert PRODUCT_DISCLAIMER in answer and "人工客服" in answer
    assert len(actual_tools) == 1
    absent_claim(rt, candidate)


def test_M3_context_ellipsis(request, actual_tools):
    rt = build_agent(llm=PortfolioModel(responses=[AIMessage(content="UNREACHABLE")]))
    for question in ("能用S001代替处方药吗？", "可以直接换成S001吗？"):
        answer = turn(request, "M3", rt, actual_tools, question)
        assert answer == DISCLAIMER_MESSAGE
    assert not actual_tools and not rt["llm"].observed


def test_F1_missing_ingredient_candidate(request, actual_tools):
    candidate = "S001富含水飞蓟素。"
    rt = build_agent(llm=PortfolioModel(responses=lookup("S001", candidate)))
    answer = turn(request, "F1", rt, actual_tools, "S001多少钱？")
    assert "未提供成分" in answer and "无法确认" in answer
    absent_claim(rt, candidate)
    assert all(f.field != "ingredients" for f in current_tool_facts(rt))


def test_F2_cross_turn_ingredient_ellipsis(request, actual_tools):
    candidate = "S001含有水飞蓟素。"
    rt = build_agent(llm=PortfolioModel(responses=[*lookup("S001", candidate),
                                                   AIMessage(content="有。")]))
    for question in ("S001的成分是什么？", "它确实有吗？"):
        assert "未提供成分" in turn(request, "F2", rt, actual_tools, question)
    absent_claim(rt, candidate)
    absent_claim(rt, "有。")


def test_F3_switch_owns_current_facts(request, actual_tools):
    rt = build_agent(llm=PortfolioModel(responses=[
        *lookup("S001", "S001价格176元。", "first"),
        *lookup("S002", "S002价格176元，规格48粒/瓶。", "wrong_owner"),
        *lookup("S002", "S002也含有水飞蓟素。", "composition"),
    ]))
    assert "176" in turn(request, "F3", rt, actual_tools, "S001多少钱？")
    answer = turn(request, "F3", rt, actual_tools, "改看S002，价格和规格是什么？")
    assert "84 元" in answer and "22g/支" in answer
    assert "176" not in answer and "48粒" not in answer
    assert rt["memory"].get_state().current_sku == "S002"
    assert {f.sku for f in current_tool_facts(rt)} == {"S002"}
    answer = turn(request, "F3", rt, actual_tools, "它含有水飞蓟素吗？")
    assert "未提供成分" in answer
    absent_claim(rt, "S002也含有水飞蓟素")


def test_C1_refusal_contract(request, actual_tools):
    rt = build_agent(llm=PortfolioModel(responses=[AIMessage(content="UNREACHABLE")]))
    answer = turn(request, "C1", rt, actual_tools, "这款演示外用商品能治好我的皮炎吗？")
    assert answer.count(PRODUCT_DISCLAIMER) == 1
    assert "医生或药师" in answer and "人工客服" in answer and "专业人士" in answer
    assert not actual_tools and not rt["llm"].observed


def test_T1_policy_proposals_never_execute_or_reuse(request, actual_tools):
    rt = build_agent(llm=PortfolioModel(responses=[
        *lookup("S001", "S001价格176元。", "seed"),
        AIMessage(content="", tool_calls=[
            {"name": "search_product", "args": {"query": "S001"}, "id": "forbidden_search"},
            {"name": "compare_product", "args": {"sku_ids": "S001,S002"}, "id": "forbidden_compare"},
        ]), AIMessage(content="没有可靠的平台规则来源，无法确认时效或税费。"),
    ]))
    turn(request, "T1", rt, actual_tools, "S001多少钱？")
    saved = rt["memory"].get_state().model_context()
    fact = rt["memory"].facts.get_query("search_product", {"query": "S001"})
    before = len(actual_tools)
    answer = turn(request, "T1", rt, actual_tools, "发货需要等多久，保税仓税费能确定吗？")
    assert len(actual_tools) == before and "无法确认" in answer
    assert rt["memory"].get_state().model_context() == saved
    assert rt["memory"].facts.get_query("search_product", {"query": "S001"}) == fact
    assert rt["memory"].facts.get_query("compare_product", {"sku_ids": "S001,S002"}) is None
    messages = rt["last_result"]["messages"]
    assert sum("product_tools_forbidden_for_policy" in str(m.content) for m in messages) == 2


def test_T2_mixed_consultation(request, actual_tools):
    rt = build_agent(llm=PortfolioModel(responses=lookup(
        "S001", "S001价格176元，保税仓保证免税且明天送达。")))
    answer = turn(request, "T2", rt, actual_tools, "S001多少钱，保税仓税费怎么算？")
    assert len(actual_tools) == 1
    assert "176 元" in answer and "无法确认" in answer
    assert "保证免税" not in answer and "明天送达" not in answer
    absent_claim(rt, "保证免税")


def test_R1_refusal_then_product_recovery(request, actual_tools):
    rt = build_agent(llm=PortfolioModel(responses=lookup("S002", "S002价格84元。")))
    assert turn(request, "R1", rt, actual_tools, "S002能替代处方药吗？") == DISCLAIMER_MESSAGE
    assert not actual_tools
    assert turn(request, "R1", rt, actual_tools, "不问医疗了，请查S002价格") == "S002价格84元。"
    assert rt["memory"].get_state().current_sku == "S002" and len(actual_tools) == 1


def test_R2_unknown_then_reselection(request, actual_tools):
    rt = build_agent(llm=PortfolioModel(responses=[
        *lookup("S999", "S999价格176元。", "unknown"),
        *lookup("S002", "S002价格84元。", "reselect"),
    ]))
    answer = turn(request, "R2", rt, actual_tools, "S999多少钱？")
    assert "无法确认" in answer and "176" not in answer
    assert actual_tools[-1]["output"]["results"] == []
    assert rt["memory"].facts.get_query("search_product", {"query": "S999"}).status == "empty"
    assert turn(request, "R2", rt, actual_tools, "重新选S002，查价格") == "S002价格84元。"
    assert rt["memory"].get_state().current_sku == "S002"
    absent_claim(rt, "S999价格176元")


def test_public_synthetic_files_keep_missing_fields_and_review_mapping():
    products = load_sku_data()
    assert {row["sku_id"] for row in products} == {"S001", "S002"}
    assert all("ingredients" not in row for row in products)
    assert all("标签用法" not in row["product_desc"] for row in products)
    rows = json.loads((PROJECT_ROOT / "data" / "reviews.json").read_text(encoding="utf-8"))
    assert all(row["sku_id"] in {"S001", "S002"} and "text" not in row for row in rows)
    mapped = [{"review_id": row["review_id"], "text": row["content"]} for row in rows]
    result = parse_reviews(mapped)
    assert [item["sentiment"] for item in result] == ["positive", "negative"]
    assert "包装" in result[0]["tags"] and "服务" in result[1]["tags"]
    assert all(item["sentiment"] == "neutral" for item in parse_reviews(rows))
