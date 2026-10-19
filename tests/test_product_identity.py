"""Program identity/clarification paths with scripted models and real local tools.

No claim is made about external model understanding or response quality.
"""

import importlib
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field

from app.agent.agent import STATE_CONTEXT_PREFIX, build_agent, run_agent_turn
from app.agent.facts import SessionFacts
from app.agent.grounding import apply_fact_boundary
from app.tools.sku_search import search_product


class IdentityModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, **kwargs)


def runtime(responses=None):
    model = IdentityModel(responses=responses or [AIMessage(content="不应执行模型")])
    return build_agent(llm=model), model


def lookup(sku, answer):
    return [AIMessage(content="", tool_calls=[{
        "name": "search_product", "args": {"query": sku}, "id": "identity_" + sku,
    }]), AIMessage(content=answer)]


def assert_history(rt, answer):
    assert rt["memory"].get_history()[-1]["content"] == answer
    assert rt["memory"].get_last_turn()["displayed_reply"] == answer


@pytest.mark.parametrize("question,sku,price", [
    ("演示星屿营养片多少钱", "S001", "176 元"),
    ("请问演示岚溪外用护理膏的价格", "S002", "84 元"),
])
def test_exact_name_binds_only_from_executed_tool_and_answers_actual_price(question, sku, price):
    rt, model = runtime([AIMessage(content=f"{sku}售价{price}。")])
    answer = run_agent_turn(rt, question)
    assert price in answer
    state = rt["memory"].get_state()
    assert state.current_sku == sku
    assert state.sources["current_sku"]["basis"] == "unique_name_search"
    snapshot = next(m.content for m in model.seen[0]
                    if m.type == "system" and str(m.content).startswith(STATE_CONTEXT_PREFIX))
    assert json.loads(snapshot.split("\n")[1])["current_sku"] == sku
    tools = [m for m in rt["last_result"]["messages"] if isinstance(m, ToolMessage)]
    assert tools and json.loads(tools[0].content)["results"][0]["sku_id"] == sku
    query = json.loads(tools[0].content)["_tool_evidence"]["query_args"]
    assert rt["memory"].facts.get_fact(sku, "price", tool="search_product", args=query).value == price
    assert_history(rt, answer)


@pytest.mark.parametrize("question,expected", [
    ("不存在的维生素多少钱", "未找到"),
    ("演示多少钱", "候选"),
    ("护理膏多少钱", "确认"),
])
def test_unmatched_multiple_or_partial_names_never_bind_previous_sku(question, expected):
    rt, model = runtime(lookup("S001", "S001售价176元。"))
    run_agent_turn(rt, "S001多少钱")
    before = len(model.seen)
    answer = run_agent_turn(rt, question)
    assert expected in answer and "176" not in answer
    assert len(model.seen) == before
    assert rt["memory"].get_state().current_sku is None
    assert rt["memory"].get_state().comparison_skus == ()
    if "演示" in question:
        assert "S001" in answer and "S002" in answer
        assert "演示星屿营养片" in answer and "演示 岚溪外用护理膏" in answer
    if "不存在" in question:
        assert "S001" not in answer and "S002" not in answer
    assert_history(rt, answer)
    again = run_agent_turn(rt, "它适合谁")
    assert "确认" in again and "SKU" in again
    assert rt["memory"].get_state().current_sku is None


def test_clarification_attribute_request_is_preserved_and_reviewed():
    rt, model = runtime()
    reviews = []

    def review(text):
        from app.safety.guard import check_ai_output
        reviews.append(text)
        return check_ai_output(text)

    answer = run_agent_turn(rt, "它适合谁", output_guard=review)
    assert answer == rt["memory"].get_state().clarification
    assert "确认" in answer and "SKU" in answer
    assert reviews == [answer] and model.seen == []
    assert rt["last_fact_check"]["decision"] == "unchanged"
    assert rt["memory"].facts.get_query("search_product", {"query": "它"}) is None
    assert_history(rt, answer)


@pytest.mark.parametrize("claim", [
    "S001售价一元。", "S001规格999粒。", "S001还剩十件，可以直接拍。",
    "S001今天优惠八折。", "你的订单明天送达。", "保证治愈皮炎。",
])
def test_clarification_output_filter_cannot_add_unverified_or_unsafe_claims(claim):
    rt, model = runtime()

    def filter_output(text):
        return {"safe": True, "filtered_text": "请确认商品 SKU。" + claim}

    answer = run_agent_turn(rt, "它适合谁", output_guard=filter_output)
    assert claim not in answer
    assert all(claim not in m["content"] for m in rt["memory"].get_history())
    assert rt["memory"].get_state().current_sku is None and model.seen == []
    assert rt["memory"].facts.get_query("search_product", {"query": "S001"}) is None


def test_explicit_sku_and_resolved_attribute_followup_remain_available():
    rt, _ = runtime(lookup("S001", "S001售价176元。") +
                    lookup("S001", "经常熬夜、轮班、希望记录作息变化的成年演示人群"))
    assert run_agent_turn(rt, "S001多少钱") == "S001售价176元。"
    answer = run_agent_turn(rt, "它适合谁")
    assert "经常熬夜" in answer and "SKU" not in answer
    assert rt["memory"].get_state().current_sku == "S001"
    assert_history(rt, answer)


@pytest.mark.parametrize("failure", ["missing", "malformed", "changed_version"])
def test_name_lookup_failure_never_binds_or_revives_old_facts(monkeypatch, failure):
    module = importlib.import_module("app.tools.sku_search")
    rt, model = runtime(lookup("S001", "S001售价176元。"))
    run_agent_turn(rt, "S001多少钱")
    before = len(model.seen)

    def load():
        if failure == "missing":
            raise FileNotFoundError("test only")
        return [{"sku_id": "bad-id", "name": "演示星屿营养片"}]

    if failure == "changed_version":
        versions = iter(["before", "after", "after", "after"])
        monkeypatch.setattr(rt["memory"].facts, "source_version", lambda: next(versions, "after"))
    else:
        monkeypatch.setattr(module, "load_sku_data", load)
    answer = run_agent_turn(rt, "演示星屿营养片多少钱")
    assert "无法确认" in answer and "176" not in answer
    assert rt["memory"].get_state().current_sku is None
    assert len(model.seen) == before
    assert_history(rt, answer)


def test_name_binding_does_not_trust_model_identity_or_cross_session_history():
    first, _ = runtime([AIMessage(content="S002售价一元。")])
    answer = run_agent_turn(first, "演示星屿营养片多少钱")
    assert first["memory"].get_state().current_sku == "S001"
    assert "S002售价一元" not in answer
    assert "176 元" in answer
    assert_history(first, answer)
    second, model = runtime()
    assert "确认" in run_agent_turn(second, "它适合谁") and model.seen == []
    first["memory"].clear()
    assert first["memory"].get_state().current_sku is None
    assert "确认" in run_agent_turn(first, "它适合谁")


def test_name_entry_runs_input_safety_before_any_lookup(monkeypatch):
    module = importlib.import_module("app.tools.sku_search")

    def forbidden():
        raise AssertionError("unsafe input must not read the catalog")

    monkeypatch.setattr(module, "load_sku_data", forbidden)
    rt, model = runtime()
    answer = run_agent_turn(rt, "演示 岚溪外用护理膏能治好皮炎吗")
    assert "请咨询医生或药师" in answer and model.seen == []
    assert rt.get("last_result") is None


def test_model_claimed_clarification_does_not_bypass_fact_check():
    candidate = "回复类型：clarification。S001售价一元。请确认。"
    rt, _ = runtime([AIMessage(content=candidate)])
    answer = run_agent_turn(rt, "S001多少钱")
    assert answer != candidate and "售价一元" not in answer
    assert_history(rt, answer)


def test_tool_normalizes_name_spacing_without_a_product_specific_alias():
    result = search_product.invoke({"query": "演示岚溪外用护理膏"})
    assert [r["sku_id"] for r in result["results"]] == ["S002"]


@pytest.mark.parametrize("duplicate", [False, True])
def test_catalog_drives_arbitrary_full_names_and_duplicate_names_require_selection(tmp_path, monkeypatch, duplicate):
    from app.tools import data_loader
    from app.tools.data_loader import load_sku_data

    row = load_sku_data()[0]
    row.update(sku_id="S731", name="清润的旅行护理霜", price="88 元")
    rows = [row]
    if duplicate:
        rows.append({**row, "sku_id": "S732"})
    source = tmp_path / "synthetic_catalog.json"
    source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(data_loader, "SKU_DATA_PATH", source)
    rt, model = runtime([AIMessage(content="S731售价88元。")])
    rt["memory"].facts = SessionFacts(source_path=source)
    answer = run_agent_turn(rt, "清润的旅行护理霜多少钱")
    if duplicate:
        assert "候选" in answer and "S731" in answer and "S732" in answer
        assert rt["memory"].get_state().current_sku is None and model.seen == []
    else:
        assert answer == "S731售价88元。"
        assert rt["memory"].get_state().current_sku == "S731"
    assert_history(rt, answer)


def test_direct_clarification_fact_review_preserves_question_and_rejects_invented_names():
    rt, _ = runtime()
    state = rt["memory"].update_state("它适合谁")
    result = apply_fact_boundary(rt, "它适合谁", state.clarification, reply_kind="clarification")
    assert result["text"] == state.clarification
    invented = state.clarification + "S999本地资料中的名称：虚构商品。"
    result = apply_fact_boundary(rt, "它适合谁", invented, reply_kind="clarification")
    assert result["text"] != invented and "虚构商品" not in result["text"]


def test_bare_name_followup_clears_selection_but_budget_concern_does_not():
    rt, model = runtime(lookup("S001", "S001售价176元。") +
                        [AIMessage(content="要不要调整预算范围？")])
    run_agent_turn(rt, "S001多少钱，预算200元以内")
    answer = run_agent_turn(rt, "有点贵呢")
    assert "调整预算" in answer
    assert rt["memory"].get_state().current_sku == "S001"
    assert rt["memory"].get_state().budget_max == 200
    before = len(model.seen)
    answer = run_agent_turn(rt, "护理膏呢")
    assert "确认" in answer and "S002" in answer
    assert "176" not in answer and len(model.seen) == before
    assert rt["memory"].get_state().current_sku is None
    assert_history(rt, answer)


@pytest.mark.parametrize("question", ["不存在的维生素多少钱", "演示多少钱"])
def test_fresh_session_name_resolution_does_not_guess(question):
    rt, model = runtime()
    answer = run_agent_turn(rt, question)
    assert rt["memory"].get_state().current_sku is None and model.seen == []
    assert "未找到" in answer if "不存在" in question else "候选" in answer
    assert_history(rt, answer)
