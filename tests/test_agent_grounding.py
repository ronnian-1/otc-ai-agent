"""Injected model candidates + actual local tools; no external model evaluation."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.agent import SAFETY_FALLBACK_MESSAGE, build_agent, run_agent_turn
from app.agent.facts import SessionFacts
from app.agent.grounding import apply_fact_boundary, current_tool_facts
from app.safety.guard import check_ai_output


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def search(sku, text):
    return [AIMessage(content="", tool_calls=[{
        "name": "search_product", "args": {"query": sku}, "id": "lookup_" + sku,
    }]), AIMessage(content=text)]


def runtime(responses):
    return build_agent(llm=ScriptedModel(responses=responses))


@pytest.mark.parametrize("question,candidate,forbidden,expected", [
    ("多久出库", "下单后24-72小时内出库。", "24-72", "无法确认具体出库"),
    ("S001有现货吗", "S001现货充足，今天能发货。", "今天能发货", "没有实时库存"),
    ("S001是否有优惠", "今天打八折，满200减50。", "满200减50", "无法确认当前优惠"),
    ("我的订单明天能送达吗", "你的订单明天送达。", "明天送达", "未接入订单"),
    ("保税仓会不会被税", "保税仓全部免税。", "全部免税", "没有可靠的平台规则来源"),
    ("S001能保价吗", "保证全网最低价，七天保价。", "七天保价", "无法确认当前优惠"),
])
def test_unsupported_business_candidates_are_bounded_and_replacement_is_reviewed(
    question, candidate, forbidden, expected,
):
    rt = runtime([AIMessage(content=candidate)])
    reviews = []

    def review(text):
        reviews.append(text)
        return check_ai_output(text)

    answer = run_agent_turn(rt, question, output_guard=review)
    assert expected in answer and forbidden not in answer
    assert reviews == [candidate, answer]
    turn = rt["memory"].get_last_turn()
    assert turn["candidate_reply"] == candidate and turn["displayed_reply"] == answer
    assert turn["output_check"]["fact_boundary"]["decision"] == "bounded"
    assert rt["memory"].get_history()[-1]["content"] == answer
    assert len(rt["memory"].get_history()) == 2
    assert rt["memory"].facts.get_query("search_product", {"query": "S001"}) is None


@pytest.mark.parametrize("question,field,local_value,limitation", [
    ("S001库存", "stock", "合成库存记录：待人工确认", "无法确认当前有货或无货"),
    ("S001配送说明", "delivery", "合成配送备注：需由演示商家另行核对，不提供发出或送达时限", "不是履约保证"),
])
def test_executed_local_business_fields_are_quoted_only_as_local_records(question, field, local_value, limitation):
    candidate = "S001现货充足，24小时出库，三天必达。"
    rt = runtime(search("S001", candidate))
    answer = run_agent_turn(rt, question)
    assert local_value in answer and "本地资料" in answer and limitation in answer
    assert "24小时" not in answer and "三天必达" not in answer
    fact = rt["memory"].facts.get_fact("S001", field, tool="search_product", args={"query": "S001"})
    assert local_value in fact.value
    payload = json.loads(next(m.content for m in rt["last_result"]["messages"] if isinstance(m, ToolMessage)))
    assert payload["_tool_evidence"]["scope"]["business_freshness"] == "unverified"
    assert payload["_tool_evidence"]["scope"]["data_updated_at"] is None


def test_order_status_is_not_derived_from_successful_product_search():
    rt = runtime(search("S001", "订单明天出库，3-7个工作日送达。"))
    answer = run_agent_turn(rt, "S001我的订单何时送达")
    assert "未接入订单" in answer and "无法确认" in answer
    assert "明天出库" not in answer
    assert "本地资料中的配送" in answer
    assert "不代表" in answer
    assert not any(f.field == "order_status" for f in current_tool_facts(rt))


def test_cached_local_query_cannot_support_unsolicited_live_promotion_or_stock(monkeypatch):
    import importlib

    module = importlib.import_module("app.tools.sku_search")
    original = module.load_sku_data
    calls = []

    def load():
        calls.append(True)
        return original()

    monkeypatch.setattr(module, "load_sku_data", load)
    rt = runtime([*search("S001", "S001价格176元。"),
                  *search("S001", "S001价格176元，实时库存充足，当前优惠八折。")])
    assert "176元" in run_agent_turn(rt, "S001多少钱")
    answer = run_agent_turn(rt, "它多少钱")
    assert len(calls) == 1  # A genuinely reused local result.
    evidence = json.loads(next(m.content for m in rt["last_result"]["messages"] if isinstance(m, ToolMessage)))["_tool_evidence"]
    assert evidence["reused"] is True
    assert "没有实时库存数据" in answer and "没有可靠的促销" in answer
    assert "S001本地资料中的价格：176 元" in answer
    assert "当前优惠八折" not in answer


@pytest.mark.parametrize("query,sku,candidate,expected,forbidden", [
    ("S001多少钱", "S001", "S001价格176元。", "176元", None),
    ("S002规格", "S002", "S002规格22g/支。", "22g/支", None),
    ("S002多少钱", "S002", "S002价格176元。", "84 元", "176元"),
    ("S002介绍一下", "S002", "S002可以美白，适合孕妇。", "虚构的外用护理展示样本，用于演示包装形式及商品类别差异", "美白"),
])
def test_supported_product_values_pass_and_unsupported_attributes_use_own_evidence(query, sku, candidate, expected, forbidden):
    rt = runtime(search(sku, candidate))
    answer = run_agent_turn(rt, query)
    assert expected in answer
    if forbidden:
        assert forbidden not in answer
    else:
        assert answer == candidate


def test_unknown_sku_cannot_inherit_previous_price_spec_or_promises():
    rt = runtime([*search("S001", "S001价格176元。"),
                  *search("S999", "S999价格176元，规格48粒/瓶，现货充足。")])
    run_agent_turn(rt, "S001多少钱")
    answer = run_agent_turn(rt, "S999价格和规格")
    assert "176" not in answer and "48粒" not in answer
    assert "无法确认" in answer and not current_tool_facts(rt)
    assert rt["memory"].facts.get_query("search_product", {"query": "S999"}).status == "empty"


def test_missing_evidence_does_not_accept_model_price_as_fact_or_history():
    rt = runtime([AIMessage(content="S001价格1元。")])
    answer = run_agent_turn(rt, "S001多少钱")
    assert "无法确认" in answer and "1元" not in answer
    assert not current_tool_facts(rt)
    assert rt["memory"].get_last_turn()["candidate_reply"] == "S001价格1元。"
    assert "1元" not in rt["memory"].get_history()[-1]["content"]


@pytest.mark.parametrize("field,candidate", [
    ("成分", "采用珍稀草本配方。"), ("产地", "原产法国。"),
    ("保质期", "三年。"), ("功效", "人人皆宜。"),
])
def test_missing_attribute_cannot_be_filled_from_model_even_after_real_lookup(field, candidate):
    rt = runtime(search("S002", candidate))
    answer = run_agent_turn(rt, "S002的" + field)
    if field == "功效":
        assert "S002本地资料中的商品描述" in answer
    else:
        assert "无法确认" in answer
    assert candidate not in answer


def test_output_filter_cannot_restore_removed_unverified_claim():
    candidate = "S001价格1元。"
    rt = runtime([AIMessage(content=candidate)])
    assert run_agent_turn(rt, "S001多少钱", output_guard=lambda text: {
        "safe": True, "filtered_text": candidate,
    }) == SAFETY_FALLBACK_MESSAGE
    assert rt["memory"].get_last_turn()["output_check"]["safe"] is False
    assert candidate not in rt["memory"].get_history()[-1]["content"]


@pytest.mark.parametrize("candidate", ["S001价格1176元。", "S001规格176元。"])
def test_matching_substring_or_another_field_is_not_fact_support(candidate):
    rt = runtime(search("S001", candidate))
    answer = run_agent_turn(rt, "S001多少钱")
    assert answer != candidate
    assert "S001本地资料中的价格：176 元" in answer


def test_comparison_business_quotes_preserve_order_and_each_sku_fields():
    rt = runtime([AIMessage(content="", tool_calls=[{
        "name": "compare_product", "args": {"sku_ids": "S002,S001"}, "id": "compare_stock",
    }]), AIMessage(content="所有商品都有现货。")])
    answer = run_agent_turn(rt, "比较S002和S001的库存")
    assert "无法确认当前有货或无货" in answer
    assert answer.index("S002本地资料") < answer.index("S001本地资料")
    assert "所有商品都有现货" not in answer
    assert rt["memory"].get_state().comparison_skus == ("S002", "S001")


def test_filtered_search_is_not_a_complete_inventory_catalog():
    rt = runtime([AIMessage(content="", tool_calls=[{
        "name": "search_product", "args": {"query": "护理膏"}, "id": "filtered_stock",
    }]), AIMessage(content="所有商品都有现货，S001库存充足。")])
    answer = run_agent_turn(rt, "S001有货吗")
    assert "没有实时库存数据" in answer
    assert "S001本地资料中的库存" not in answer  # Result contains S002 only.
    assert "所有商品都有现货" not in answer
    record = rt["memory"].facts.get_query("search_product", {"query": "护理膏"})
    assert {f.sku for f in record.facts} == {"S002"}


def test_expiry_version_isolation_and_reset_remove_readable_evidence(tmp_path):
    source = tmp_path / "snapshot.json"
    source.write_text("[]", encoding="utf-8")
    ticks = [10.0]
    rt = runtime(search("S001", "S001价格176元。"))
    rt["memory"].facts = SessionFacts(source_path=source, clock=lambda: ticks[0])
    run_agent_turn(rt, "S001多少钱")
    assert current_tool_facts(rt)
    second = runtime([AIMessage(content="S001价格176元。")])
    assert "无法确认" in run_agent_turn(second, "S001多少钱")
    ticks[0] = 311
    assert not current_tool_facts(rt)
    result = apply_fact_boundary(rt, "S001价格", "S001价格176元。")
    assert "176" not in result["text"]
    ticks[0] = 10
    source.write_text("[{}]", encoding="utf-8")
    assert not current_tool_facts(rt)
    rt["memory"].clear()
    assert rt["memory"].get_history() == [] and not current_tool_facts(rt)


def test_medical_review_precedes_boundary_and_replacement_cannot_bypass_review():
    rt = runtime([AIMessage(content="现货充足，保证治愈。")])
    assert run_agent_turn(rt, "S001有现货吗") == SAFETY_FALLBACK_MESSAGE
    assert rt["memory"].get_last_turn()["output_check"]["safe"] is False
    assert "last_fact_check" not in rt

    rt = runtime([AIMessage(content="24小时出库。")])
    calls = []

    def rejecting_replacement(text):
        calls.append(text)
        return {"safe": len(calls) == 1, "filtered_text": text}

    assert run_agent_turn(rt, "多久出库", output_guard=rejecting_replacement) == SAFETY_FALLBACK_MESSAGE
    assert len(calls) == 2
    assert rt["memory"].get_last_turn()["output_check"]["safe"] is False


@pytest.mark.parametrize("question,queried,candidate", [
    ("S001多少钱", None, "S001售价一元。"),
    ("S999多少钱", None, "S999售价一元。"),
    ("S002多少钱", "S002", "按预算S002售价1元。"),
    ("S001介绍一下", "S001", "S001还剩十件，可以直接拍。"),
    ("S002多少钱", "S002", "预算200元，S002售价一元。"),
    ("S002多少钱", "S001", "S002售价一百七十六元。"),
    ("S999多少钱", "S999", "S999售价1元。"),
    ("S002多少钱", "S002", "S002售价84元，还剩十件，可以直接拍。"),
    ("S002多少钱", "S002", "S002售价壹元。"),
    ("S002多少钱", "S002", "S002售价预算一元。"),
    ("S002多少钱", "S002", "S002售价-84元。"),
    ("S002多少钱", "S002", "S002售价负八十四元。"),
])
def test_price_and_counted_stock_repros_final_display_and_history(question, queried, candidate):
    rt = runtime(search(queried, candidate) if queried else [AIMessage(content=candidate)])
    answer = run_agent_turn(rt, question)
    assert answer != candidate
    assert "售价一元" not in answer and "售价1元" not in answer
    assert "还剩十件" not in answer and "可以直接拍" not in answer
    assert rt["memory"].get_history()[-1]["content"] == answer
    assert all(candidate not in m["content"] for m in rt["memory"].get_history())
    assert rt["memory"].get_last_turn()["candidate_reply"] == candidate
    assert check_ai_output(answer)["safe"] is True
    result = apply_fact_boundary(rt, question, candidate)
    assert result["text"] != candidate
    assert not any(str(f.value) in ("1", "1元", "一元") for f in current_tool_facts(rt))


@pytest.mark.parametrize("candidate", [
    "S002售价八十四元。", "S002售价84元。", "预算200元，S002售价84元。",
])
def test_price_value_is_matched_to_structured_current_sku_field(candidate):
    rt = runtime(search("S002", candidate))
    assert run_agent_turn(rt, "S002多少钱") == candidate
    assert rt["memory"].get_history()[-1]["content"] == candidate


@pytest.mark.parametrize("fault", ["failure", "missing", "wrong_sku"])
def test_invalid_tool_price_evidence_cannot_license_final_claim(monkeypatch, fault):
    import importlib
    module = importlib.import_module("app.tools.sku_search")
    original = module.load_sku_data

    def load():
        if fault == "failure":
            raise OSError("test-only failure")
        rows = original()
        if fault == "missing":
            for row in rows:
                if row["sku_id"] == "S002":
                    row.pop("price", None)
        return rows

    monkeypatch.setattr(module, "load_sku_data", load)
    if fault == "wrong_sku":
        monkeypatch.setattr(module, "_matches_product", lambda item, query: item["sku_id"] == "S001")
    candidate = "S002售价八十四元。"
    rt = runtime(search("S002", candidate))
    answer = run_agent_turn(rt, "S002多少钱")
    assert answer != candidate and "无法确认" in answer
    assert not any(f.field == "price" for f in current_tool_facts(rt))
    assert all(candidate not in m["content"] for m in rt["memory"].get_history())


def test_replacement_is_fact_reviewed_even_without_output_filter_changes(monkeypatch):
    import importlib
    module = importlib.import_module("app.agent.agent")
    original = module.apply_fact_boundary
    calls = []

    def boundary(rt, question, text, **kwargs):
        calls.append(text)
        return original(rt, question, text, **kwargs)

    monkeypatch.setattr(module, "apply_fact_boundary", boundary)
    rt = runtime([AIMessage(content="S001售价一元。")])
    answer = run_agent_turn(rt, "S001多少钱")
    assert len(calls) >= 2 and calls[-1] == answer
    assert rt["memory"].get_history()[-1]["content"] == answer


@pytest.mark.parametrize("candidate", [
    "S002售价八十四元，S001售价一百七十六元。",
    "S002售价一百七十六元，S001售价八十四元。",
])
def test_comparison_price_validation_keeps_per_sku_ownership(candidate):
    rt = runtime([AIMessage(content="", tool_calls=[{
        "name": "compare_product", "args": {"sku_ids": "S002,S001"}, "id": "prices",
    }]), AIMessage(content=candidate)])
    answer = run_agent_turn(rt, "比较S002和S001的价格")
    if candidate.startswith("S002售价八十四"):
        assert answer == candidate
    else:
        assert answer != candidate
        assert "S002本地资料中的价格：84 元" in answer
        assert "S001本地资料中的价格：176 元" in answer
    assert rt["memory"].get_history()[-1]["content"] == answer


@pytest.mark.parametrize("injected", [
    "S001售价一元。", "S001还剩十件，可以直接拍。",
    "根据你的皮炎情况，每周用一支。",
])
def test_replacement_output_filter_cannot_introduce_new_claim_or_dose(injected):
    rt = runtime([AIMessage(content="S001售价二元。")])
    reviews = []

    def filter_output(text):
        reviews.append(text)
        return {"safe": True, "filtered_text": text if len(reviews) == 1 else injected}

    answer = run_agent_turn(rt, "S001多少钱", output_guard=filter_output)
    assert answer == SAFETY_FALLBACK_MESSAGE
    assert len(reviews) == 2
    assert rt["memory"].get_last_turn()["output_check"]["safe"] is False
    assert rt["memory"].get_last_turn()["displayed_reply"] == answer
    assert all(injected not in m["content"] for m in rt["memory"].get_history())
    assert not current_tool_facts(rt)


def test_boundary_generated_replacement_itself_cannot_skip_fact_review(monkeypatch):
    import importlib
    module = importlib.import_module("app.agent.agent")
    original = module.apply_fact_boundary

    def broken_renderer(rt, question, text, **kwargs):
        result = original(rt, question, text, **kwargs)
        if not kwargs.get("verify_only"):
            return {**result, "text": "S001售价一元。"}
        return result

    monkeypatch.setattr(module, "apply_fact_boundary", broken_renderer)
    rt = runtime([AIMessage(content="S001售价二元。")])
    answer = run_agent_turn(rt, "S001多少钱")
    assert answer == SAFETY_FALLBACK_MESSAGE
    assert rt["memory"].get_last_turn()["output_check"]["replacement_fact_review"]["verified"] is False
    assert "售价一元" not in rt["memory"].get_history()[-1]["content"]


def test_local_stock_is_not_live_even_with_matching_literal_count_in_tool(monkeypatch):
    import importlib
    module = importlib.import_module("app.tools.sku_search")
    original = module.load_sku_data

    def load():
        rows = original()
        for row in rows:
            if row["sku_id"] == "S001":
                row["stock"] = "十件"
        return rows

    monkeypatch.setattr(module, "load_sku_data", load)
    candidate = "S001还剩十件，可以直接拍。"
    rt = runtime(search("S001", candidate))
    answer = run_agent_turn(rt, "S001库存")
    assert "本地资料中的库存：十件" in answer
    assert "没有实时库存数据" in answer and "不代表实时库存" in answer
    assert "可以直接拍" not in answer
    assert rt["memory"].get_history()[-1]["content"] == answer
    assert apply_fact_boundary(rt, "S001库存", answer, verify_only=True)["verified"] is True
    assert apply_fact_boundary(rt, "S001库存", answer + candidate, verify_only=True)["verified"] is False


@pytest.mark.parametrize("question,queried,candidate,expected", [
    ("S999规格", None, "S999规格二十二克/支。", "无法确认"),
    ("S999规格", "S999", "S999规格二十二克/支。", "无法确认"),
    ("S002规格", "S002", "S002规格三十克/支。", "22g/支"),
    ("S002规格", "S002", "S002规格30g/支。", "22g/支"),
    ("S002规格", "S002", "S002规格一百毫升/瓶。", "22g/支"),
    ("S002规格", "S002", "S002每盒二十四片。", "22g/支"),
])
def test_spec_claim_repros_use_final_review_and_never_pollute_history_or_facts(
    question, queried, candidate, expected,
):
    rt = runtime(search(queried, candidate) if queried else [AIMessage(content=candidate)])
    reviews = []

    def review(text):
        reviews.append(text)
        return check_ai_output(text)

    answer = run_agent_turn(rt, question, output_guard=review)
    assert answer != candidate and expected in answer
    assert reviews == [candidate, answer]
    assert rt["memory"].get_history() == [
        {"role": "user", "content": question}, {"role": "assistant", "content": answer},
    ]
    turn = rt["memory"].get_last_turn()
    assert turn["candidate_reply"] == candidate
    assert turn["output_check"]["replacement_fact_review"]["verified"] is True
    assert turn["output_check"]["replacement_safety_review"]["safe"] is True
    assert apply_fact_boundary(rt, question, candidate, verify_only=True)["verified"] is False
    facts = current_tool_facts(rt)
    assert [f.value for f in facts if f.field == "spec"] == (["22g/支"] if queried == "S002" else [])


@pytest.fixture
def spec_catalog(tmp_path, monkeypatch):
    """Exercise actual search on a temporary source, never edit project data."""
    from app.tools import data_loader

    rows = data_loader.load_sku_data()
    source = tmp_path / "spec_catalog.json"
    monkeypatch.setattr(data_loader, "SKU_DATA_PATH", source)

    def make(spec, candidate):
        rows[1]["spec"] = spec
        source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        rt = runtime(search("S002", candidate))
        rt["memory"].facts = SessionFacts(source_path=source)
        return rt

    return make


@pytest.mark.parametrize("spec", [
    "二十二克/支", "一百毫升/瓶", "每盒二十四片", "22g/支", "24片/盒", "48 粒/瓶",
])
def test_source_supported_spec_is_preserved_with_real_temporary_tool(spec_catalog, spec):
    candidate = "S002规格：" + spec + "。"
    rt = spec_catalog(spec, candidate)
    assert run_agent_turn(rt, "S002规格") == candidate
    assert rt["memory"].get_history()[-1]["content"] == candidate
    fact = rt["memory"].facts.get_fact("S002", "spec", tool="search_product", args={"query": "S002"})
    assert fact.value == spec
    assert apply_fact_boundary(rt, "S002规格", candidate, verify_only=True)["verified"] is True


@pytest.mark.parametrize("spec,candidate", [
    ("一百毫升/瓶", "S002规格一千一百毫升/瓶。"),
    ("二十二克/支", "S002规格一百二十二克/支。"),
    ("每盒二十四片", "S002每瓶二十四片。"),
    ("二十二克/支", "S002规格二十二克/瓶。"),
    ("二十二克/支", "S002规格二十二克/支，另有三十克/支。"),
    ("22g/支", "S002规格若干克/支。"),
    ("22g/支", "S002规格半两/支。"),
])
def test_spec_quantity_package_and_unparsed_claims_need_own_full_field(spec_catalog, spec, candidate):
    rt = spec_catalog(spec, candidate)
    answer = run_agent_turn(rt, "S002规格")
    assert answer != candidate and spec in answer
    assert rt["memory"].get_history()[-1]["content"] == answer
    assert rt["memory"].get_last_turn()["output_check"]["replacement_fact_review"]["verified"] is True


@pytest.mark.parametrize("fault", ["wrong_sku", "missing", "wrong_field", "failure", "no_current_lookup"])
def test_spec_evidence_requires_current_tool_correct_sku_and_spec_field(monkeypatch, fault):
    import importlib

    module = importlib.import_module("app.tools.sku_search")
    original = module.load_sku_data

    def load():
        if fault == "failure":
            raise OSError("test-only spec lookup failure")
        rows = original()
        if fault in ("missing", "wrong_field"):
            rows[1].pop("spec", None)
        if fault == "wrong_field":
            rows[1]["product_desc"] = "每盒二十四片"
        return rows

    monkeypatch.setattr(module, "load_sku_data", load)
    candidate = "S002每盒二十四片。"
    queried = "S001" if fault == "wrong_sku" else "S002"
    if fault == "wrong_sku":
        candidate = "S002规格48 粒/瓶。"  # Exact value, but owned by S001.
    responses = search(queried, candidate)
    if fault == "no_current_lookup":
        candidate = "S002规格22g/支。"  # Exact cached value, no tool scope this turn.
        responses = search("S002", "S002价格84元。") + [AIMessage(content=candidate)]
    rt = runtime(responses)
    if fault == "no_current_lookup":
        run_agent_turn(rt, "S002多少钱")
    answer = run_agent_turn(rt, "S002规格")
    assert answer != candidate and "二十四片" not in answer
    assert rt["memory"].get_history()[-1]["content"] == answer
    assert all(candidate not in m["content"] for m in rt["memory"].get_history())
    assert not any(f.field == "spec" and f.value == "每盒二十四片" for f in current_tool_facts(rt))


@pytest.mark.parametrize("claim", [
    "S999规格三十克/支。", "S999规格一百毫升/瓶。", "S999每盒二十四片。",
])
def test_clarification_cannot_license_chinese_spec_and_restores_trusted_question(claim):
    rt = runtime([AIMessage(content="不应调用模型")])
    injected = "请确认商品 SKU。" + claim
    reviews = []

    def filter_output(text):
        reviews.append(text)
        return {"safe": True, "filtered_text": injected if len(reviews) == 1 else text}

    answer = run_agent_turn(rt, "它适合谁", output_guard=filter_output)
    assert answer == rt["memory"].get_state().clarification
    assert "确认" in answer and "SKU" in answer and claim not in answer
    assert len(reviews) == 2
    assert rt["memory"].get_history()[-1]["content"] == answer
    turn = rt["memory"].get_last_turn()
    assert turn["output_check"]["replacement_fact_review"]["verified"] is True
    assert turn["output_check"]["replacement_safety_review"]["safe"] is True
    assert not current_tool_facts(rt)


@pytest.mark.parametrize("candidate", [
    "S002售价八十四元。", "可以先比较，再慢慢考虑。", "每月购买一瓶是你的购物计划。",
])
def test_non_spec_price_description_and_purchase_count_are_not_spec_assertions(candidate):
    rt = runtime(search("S002", candidate))
    assert run_agent_turn(rt, "S002多少钱") == candidate
    assert rt["memory"].get_history()[-1]["content"] == candidate


@pytest.mark.parametrize("invalidation", ["expired", "version_changed"])
def test_chinese_spec_evidence_must_still_be_valid_at_final_review(spec_catalog, monkeypatch, invalidation):
    candidate = "S002规格一百毫升/瓶。"
    rt = spec_catalog("一百毫升/瓶", candidate)

    def invalidate_after_tool(text):
        facts = rt["memory"].facts
        if invalidation == "expired":
            record = facts.get_query("search_product", {"query": "S002"})
            monkeypatch.setattr(facts, "_clock", lambda: record.expires_at)
        else:
            source = facts.source_path
            source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return check_ai_output(text)

    answer = run_agent_turn(rt, "S002规格", output_guard=invalidate_after_tool)
    assert "无法确认" in answer and "一百毫升" not in answer
    assert rt["memory"].get_history()[-1]["content"] == answer
    assert not current_tool_facts(rt)
    assert rt["memory"].get_last_turn()["output_check"]["replacement_fact_review"]["verified"] is True


@pytest.mark.parametrize("injection_point", ["output_filter", "boundary_renderer"])
def test_replacement_cannot_reintroduce_unverified_chinese_spec(monkeypatch, injection_point):
    import importlib

    module = importlib.import_module("app.agent.agent")
    injected = "S002规格三十克/支。"
    rt = runtime(search("S002", "S002规格一百毫升/瓶。"))
    calls = []

    def filter_output(text):
        calls.append(text)
        return {"safe": True, "filtered_text": injected if len(calls) == 2 else text}

    if injection_point == "boundary_renderer":
        original = module.apply_fact_boundary

        def altered_boundary(runtime, question, text, **kwargs):
            result = original(runtime, question, text, **kwargs)
            return result if kwargs.get("verify_only") else {**result, "text": injected}

        monkeypatch.setattr(module, "apply_fact_boundary", altered_boundary)
    answer = run_agent_turn(
        rt, "S002规格", **({"output_guard": filter_output} if injection_point == "output_filter" else {}),
    )
    assert answer == SAFETY_FALLBACK_MESSAGE
    review = rt["memory"].get_last_turn()["output_check"]
    assert review["replacement_fact_review"]["verified"] is False
    assert review["safe"] is False
    assert all(injected not in m["content"] for m in rt["memory"].get_history())
    assert [f.value for f in current_tool_facts(rt) if f.field == "spec"] == ["22g/支"]
