"""Fact-store contracts with synthetic tool payloads and an injectable clock."""

import pytest

from app.agent.facts import PRODUCT_FIELDS, SessionFacts


@pytest.fixture
def store(tmp_path):
    source = tmp_path / "sku.json"
    source.write_text("[]", encoding="utf-8")
    ticks = [1000.0]
    return SessionFacts(source_path=source, ttl_seconds=300, clock=lambda: ticks[0]), ticks


def product(sku="S001"):
    return {"sku_id": sku, **{field: f"{sku}:{field}" for field in PRODUCT_FIELDS}}


def search_result(sku="S001"):
    return {"query": sku, "results": [product(sku)]}


def record(store, result, *, tool="search_product", args=None, **kwargs):
    args = args if args is not None else {"query": "S001"}
    version = store.source_version()
    return store.record_result(tool, args, result, version_before=version,
                               version_after=version, **kwargs)


def read(store, sku="S001", field="price", *, args=None, tool="search_product"):
    return store.get_fact(sku, field, tool=tool,
                          args=args if args is not None else {"query": "S001"})


def test_facts_are_scoped_by_sku_field_source_and_exact_conditions(store):
    facts, _ = store
    payload = {"query": "S001", "results": [{"sku_id": "S001", "price": "176 元"}]}
    result = record(facts, payload)
    price = read(facts)
    assert result.status == "partial"
    assert price.value == "176 元"
    assert price.source_tool == "search_product"
    assert price.query_args == {"query": "S001"}
    assert price.source_path == str(facts.source_path)
    assert price.source_kind == "local_json_snapshot"
    assert price.success and not price.missing
    assert read(facts, "S002") is None
    assert read(facts, field="taboo") is None
    assert read(facts, args={"query": "S002"}) is None
    assert read(facts, args={"query": "S001", "budget": 100}) is None
    assert read(facts, tool="compare_product") is None
    assert facts.reusable_result("search_product", {"query": "S001"}) is None
    missing = next(f for f in result.facts if f.field == "taboo")
    assert missing.missing and not missing.success and missing.value is None


@pytest.mark.parametrize("payload,status", [
    ({"query": "S001", "results": []}, "empty"),
    ({"query": "S001", "results": [], "error": "not found"}, "error"),
    ({"query": "S001", "results": [product("S999")]}, "error"),
    ({"query": "S001", "results": [{"price": "176 元"}]}, "error"),
    ({"query": "S001", "results": [product(), product()]}, "error"),
    ({"query": "S002", "results": [product()]}, "error"),
    ({"query": "S001", "results": "invalid"}, "error"),
    ({"query": "S001", "results": [None]}, "error"),
    ([product()], "error"),
    ("not json", "error"),
    (None, "error"),
])
def test_invalid_or_empty_results_replace_old_facts_without_fallback(store, payload, status):
    facts, _ = store
    record(facts, search_result())
    assert read(facts) is not None
    result = record(facts, payload)
    assert result.status == status
    assert result.facts == ()
    assert read(facts) is None
    assert facts.reusable_result("search_product", {"query": "S001"}) is None


def test_missing_field_replaces_previously_successful_field(store):
    facts, _ = store
    record(facts, search_result())
    payload = search_result()
    payload["results"][0].pop("price")
    record(facts, payload)
    assert read(facts) is None
    assert read(facts, field="spec").value == "S001:spec"


def test_exception_or_unstable_source_cannot_preserve_previous_query_success(store):
    facts, _ = store
    record(facts, search_result())
    record(facts, search_result(), execution_error="read failed")
    assert read(facts) is None
    version = facts.source_version()
    outcome = facts.record_result("search_product", {"query": "S001"}, search_result(),
                                 version_before=version, version_after="changed")
    assert outcome.status == "error" and not outcome.facts
    assert read(facts) is None


def test_exact_query_reuse_requires_unexpired_unchanged_readable_source(store):
    facts, ticks = store
    record(facts, search_result())
    assert facts.reusable_result("search_product", {"query": "S001"}) is not None
    ticks[0] = 1300
    assert facts.reusable_result("search_product", {"query": "S001"}) is None
    assert read(facts) is None
    ticks[0] = 999  # Clock rollback must not extend the evidence's life.
    assert read(facts) is None
    ticks[0] = 1001
    facts.source_path.write_text("[{}]", encoding="utf-8")
    assert read(facts) is None
    assert facts.reusable_result("search_product", {"query": "S001"}) is None
    record(facts, search_result())
    facts.source_path.unlink()
    assert read(facts) is None


def test_search_scope_never_becomes_unfiltered_catalog_or_other_query(store):
    facts, _ = store
    scoped = {"query": "星屿", "results": [product()]}
    record(facts, scoped, args={"query": "星屿"})
    assert read(facts, args={"query": "星屿"}) is not None
    assert read(facts, args={"query": ""}) is None
    assert read(facts, args={"query": "S001"}) is None
    assert facts.reusable_result("search_product", {"query": "星屿"}) is None
    unsupported = record(facts, scoped, args={"query": "星屿", "budget": 100})
    assert unsupported.status == "error" and not unsupported.facts


def comparison(ids=("S002", "S001")):
    rows = [product(sku) for sku in ids]
    return {
        "sku_ids": list(ids),
        "basic_difference": {field: {"sku_a": rows[0][field], "sku_b": rows[1][field]}
                             for field in PRODUCT_FIELDS[:6]},
        **{block: {side: {"sku_id": row["sku_id"], "value": row[field]}
                   for side, row in zip(("sku_a", "sku_b"), rows)}
           for field, block in (("suitable_crowd", "suitable_crowd_difference"),
                                ("taboo", "taboo_difference"),
                                ("product_desc", "product_position_difference"))},
    }


def test_comparison_fields_belong_to_each_sku_in_actual_order(store):
    facts, _ = store
    args = {"sku_ids": "S002,S001"}
    result = record(facts, comparison(), tool="compare_product", args=args)
    assert result.status == "success"
    assert result.result["sku_ids"] == ["S002", "S001"]
    for sku in ("S002", "S001"):
        for field in ("price", "spec", "taboo"):
            assert read(facts, sku, field, tool="compare_product", args=args).value == f"{sku}:{field}"
    assert facts.reusable_result("compare_product", args) is None
    assert read(facts, "S001", tool="compare_product", args={"sku_ids": "S001,S002"}) is None
    swapped = comparison()
    swapped["taboo_difference"]["sku_a"]["sku_id"] = "S001"
    assert record(facts, swapped, tool="compare_product", args=args).status == "error"
    assert read(facts, "S002", tool="compare_product", args=args) is None


def test_mutation_and_clear_do_not_leak_evidence(store):
    facts, _ = store
    payload = search_result()
    result = record(facts, payload)
    payload["results"][0]["price"] = "fake"
    result.result["results"][0]["price"] = "fake"
    copy = facts.reusable_result("search_product", {"query": "S001"})
    copy.result["results"][0]["price"] = "fake"
    price = read(facts)
    price.query_args["query"] = "S002"
    assert read(facts).value == "S001:price"
    assert read(facts).query_args == {"query": "S001"}
    facts.clear()
    assert facts.get_query("search_product", {"query": "S001"}) is None
    assert read(facts) is None
