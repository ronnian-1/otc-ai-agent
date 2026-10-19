"""Basic tests for the local SKU search LangChain Tool.

These tests verify the current knowledge-base lookup behavior without invoking
the Agent layer or modifying business code.
"""

import pytest

from app.tools.sku_search import search_product


def test_search_by_sku_id_returns_s001() -> None:
    """SKU ID lookup should return the matching product."""
    result = search_product.invoke({"query": "S001"})

    assert result["results"]
    assert result["results"][0]["sku_id"] == "S001"
    assert result["results"][0]["name"] == "演示星屿营养片"


def test_search_by_fuzzy_product_name_matches_s001() -> None:
    """Fuzzy product-name lookup should match S001."""
    result = search_product.invoke({"query": "星屿营养"})

    assert result["results"]
    assert result["results"][0]["sku_id"] == "S001"


def test_search_unknown_product_returns_empty_results() -> None:
    """Unknown product names should return an empty result set."""
    result = search_product.invoke({"query": "不存在的商品名称"})

    assert result["results"] == []
    assert "error" not in result


def test_search_empty_query_returns_error_without_crashing() -> None:
    """Empty queries should return a controlled error response."""
    result = search_product.invoke({"query": ""})

    assert result["results"] == []
    assert "error" in result


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LangChain 1.x validates tool input before function execution; "
        "None is rejected by the generated string schema."
    ),
)
def test_search_none_query_current_langchain_validation_behavior() -> None:
    """Document current LangChain schema validation behavior for None input."""
    search_product.invoke({"query": None})
