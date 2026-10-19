"""Basic tests for the local product comparison LangChain Tool.

These tests verify that compare_product reads local SKU data and returns a
stable structured comparison without invoking the Agent layer.
"""

from app.tools.compare import compare_product


REQUIRED_COMPARISON_KEYS = {
    "basic_difference",
    "suitable_crowd_difference",
    "taboo_difference",
    "product_position_difference",
}


def test_compare_two_existing_skus_returns_four_blocks() -> None:
    """Normal comparison should return all required structured blocks."""
    result = compare_product.invoke({"sku_ids": "S001,S002"})

    assert "error" not in result
    assert REQUIRED_COMPARISON_KEYS.issubset(result.keys())
    assert result["sku_ids"] == ["S001", "S002"]


def test_compare_missing_sku_returns_error() -> None:
    """Missing SKU IDs should return a controlled error response."""
    result = compare_product.invoke({"sku_ids": "S001,S999"})

    assert "error" in result
    assert "S999" in result["error"]


def test_compare_empty_input_returns_error_without_crashing() -> None:
    """Invalid input format should not crash the tool."""
    result = compare_product.invoke({"sku_ids": ""})

    assert "error" in result
