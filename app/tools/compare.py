"""LangChain tool for comparing two local SKU records.

This module exposes a LangChain 1.x compatible Tool named ``compare_product``.
It reads product data from ``data/sku.json`` through the shared data loader and
returns structured differences without making medical judgments.
"""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from app.tools.data_loader import load_sku_data


BASIC_FIELDS = {
    "name": "商品名称",
    "category": "类目",
    "spec": "规格",
    "price": "价格",
    "stock": "库存",
    "delivery": "发货方式",
}


def _parse_sku_ids(raw_sku_ids: str) -> tuple[str, str]:
    """Parse two comma-separated SKU IDs."""
    sku_ids = [sku_id.strip() for sku_id in raw_sku_ids.split(",") if sku_id.strip()]
    if len(sku_ids) != 2:
        raise ValueError("Please provide exactly two SKU IDs separated by a comma.")
    return sku_ids[0], sku_ids[1]


def _find_product(products: list[dict[str, Any]], sku_id: str) -> dict[str, Any] | None:
    """Find one product by SKU ID using case-insensitive matching."""
    normalized_sku_id = sku_id.strip().lower()
    for product in products:
        if str(product.get("sku_id", "")).strip().lower() == normalized_sku_id:
            return product
    return None


def _basic_difference(product_a: dict[str, Any], product_b: dict[str, Any]) -> dict[str, Any]:
    """Build the basic product comparison block."""
    return {
        field: {
            "label": label,
            "sku_a": product_a.get(field, ""),
            "sku_b": product_b.get(field, ""),
        }
        for field, label in BASIC_FIELDS.items()
    }


def _build_comparison(product_a: dict[str, Any], product_b: dict[str, Any]) -> dict[str, Any]:
    """Build structured differences from two original SKU records."""
    sku_a = product_a.get("sku_id", "")
    sku_b = product_b.get("sku_id", "")

    return {
        "basic_difference": _basic_difference(product_a, product_b),
        "suitable_crowd_difference": {
            "sku_a": {"sku_id": sku_a, "value": product_a.get("suitable_crowd", "")},
            "sku_b": {"sku_id": sku_b, "value": product_b.get("suitable_crowd", "")},
        },
        "taboo_difference": {
            "sku_a": {"sku_id": sku_a, "value": product_a.get("taboo", "")},
            "sku_b": {"sku_id": sku_b, "value": product_b.get("taboo", "")},
        },
        "product_position_difference": {
            "sku_a": {"sku_id": sku_a, "value": product_a.get("product_desc", "")},
            "sku_b": {"sku_id": sku_b, "value": product_b.get("product_desc", "")},
            "summary": (
                f"{sku_a} 基于商品描述定位为：{product_a.get('product_desc', '')}；"
                f"{sku_b} 基于商品描述定位为：{product_b.get('product_desc', '')}"
            ),
        },
    }


@tool(
    "compare_product",
    description=(
        "Compare two local SKU records by comma-separated SKU IDs, such as "
        "'S001,S002'. Returns structured differences from data/sku.json."
    ),
)
def compare_product(sku_ids: str) -> dict[str, Any]:
    """Compare two products by SKU ID."""
    try:
        first_sku_id, second_sku_id = _parse_sku_ids(sku_ids)
        products = load_sku_data()

        first_product = _find_product(products, first_sku_id)
        second_product = _find_product(products, second_sku_id)

        missing_sku_ids = [
            sku_id
            for sku_id, product in (
                (first_sku_id, first_product),
                (second_sku_id, second_product),
            )
            if product is None
        ]
        if missing_sku_ids:
            return {
                "sku_ids": [first_sku_id, second_sku_id],
                "error": f"SKU not found: {', '.join(missing_sku_ids)}",
            }

        return {
            "sku_ids": [first_sku_id, second_sku_id],
            **_build_comparison(first_product, second_product),
        }
    except FileNotFoundError as exc:
        return {"sku_ids": sku_ids, "error": f"SKU data file not found: {exc}"}
    except ValueError as exc:
        return {"sku_ids": sku_ids, "error": str(exc)}
    except Exception as exc:
        return {"sku_ids": sku_ids, "error": f"Product comparison failed: {exc}"}


def compare_skus(sku_ids: list[str]) -> dict[str, Any]:
    """Compatibility wrapper for direct Python calls during local development."""
    return compare_product.invoke({"sku_ids": ",".join(sku_ids)})

