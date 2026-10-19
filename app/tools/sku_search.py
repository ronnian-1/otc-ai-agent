"""LangChain tool for searching the local SKU knowledge base.

This module exposes a LangChain 1.x compatible Tool named ``search_product``.
It reads product data dynamically from ``data/sku.json`` and returns original
structured product fields for the shopping guide agent.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from langchain.tools import tool

from app.tools.data_loader import SKU_DATA_PATH, load_sku_data

RETURN_FIELDS = (
    "sku_id",
    "name",
    "category",
    "spec",
    "price",
    "stock",
    "delivery",
    "suitable_crowd",
    "taboo",
    "product_desc",
)
def normalize_product_name(text: str) -> str:
    """Ignore typography/spacing only; no guessed aliases or semantic matches."""
    return re.sub(r"[\s\u200b-\u200f\u2060\ufeff]+", "", unicodedata.normalize("NFKC", text)).casefold()


def _matches_product(item: dict[str, Any], query: str) -> bool:
    """Match by exact SKU ID or fuzzy product name."""
    normalized_query = normalize_product_name(query)
    sku_id = str(item.get("sku_id", "")).strip().lower()
    name = normalize_product_name(str(item.get("name", "")))

    return normalized_query == sku_id or bool(name and normalized_query) and (normalized_query in name or name in normalized_query)


def _format_product(item: dict[str, Any]) -> dict[str, Any]:
    """Return original product fields allowed in the search response."""
    return {field: item.get(field, "") for field in RETURN_FIELDS}


@tool(
    "search_product",
    description=(
        "Search local OTC and supplement SKU data by exact SKU ID or fuzzy "
        "product name. Returns structured product information from data/sku.json."
    ),
)
def search_product(query: str) -> dict[str, Any]:
    """Search local product data by SKU ID or product name."""
    try:
        normalized_query = query.strip()
        if not normalized_query:
            return {"query": query, "results": [], "error": "Query must not be empty."}

        products = load_sku_data()
        results = [
            _format_product(item)
            for item in products
            if _matches_product(item, normalized_query)
        ]

        return {"query": query, "results": results}
    except FileNotFoundError:
        return {
            "query": query,
            "results": [],
            "error": f"SKU data file not found: {SKU_DATA_PATH}",
        }
    except json.JSONDecodeError as exc:
        return {
            "query": query,
            "results": [],
            "error": f"SKU data file is not valid JSON: {exc}",
        }
    except Exception as exc:
        return {"query": query, "results": [], "error": f"SKU search failed: {exc}"}


def search_sku(query: str) -> dict[str, Any]:
    """Compatibility wrapper for direct Python calls during local development."""
    return search_product.invoke({"query": query})

