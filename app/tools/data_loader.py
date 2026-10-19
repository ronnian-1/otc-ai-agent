"""Shared data access helpers for local tool modules.

This module owns dynamic loading of local SKU data from ``data/sku.json`` so
individual LangChain tools do not depend on each other for data access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKU_DATA_PATH = PROJECT_ROOT / "data" / "sku.json"


def load_sku_data() -> list[dict[str, Any]]:
    """Load SKU data from the project data directory."""
    with SKU_DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("data/sku.json must contain a JSON array.")

    return [item for item in data if isinstance(item, dict)]
