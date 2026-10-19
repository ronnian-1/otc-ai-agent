"""Basic tests for shared local SKU data loading.

These tests ensure tool modules can rely on one shared data access layer for
reading ``data/sku.json``.
"""

from app.tools.data_loader import load_sku_data


def test_load_sku_data_returns_items() -> None:
    """SKU data loader should return a non-empty list of records."""
    data = load_sku_data()

    assert isinstance(data, list)
    assert data


def test_load_sku_data_contains_required_seed_skus() -> None:
    """Seed data should retain S001 and S002 for demos and tests."""
    data = load_sku_data()
    sku_ids = {item.get("sku_id") for item in data}

    assert "S001" in sku_ids
    assert "S002" in sku_ids
