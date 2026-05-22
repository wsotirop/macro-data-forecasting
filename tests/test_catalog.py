"""Tests for the source catalog."""

from macro_data_forecasting.sources.catalog import FRED_SERIES_CATALOG


def test_catalog_contains_cpiaucsl() -> None:
    """Verify the initial FRED catalog includes headline CPI."""
    entry = FRED_SERIES_CATALOG["CPIAUCSL"]

    assert entry["series_id"] == "CPIAUCSL"
    assert entry["source"] == "fred"
    assert entry["category"] == "inflation"
