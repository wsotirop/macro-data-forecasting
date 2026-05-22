"""Tests for the source catalog."""

from macro_data_forecasting.sources.catalog import (
    BLS_SERIES_CATALOG,
    FRED_SERIES_CATALOG,
)


def test_catalog_contains_cpiaucsl() -> None:
    """Verify the initial FRED catalog includes headline CPI."""
    entry = FRED_SERIES_CATALOG["CPIAUCSL"]

    assert entry["series_id"] == "CPIAUCSL"
    assert entry["source"] == "fred"
    assert entry["category"] == "inflation"


def test_catalog_contains_cusr0000sa0() -> None:
    """Verify the initial BLS catalog includes headline CPI."""
    entry = BLS_SERIES_CATALOG["CUSR0000SA0"]

    assert entry["series_id"] == "CUSR0000SA0"
    assert entry["source"] == "bls"
    assert entry["category"] == "inflation"
