"""Tests for feature dataset diagnostics."""

import pandas as pd
import pytest

from macro_data_forecasting.features.diagnostics import summarize_feature_missingness


def test_summarize_feature_missingness_returns_one_row_per_feature() -> None:
    """Verify feature missingness summary includes expected fields."""
    dataset = pd.DataFrame(
        {
            "forecast_timestamp": pd.date_range("2026-01-01", periods=4, freq="MS"),
            "target_value": [1.0, 2.0, 3.0, 4.0],
            "feature_a": [None, 1.0, 2.0, None],
            "feature_b": [None, None, None, None],
        },
    )

    summary = summarize_feature_missingness(dataset)

    feature_a = summary.loc[summary["feature_name"] == "feature_a"].iloc[0]
    feature_b = summary.loc[summary["feature_name"] == "feature_b"].iloc[0]
    assert len(summary) == 2
    assert feature_a["missing_count"] == 2
    assert feature_a["missing_pct"] == pytest.approx(50.0)
    assert feature_a["first_valid_timestamp"] == pd.Timestamp("2026-02-01")
    assert feature_a["last_valid_timestamp"] == pd.Timestamp("2026-03-01")
    assert not feature_a["all_missing"]
    assert feature_b["missing_count"] == 4
    assert feature_b["missing_pct"] == pytest.approx(100.0)
    assert feature_b["all_missing"]


def test_summarize_feature_missingness_requires_forecast_timestamp() -> None:
    """Verify diagnostics require forecast timestamps."""
    with pytest.raises(ValueError, match="forecast_timestamp"):
        summarize_feature_missingness(pd.DataFrame({"feature_a": [1.0]}))


def test_summarize_feature_missingness_handles_no_feature_columns() -> None:
    """Verify diagnostics return an empty schema when no features exist."""
    summary = summarize_feature_missingness(
        pd.DataFrame({"forecast_timestamp": ["2026-01-01"], "target_value": [1.0]}),
    )

    assert summary.empty
    assert list(summary.columns) == [
        "feature_name",
        "missing_count",
        "missing_pct",
        "first_valid_timestamp",
        "last_valid_timestamp",
        "all_missing",
    ]
