"""Tests for forecast evaluation metrics."""

import pandas as pd
import pytest

from macro_data_forecasting.evaluation.metrics import (
    compare_to_naive,
    evaluate_forecasts,
)


def _forecasts(predictions: list[float], model_name: str = "ridge") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_timestamp": pd.date_range("2026-01-01", periods=3, freq="MS"),
            "target_id": ["cpi_mom"] * 3,
            "target_reference_date": pd.date_range(
                "2025-12-01",
                periods=3,
                freq="MS",
            ),
            "target_release_date": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="MS",
            ),
            "actual": [1.0, 2.0, 4.0],
            "prediction": predictions,
            "model_name": [model_name] * 3,
            "fold_number": [1, 2, 3],
        },
    )


def test_evaluate_forecasts_computes_rmse_and_mae() -> None:
    """Verify forecast metrics compute RMSE and MAE."""
    metrics = evaluate_forecasts(_forecasts([1.0, 1.0, 5.0]))

    assert metrics["rmse"] == pytest.approx((2 / 3) ** 0.5)
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["n_forecasts"] == 3.0


def test_compare_to_naive_compares_aligned_forecasts() -> None:
    """Verify model and naive forecasts are compared on aligned rows."""
    model = _forecasts([1.0, 2.0, 4.0], "ridge")
    naive = _forecasts([0.0, 1.0, 2.0], "naive_last_value")

    comparison = compare_to_naive(model, naive)

    assert comparison["model_rmse"] == pytest.approx(0.0)
    assert comparison["naive_rmse"] > comparison["model_rmse"]
    assert comparison["rmse_improvement"] > 0


def test_compare_to_naive_reports_whether_model_beats_naive() -> None:
    """Verify comparison reports model wins as numeric indicators."""
    model = _forecasts([1.0, 2.0, 4.0], "ridge")
    naive = _forecasts([0.0, 1.0, 2.0], "naive_last_value")

    comparison = compare_to_naive(model, naive)

    assert comparison["model_beats_naive_rmse"] == 1.0
    assert comparison["model_beats_naive_mae"] == 1.0
