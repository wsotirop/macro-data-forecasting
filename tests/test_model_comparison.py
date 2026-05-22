"""Tests for walk-forward model comparison utilities."""

import pandas as pd
import pytest

from macro_data_forecasting.models.comparison import (
    METRICS_COLUMNS,
    ModelComparisonError,
    _align_forecasts_to_naive,
    run_model_comparison,
    save_model_comparison_outputs,
)


def _feature_dataset(rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_timestamp": pd.date_range("2026-01-01", periods=rows, freq="MS"),
            "target_id": ["cpi_mom"] * rows,
            "target_reference_date": pd.date_range(
                "2025-12-01",
                periods=rows,
                freq="MS",
            ),
            "target_release_date": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="MS",
            ),
            "target_value": [float(index) for index in range(rows)],
            "feature_UNRATE_latest": [float(index) for index in range(rows)],
        },
    )


def _forecasts(model_name: str = "ridge") -> pd.DataFrame:
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
            "actual": [1.0, 2.0, 3.0],
            "prediction": [1.0, 2.0, 3.0],
            "model_name": [model_name] * 3,
            "fold_number": [1, 2, 3],
        },
    )


def test_run_model_comparison_returns_forecasts_for_requested_models() -> None:
    """Verify internal naive benchmarking does not leak into requested forecasts."""
    forecasts, _metrics = run_model_comparison(
        _feature_dataset(),
        models=["ridge"],
        min_train_size=4,
    )

    assert set(forecasts["model_name"]) == {"ridge"}


def test_run_model_comparison_metrics_one_row_per_requested_model() -> None:
    """Verify metrics table has one row per requested model."""
    _forecasts, metrics = run_model_comparison(
        _feature_dataset(),
        models=["naive_last_value", "ridge"],
        min_train_size=4,
    )

    assert list(metrics["model_name"]) == ["naive_last_value", "ridge"]
    assert len(metrics) == 2


def test_run_model_comparison_uses_naive_benchmark_when_not_requested() -> None:
    """Verify ridge metrics are benchmarked against internally-run naive forecasts."""
    forecasts, metrics = run_model_comparison(
        _feature_dataset(),
        models=["ridge"],
        min_train_size=4,
    )

    ridge = metrics.iloc[0]
    assert set(forecasts["model_name"]) == {"ridge"}
    assert ridge["model_name"] == "ridge"
    assert pd.notna(ridge["rmse_vs_naive"])
    assert pd.notna(ridge["mae_vs_naive"])


def test_ridge_comparison_aligns_against_naive() -> None:
    """Verify comparison metrics are built on aligned forecast rows."""
    _forecasts, metrics = run_model_comparison(
        _feature_dataset(),
        models=["naive_last_value", "ridge"],
        min_train_size=4,
    )

    naive = metrics.loc[metrics["model_name"] == "naive_last_value"].iloc[0]
    ridge = metrics.loc[metrics["model_name"] == "ridge"].iloc[0]
    assert ridge["rmse_vs_naive"] == pytest.approx(naive["rmse"] - ridge["rmse"])
    assert ridge["mae_vs_naive"] == pytest.approx(naive["mae"] - ridge["mae"])


def test_mismatched_model_and_naive_forecasts_raise() -> None:
    """Verify model and naive forecasts must contain the same forecast rows."""
    model = _forecasts("ridge")
    naive = _forecasts("naive_last_value").iloc[:-1].copy()

    with pytest.raises(ModelComparisonError, match="do not align"):
        _align_forecasts_to_naive(model, naive, "ridge")


def test_metrics_table_includes_expected_columns() -> None:
    """Verify comparison metrics table schema."""
    _forecasts, metrics = run_model_comparison(
        _feature_dataset(),
        models=["ridge"],
        min_train_size=4,
    )

    assert list(metrics.columns) == METRICS_COLUMNS


def test_run_model_comparison_insufficient_data_raises() -> None:
    """Verify insufficient validation rows propagate a clear error."""
    with pytest.raises(ValueError, match="Insufficient rows"):
        run_model_comparison(
            _feature_dataset(rows=4),
            models=["ridge"],
            min_train_size=4,
        )


def test_save_model_comparison_outputs_writes_csv_files(tmp_path) -> None:
    """Verify comparison outputs are saved and returned as paths."""
    forecasts, metrics = run_model_comparison(
        _feature_dataset(),
        models=["naive_last_value", "ridge"],
        min_train_size=4,
    )

    paths = save_model_comparison_outputs(
        forecasts,
        metrics,
        output_dir=tmp_path,
        prefix="baseline",
    )

    assert paths["forecasts"].exists()
    assert paths["metrics"].exists()
    assert paths["forecasts"].name == "baseline_forecasts.csv"
    assert paths["metrics"].name == "baseline_metrics.csv"
