"""Tests for report plotting utilities."""

import pandas as pd
import pytest

from macro_data_forecasting.reports.plots import (
    plot_forecast_errors,
    plot_metric_comparison,
    plot_predictions_vs_actuals,
)


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-01-01", "2026-02-01"],
            ),
            "actual": [1.0, 2.0, 1.0, 2.0],
            "prediction": [0.9, 1.8, 1.2, 2.3],
            "model_name": ["naive_last_value", "naive_last_value", "ridge", "ridge"],
        },
    )


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["naive_last_value", "ridge"],
            "rmse": [1.0, 1.2],
            "mae": [0.8, 0.9],
        },
    )


def test_plot_predictions_vs_actuals_writes_png(tmp_path) -> None:
    """Verify predictions-vs-actuals plot writes a PNG file."""
    output_path = tmp_path / "predictions_vs_actuals.png"

    result = plot_predictions_vs_actuals(_forecasts(), output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_plot_forecast_errors_writes_png(tmp_path) -> None:
    """Verify forecast error plot writes a PNG file."""
    output_path = tmp_path / "forecast_errors.png"

    result = plot_forecast_errors(_forecasts(), output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_plot_metric_comparison_writes_rmse_png(tmp_path) -> None:
    """Verify RMSE metric comparison plot writes a PNG file."""
    output_path = tmp_path / "rmse_comparison.png"

    result = plot_metric_comparison(_metrics(), output_path, metric="rmse")

    assert result == output_path
    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_plot_metric_comparison_writes_mae_png(tmp_path) -> None:
    """Verify MAE metric comparison plot writes a PNG file."""
    output_path = tmp_path / "mae_comparison.png"

    result = plot_metric_comparison(_metrics(), output_path, metric="mae")

    assert result == output_path
    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_plotting_functions_raise_on_empty_data() -> None:
    """Verify plot functions reject empty data clearly."""
    with pytest.raises(ValueError, match="at least one forecast row"):
        plot_predictions_vs_actuals(pd.DataFrame(), "unused.png")

    with pytest.raises(ValueError, match="at least one metrics row"):
        plot_metric_comparison(pd.DataFrame(), "unused.png")


def test_plotting_functions_raise_on_missing_columns() -> None:
    """Verify plot functions reject missing required columns clearly."""
    with pytest.raises(ValueError, match="missing required columns"):
        plot_forecast_errors(_forecasts().drop(columns=["prediction"]), "unused.png")

    with pytest.raises(ValueError, match="missing required column"):
        plot_metric_comparison(_metrics().drop(columns=["rmse"]), "unused.png")
