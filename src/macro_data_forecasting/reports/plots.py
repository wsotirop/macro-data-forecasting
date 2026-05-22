"""Matplotlib plots for forecast reports."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_predictions_vs_actuals(
    forecasts: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a line plot of actual values and model predictions."""
    prepared = _prepare_forecasts(forecasts)
    path = _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(10, 5))
    actuals = (
        prepared.sort_values("forecast_timestamp")
        .drop_duplicates(subset=["forecast_timestamp"])
        .loc[:, ["forecast_timestamp", "actual"]]
    )
    axis.plot(
        actuals["forecast_timestamp"],
        actuals["actual"],
        label="actual",
        linewidth=2,
    )
    for model_name, model_rows in prepared.groupby("model_name", sort=True):
        model_rows = model_rows.sort_values("forecast_timestamp")
        axis.plot(
            model_rows["forecast_timestamp"],
            model_rows["prediction"],
            label=f"{model_name} prediction",
        )
    axis.set_title("Predictions vs Actuals")
    axis.set_xlabel("Forecast timestamp")
    axis.set_ylabel("Target value")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_forecast_errors(
    forecasts: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a line plot of prediction minus actual forecast errors."""
    prepared = _prepare_forecasts(forecasts)
    prepared["error"] = prepared["prediction"] - prepared["actual"]
    path = _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(10, 5))
    for model_name, model_rows in prepared.groupby("model_name", sort=True):
        model_rows = model_rows.sort_values("forecast_timestamp")
        axis.plot(
            model_rows["forecast_timestamp"],
            model_rows["error"],
            label=f"{model_name} error",
        )
    axis.axhline(0.0, linestyle="--", linewidth=1)
    axis.set_title("Forecast Errors")
    axis.set_xlabel("Forecast timestamp")
    axis.set_ylabel("Prediction minus actual")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_metric_comparison(
    metrics: pd.DataFrame,
    output_path: str | Path,
    metric: str = "rmse",
) -> Path:
    """Save a bar chart comparing models on a selected metric."""
    if metrics.empty:
        msg = "Metric comparison plot requires at least one metrics row."
        raise ValueError(msg)
    if "model_name" not in metrics:
        msg = "Metric comparison plot requires a model_name column."
        raise ValueError(msg)
    if metric not in {"rmse", "mae"}:
        msg = f"Unsupported metric for comparison plot: {metric}"
        raise ValueError(msg)
    if metric not in metrics:
        msg = f"Metric comparison plot missing required column: {metric}"
        raise ValueError(msg)

    prepared = metrics.loc[:, ["model_name", metric]].copy()
    prepared[metric] = pd.to_numeric(prepared[metric], errors="coerce")
    if prepared[metric].dropna().empty:
        msg = f"Metric comparison plot has no numeric values for {metric}."
        raise ValueError(msg)

    path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(prepared["model_name"].astype(str), prepared[metric])
    axis.set_title(f"{metric.upper()} by Model")
    axis.set_xlabel("Model")
    axis.set_ylabel(metric.upper())
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _prepare_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        msg = "Forecast plot requires at least one forecast row."
        raise ValueError(msg)
    required = ["forecast_timestamp", "actual", "prediction", "model_name"]
    missing = [column for column in required if column not in forecasts]
    if missing:
        msg = f"Forecast plot missing required columns: {missing}"
        raise ValueError(msg)

    prepared = forecasts.copy()
    prepared["forecast_timestamp"] = pd.to_datetime(
        prepared["forecast_timestamp"],
        errors="raise",
    )
    prepared["actual"] = pd.to_numeric(prepared["actual"], errors="raise")
    prepared["prediction"] = pd.to_numeric(prepared["prediction"], errors="raise")
    return prepared.dropna(subset=["forecast_timestamp", "actual", "prediction"])


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
