"""Forecast evaluation metrics."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _as_float_arrays(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        msg = f"Shape mismatch: y_true has {actual.shape}, y_pred has {predicted.shape}"
        raise ValueError(msg)
    if actual.size == 0:
        msg = "Metric inputs must contain at least one observation."
        raise ValueError(msg)
    return actual, predicted


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the mean absolute forecast error."""
    actual, predicted = _as_float_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the root mean squared forecast error."""
    actual, predicted = _as_float_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def directional_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the share of matching actual and predicted directions."""
    actual, predicted = _as_float_arrays(y_true, y_pred)
    return float(np.mean(np.sign(actual) == np.sign(predicted)))


def _valid_forecast_arrays(forecasts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [
        column for column in ["actual", "prediction"] if column not in forecasts.columns
    ]
    if missing:
        msg = f"Forecasts missing required metric columns: {missing}"
        raise ValueError(msg)
    valid = forecasts.loc[:, ["actual", "prediction"]].dropna()
    if valid.empty:
        msg = "Forecast metrics require at least one non-missing forecast."
        raise ValueError(msg)
    return (
        valid["actual"].to_numpy(dtype=float),
        valid["prediction"].to_numpy(dtype=float),
    )


def evaluate_forecasts(forecasts: pd.DataFrame) -> dict[str, float]:
    """Evaluate forecast-level predictions with standard metrics."""
    actual, predicted = _valid_forecast_arrays(forecasts)
    if len(actual) < 2:
        direction = float("nan")
    else:
        direction = float(
            np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))),
        )
    return {
        "rmse": root_mean_squared_error(actual, predicted),
        "mae": mean_absolute_error(actual, predicted),
        "directional_accuracy": direction,
        "n_forecasts": float(len(actual)),
    }


def compare_to_naive(
    model_forecasts: pd.DataFrame,
    naive_forecasts: pd.DataFrame,
) -> dict[str, float]:
    """Compare model forecast metrics against aligned naive forecasts."""
    keys = [
        "forecast_timestamp",
        "target_id",
        "target_reference_date",
        "target_release_date",
    ]
    missing_model = [column for column in keys if column not in model_forecasts.columns]
    missing_naive = [column for column in keys if column not in naive_forecasts.columns]
    if missing_model or missing_naive:
        msg = (
            "Forecast comparison requires alignment columns. "
            f"Missing model={missing_model}, missing naive={missing_naive}"
        )
        raise ValueError(msg)

    aligned = model_forecasts.merge(
        naive_forecasts,
        on=keys,
        suffixes=("_model", "_naive"),
    )
    if aligned.empty:
        msg = "No aligned forecasts available for naive comparison."
        raise ValueError(msg)

    model_metrics = evaluate_forecasts(
        aligned.rename(
            columns={"actual_model": "actual", "prediction_model": "prediction"},
        ),
    )
    naive_metrics = evaluate_forecasts(
        aligned.rename(
            columns={"actual_naive": "actual", "prediction_naive": "prediction"},
        ),
    )
    model_rmse = model_metrics["rmse"]
    naive_rmse = naive_metrics["rmse"]
    model_mae = model_metrics["mae"]
    naive_mae = naive_metrics["mae"]
    return {
        "model_rmse": model_rmse,
        "naive_rmse": naive_rmse,
        "rmse_improvement": naive_rmse - model_rmse,
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "mae_improvement": naive_mae - model_mae,
        "model_beats_naive_rmse": 1.0 if model_rmse < naive_rmse else 0.0,
        "model_beats_naive_mae": 1.0 if model_mae < naive_mae else 0.0,
    }
