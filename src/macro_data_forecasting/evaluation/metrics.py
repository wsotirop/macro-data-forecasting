"""Forecast evaluation metrics."""

import numpy as np
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
