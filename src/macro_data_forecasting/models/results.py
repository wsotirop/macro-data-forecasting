"""Typed result containers for model validation."""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class ForecastResult:
    """One forecast generated during walk-forward validation."""

    forecast_timestamp: date | datetime
    target_id: str
    target_reference_date: date | datetime
    target_release_date: date | datetime
    actual: float
    prediction: float
    model_name: str
    fold_number: int


@dataclass(frozen=True)
class ValidationSummary:
    """Aggregate validation metrics for one model."""

    model_name: str
    n_forecasts: int
    rmse: float
    mae: float
    directional_accuracy: float
