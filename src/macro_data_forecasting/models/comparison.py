"""Run and save walk-forward model comparisons."""

from pathlib import Path

import numpy as np
import pandas as pd

from macro_data_forecasting.evaluation.metrics import evaluate_forecasts
from macro_data_forecasting.models.validation import walk_forward_validate

SUPPORTED_COMPARISON_MODELS = {"naive_last_value", "ridge"}
ALIGNMENT_KEYS = ["forecast_timestamp", "target_id", "target_reference_date"]
METRICS_COLUMNS = [
    "model_name",
    "n_forecasts",
    "rmse",
    "mae",
    "directional_accuracy",
    "beats_naive_rmse",
    "beats_naive_mae",
    "rmse_vs_naive",
    "mae_vs_naive",
]


class ModelComparisonError(ValueError):
    """Raised when model comparison inputs cannot be aligned safely."""


def run_model_comparison(
    dataset: pd.DataFrame,
    models: list[str],
    min_train_size: int = 24,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward validation for requested models and compare to naive."""
    requested_models = _validate_models(models)
    models_to_run = list(requested_models)
    if "naive_last_value" not in models_to_run:
        models_to_run.append("naive_last_value")

    forecasts_by_model = {
        model_name: walk_forward_validate(
            dataset,
            model_name=model_name,
            min_train_size=min_train_size,
            feature_columns=feature_columns,
        )
        for model_name in models_to_run
    }
    naive_forecasts = forecasts_by_model["naive_last_value"]

    metrics_rows = [
        _build_metrics_row(
            forecasts_by_model[model_name],
            naive_forecasts,
            model_name,
        )
        for model_name in requested_models
    ]
    requested_forecasts = [forecasts_by_model[model] for model in requested_models]
    all_forecasts = pd.concat(requested_forecasts, ignore_index=True)
    metrics_table = pd.DataFrame(metrics_rows, columns=METRICS_COLUMNS)
    return all_forecasts, metrics_table


def save_model_comparison_outputs(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "model_comparison",
) -> dict[str, Path]:
    """Save forecast-level and metric-level comparison outputs as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "forecasts": output_path / f"{prefix}_forecasts.csv",
        "metrics": output_path / f"{prefix}_metrics.csv",
    }
    forecasts.to_csv(paths["forecasts"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    return paths


def _validate_models(models: list[str]) -> list[str]:
    if not models:
        msg = "At least one model is required for comparison."
        raise ModelComparisonError(msg)

    requested_models: list[str] = []
    for model_name in models:
        if model_name not in SUPPORTED_COMPARISON_MODELS:
            msg = f"Unsupported model for comparison: {model_name}"
            raise ModelComparisonError(msg)
        if model_name not in requested_models:
            requested_models.append(model_name)
    return requested_models


def _build_metrics_row(
    model_forecasts: pd.DataFrame,
    naive_forecasts: pd.DataFrame,
    model_name: str,
) -> dict[str, float | str]:
    aligned = _align_forecasts_to_naive(model_forecasts, naive_forecasts, model_name)
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

    rmse_vs_naive = naive_metrics["rmse"] - model_metrics["rmse"]
    mae_vs_naive = naive_metrics["mae"] - model_metrics["mae"]
    return {
        "model_name": model_name,
        "n_forecasts": int(model_metrics["n_forecasts"]),
        "rmse": model_metrics["rmse"],
        "mae": model_metrics["mae"],
        "directional_accuracy": model_metrics["directional_accuracy"],
        "beats_naive_rmse": 0.0
        if model_name == "naive_last_value"
        else float(model_metrics["rmse"] < naive_metrics["rmse"]),
        "beats_naive_mae": 0.0
        if model_name == "naive_last_value"
        else float(model_metrics["mae"] < naive_metrics["mae"]),
        "rmse_vs_naive": 0.0 if model_name == "naive_last_value" else rmse_vs_naive,
        "mae_vs_naive": 0.0 if model_name == "naive_last_value" else mae_vs_naive,
    }


def _align_forecasts_to_naive(
    model_forecasts: pd.DataFrame,
    naive_forecasts: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    model = _normalize_forecast_alignment_frame(model_forecasts, model_name)
    naive = _normalize_forecast_alignment_frame(naive_forecasts, "naive_last_value")

    model_keys = set(model.loc[:, ALIGNMENT_KEYS].itertuples(index=False, name=None))
    naive_keys = set(naive.loc[:, ALIGNMENT_KEYS].itertuples(index=False, name=None))
    if model_keys != naive_keys:
        missing_from_model = sorted(naive_keys - model_keys)
        missing_from_naive = sorted(model_keys - naive_keys)
        msg = (
            f"Forecast rows for {model_name} do not align with naive benchmark. "
            f"Missing from model={missing_from_model}; "
            f"missing from naive={missing_from_naive}."
        )
        raise ModelComparisonError(msg)

    aligned = model.merge(
        naive,
        on=ALIGNMENT_KEYS,
        suffixes=("_model", "_naive"),
        validate="one_to_one",
    ).sort_values(ALIGNMENT_KEYS)

    has_release_dates = (
        "target_release_date_model" in aligned
        and "target_release_date_naive" in aligned
    )
    if has_release_dates:
        model_release = pd.to_datetime(aligned["target_release_date_model"])
        naive_release = pd.to_datetime(aligned["target_release_date_naive"])
        if not (model_release == naive_release).all():
            msg = f"Forecast target_release_date mismatch for {model_name}."
            raise ModelComparisonError(msg)

    actual_model = aligned["actual_model"].to_numpy(dtype=float)
    actual_naive = aligned["actual_naive"].to_numpy(dtype=float)
    if not np.allclose(actual_model, actual_naive, equal_nan=True):
        msg = f"Forecast actual values do not match naive benchmark for {model_name}."
        raise ModelComparisonError(msg)

    return aligned


def _normalize_forecast_alignment_frame(
    forecasts: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    required = [*ALIGNMENT_KEYS, "target_release_date", "actual", "prediction"]
    missing = [column for column in required if column not in forecasts.columns]
    if missing:
        msg = f"Forecasts for {model_name} missing required columns: {missing}"
        raise ModelComparisonError(msg)

    normalized = forecasts.copy()
    normalized["forecast_timestamp"] = pd.to_datetime(
        normalized["forecast_timestamp"],
        errors="raise",
    )
    normalized["target_reference_date"] = pd.to_datetime(
        normalized["target_reference_date"],
        errors="raise",
    )
    normalized["target_release_date"] = pd.to_datetime(
        normalized["target_release_date"],
        errors="raise",
    )
    duplicates = normalized.duplicated(subset=ALIGNMENT_KEYS)
    if duplicates.any():
        duplicate_keys = normalized.loc[duplicates, ALIGNMENT_KEYS].to_dict("records")
        msg = f"Duplicate forecast alignment rows for {model_name}: {duplicate_keys}"
        raise ModelComparisonError(msg)
    return normalized
