"""Walk-forward validation for time-series macro forecasts."""

import warnings

import pandas as pd

from macro_data_forecasting.models.baselines import fit_predict_ridge

FORECAST_COLUMNS = [
    "forecast_timestamp",
    "target_id",
    "target_reference_date",
    "target_release_date",
    "actual",
    "prediction",
    "model_name",
    "fold_number",
]

REQUIRED_DATASET_COLUMNS = [
    "forecast_timestamp",
    "target_id",
    "target_reference_date",
    "target_release_date",
    "target_value",
]


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_DATASET_COLUMNS if column not in dataset]
    if missing:
        msg = f"Dataset missing required validation columns: {missing}"
        raise ValueError(msg)

    prepared = dataset.copy()
    prepared["forecast_timestamp"] = pd.to_datetime(
        prepared["forecast_timestamp"],
        errors="raise",
    )
    prepared["target_reference_date"] = pd.to_datetime(
        prepared["target_reference_date"],
        errors="raise",
    )
    prepared["target_release_date"] = pd.to_datetime(
        prepared["target_release_date"],
        errors="raise",
    )
    prepared["target_value"] = pd.to_numeric(
        prepared["target_value"],
        errors="coerce",
    )
    missing_target_count = int(prepared["target_value"].isna().sum())
    if missing_target_count:
        warnings.warn(
            f"Dropping {missing_target_count} rows with missing target_value.",
            RuntimeWarning,
            stacklevel=2,
        )
        prepared = prepared.dropna(subset=["target_value"]).copy()
    return prepared.sort_values("forecast_timestamp").reset_index(drop=True)


def _resolve_feature_columns(
    dataset: pd.DataFrame,
    feature_columns: list[str] | None,
) -> list[str]:
    resolved = (
        [column for column in dataset.columns if column.startswith("feature_")]
        if feature_columns is None
        else feature_columns
    )
    if not resolved:
        msg = "No feature columns available for ridge validation."
        raise ValueError(msg)
    missing = [column for column in resolved if column not in dataset.columns]
    if missing:
        msg = f"Requested feature columns are missing: {missing}"
        raise ValueError(msg)
    return resolved


def walk_forward_validate(
    dataset: pd.DataFrame,
    model_name: str,
    min_train_size: int = 24,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Run expanding-window walk-forward validation with prior rows only."""
    if min_train_size < 1:
        msg = "min_train_size must be at least 1."
        raise ValueError(msg)
    if model_name not in {"naive_last_value", "ridge"}:
        msg = f"Unsupported model_name: {model_name}"
        raise ValueError(msg)

    prepared = _prepare_dataset(dataset)
    if len(prepared) <= min_train_size:
        msg = (
            "Insufficient rows for walk-forward validation: "
            f"{len(prepared)} rows available, min_train_size={min_train_size}."
        )
        raise ValueError(msg)

    resolved_features: list[str] = []
    if model_name == "ridge":
        resolved_features = _resolve_feature_columns(prepared, feature_columns)

    forecasts: list[dict[str, object]] = []
    fold_number = 1
    for row_index in range(min_train_size, len(prepared)):
        train = prepared.iloc[:row_index].copy()
        test = prepared.iloc[[row_index]].copy()
        if model_name == "naive_last_value":
            prediction = float(train["target_value"].iloc[-1])
        else:
            prediction = float(
                fit_predict_ridge(
                    train,
                    test,
                    resolved_features,
                    target_column="target_value",
                )[0],
            )

        row = test.iloc[0]
        forecasts.append(
            {
                "forecast_timestamp": row["forecast_timestamp"],
                "target_id": row["target_id"],
                "target_reference_date": row["target_reference_date"],
                "target_release_date": row["target_release_date"],
                "actual": float(row["target_value"]),
                "prediction": prediction,
                "model_name": model_name,
                "fold_number": fold_number,
            },
        )
        fold_number += 1

    return pd.DataFrame(forecasts, columns=FORECAST_COLUMNS)


def expanding_window_walk_forward(
    model_name: str,
    dataset: pd.DataFrame,
    min_train_size: int = 24,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper for expanding walk-forward validation."""
    return walk_forward_validate(
        dataset=dataset,
        model_name=model_name,
        min_train_size=min_train_size,
        feature_columns=feature_columns,
    )
