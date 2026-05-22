"""Point-in-time feature matrix construction."""

from collections.abc import Sequence

import pandas as pd

from macro_data_forecasting.features.dataset_contract import (
    FEATURE_DATASET_BASE_COLUMNS,
    TARGET_REQUIRED_COLUMNS,
    create_empty_feature_dataset,
)

OBSERVATION_REQUIRED_COLUMNS = [
    "series_id",
    "source",
    "date",
    "value",
    "release_date",
]


class FeatureMatrixError(ValueError):
    """Raised when point-in-time feature construction is unsafe."""


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        msg = f"Missing required {label} columns: {missing}"
        raise FeatureMatrixError(msg)


def _normalize_targets(targets: pd.DataFrame) -> pd.DataFrame:
    if set(TARGET_REQUIRED_COLUMNS).issubset(targets.columns):
        normalized = create_empty_feature_dataset(targets)
    else:
        _require_columns(targets, FEATURE_DATASET_BASE_COLUMNS, "target dataset")
        normalized = targets.loc[:, FEATURE_DATASET_BASE_COLUMNS].copy()

    normalized["forecast_timestamp"] = pd.to_datetime(
        normalized["forecast_timestamp"],
        errors="coerce",
    )
    normalized["target_reference_date"] = pd.to_datetime(
        normalized["target_reference_date"],
        errors="raise",
    )
    normalized["target_release_date"] = pd.to_datetime(
        normalized["target_release_date"],
        errors="coerce",
    )
    normalized["target_value"] = pd.to_numeric(
        normalized["target_value"],
        errors="raise",
    )
    if normalized["forecast_timestamp"].isna().any():
        msg = "forecast_timestamp cannot be missing for point-in-time features."
        raise FeatureMatrixError(msg)
    if normalized["target_release_date"].isna().any():
        msg = "target_release_date cannot be missing for point-in-time features."
        raise FeatureMatrixError(msg)

    duplicate_targets = normalized.duplicated(
        ["target_id", "target_reference_date"],
        keep=False,
    )
    if duplicate_targets.any():
        msg = "Duplicate target rows found for target_id and target_reference_date."
        raise FeatureMatrixError(msg)

    return normalized.sort_values(["target_id", "target_reference_date"]).reset_index(
        drop=True,
    )


def _normalize_observations(
    observations: pd.DataFrame,
    feature_series: list[str] | None,
    strict_release_dates: bool,
) -> pd.DataFrame:
    _require_columns(observations, OBSERVATION_REQUIRED_COLUMNS, "observation")
    normalized = observations.loc[:, OBSERVATION_REQUIRED_COLUMNS].copy()
    normalized["_row_order"] = range(len(normalized))
    if feature_series is not None:
        normalized = normalized.loc[normalized["series_id"].isin(feature_series)].copy()

    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise")
    raw_release_dates = normalized["release_date"].copy()
    normalized["release_date"] = pd.to_datetime(
        normalized["release_date"],
        errors="coerce",
    )
    invalid_release_dates = (
        raw_release_dates.notna() & normalized["release_date"].isna()
    )
    if invalid_release_dates.any():
        msg = "Observation release_date contains invalid date values."
        raise FeatureMatrixError(msg)
    if strict_release_dates and normalized["release_date"].isna().any():
        msg = "Observation release_date cannot be missing in strict mode."
        raise FeatureMatrixError(msg)

    normalized = normalized.dropna(subset=["release_date"]).copy()
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    return normalized.sort_values(
        ["series_id", "release_date", "date", "source", "_row_order"],
    ).reset_index(drop=True)


def build_point_in_time_feature_matrix(
    targets: pd.DataFrame,
    observations: pd.DataFrame,
    feature_series: list[str] | None = None,
    strict_release_dates: bool = True,
) -> pd.DataFrame:
    """Build latest-value features using only released observations."""
    target_dataset = _normalize_targets(targets)
    normalized_observations = _normalize_observations(
        observations,
        feature_series,
        strict_release_dates,
    )
    series_ids = (
        list(feature_series)
        if feature_series is not None
        else sorted(normalized_observations["series_id"].dropna().unique())
    )

    feature_rows: list[dict[str, object]] = []
    for _, target_row in target_dataset.iterrows():
        forecast_timestamp = target_row["forecast_timestamp"]
        row = target_row.to_dict()
        for series_id in series_ids:
            eligible = normalized_observations.loc[
                (normalized_observations["series_id"] == series_id)
                & (normalized_observations["release_date"] <= forecast_timestamp)
            ]
            feature_name = f"feature_{series_id}_latest"
            if eligible.empty:
                row[feature_name] = float("nan")
                continue

            # Deterministic tie-break: newest release_date wins, then newest
            # reference date, then source lexical order, then original row order.
            latest = eligible.sort_values(
                ["release_date", "date", "source", "_row_order"],
            ).iloc[-1]
            row[feature_name] = latest["value"]
        feature_rows.append(row)

    feature_matrix = pd.DataFrame(feature_rows)
    ordered_columns = FEATURE_DATASET_BASE_COLUMNS + [
        f"feature_{series_id}_latest" for series_id in series_ids
    ]
    return feature_matrix.loc[:, ordered_columns]


def add_lagged_target_features(
    dataset: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged target values without using the current target row."""
    requested_lags = [1] if lags is None else lags
    if any(lag < 1 for lag in requested_lags):
        msg = "Target lags must be positive integers."
        raise FeatureMatrixError(msg)
    _require_columns(dataset, FEATURE_DATASET_BASE_COLUMNS, "feature dataset")

    lagged = dataset.copy()
    lagged["target_reference_date"] = pd.to_datetime(
        lagged["target_reference_date"],
        errors="raise",
    )
    lagged = lagged.sort_values(["target_id", "target_reference_date"]).reset_index(
        drop=True,
    )
    grouped_target_values = lagged.groupby("target_id")["target_value"]
    for lag in requested_lags:
        lagged[f"feature_target_lag_{lag}"] = grouped_target_values.shift(lag)
    return lagged
