"""Diagnostics for model-ready feature datasets."""

import pandas as pd

FEATURE_MISSINGNESS_COLUMNS = [
    "feature_name",
    "missing_count",
    "missing_pct",
    "first_valid_timestamp",
    "last_valid_timestamp",
    "all_missing",
]


def summarize_feature_missingness(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness for feature columns in a model dataset."""
    if "forecast_timestamp" not in dataset.columns:
        msg = "Feature diagnostics require a forecast_timestamp column."
        raise ValueError(msg)

    feature_columns = [
        column for column in dataset.columns if column.startswith("feature_")
    ]
    if not feature_columns:
        return pd.DataFrame(columns=FEATURE_MISSINGNESS_COLUMNS)

    timestamps = pd.to_datetime(dataset["forecast_timestamp"], errors="raise")
    row_count = len(dataset)
    rows: list[dict[str, object]] = []
    for feature_name in feature_columns:
        valid_mask = dataset[feature_name].notna()
        missing_count = int((~valid_mask).sum())
        valid_timestamps = timestamps.loc[valid_mask]
        all_missing = bool(valid_timestamps.empty)
        rows.append(
            {
                "feature_name": feature_name,
                "missing_count": missing_count,
                "missing_pct": (missing_count / row_count * 100.0)
                if row_count
                else 0.0,
                "first_valid_timestamp": pd.NaT
                if all_missing
                else valid_timestamps.min(),
                "last_valid_timestamp": pd.NaT
                if all_missing
                else valid_timestamps.max(),
                "all_missing": all_missing,
            },
        )
    return pd.DataFrame(rows, columns=FEATURE_MISSINGNESS_COLUMNS)
