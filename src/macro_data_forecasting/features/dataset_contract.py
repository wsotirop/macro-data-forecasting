"""Dataset contract helpers for point-in-time modeling datasets."""

import pandas as pd

TARGET_REQUIRED_COLUMNS = [
    "target_id",
    "target_name",
    "series_id",
    "source",
    "reference_date",
    "release_date",
    "target_value",
    "target_units",
]

FEATURE_DATASET_BASE_COLUMNS = [
    "forecast_timestamp",
    "target_id",
    "target_reference_date",
    "target_release_date",
    "target_value",
]


def _require_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        msg = f"Missing required target columns: {missing}"
        raise ValueError(msg)


def validate_target_frame(targets: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize target rows for point-in-time datasets."""
    _require_columns(targets, TARGET_REQUIRED_COLUMNS)
    normalized = targets.loc[:, TARGET_REQUIRED_COLUMNS].copy()
    normalized["reference_date"] = pd.to_datetime(
        normalized["reference_date"],
        errors="raise",
    )
    normalized["release_date"] = pd.to_datetime(
        normalized["release_date"],
        errors="coerce",
    )
    if normalized["release_date"].isna().any():
        msg = "target release_date cannot be missing."
        raise ValueError(msg)

    normalized["target_value"] = pd.to_numeric(
        normalized["target_value"],
        errors="raise",
    )
    duplicate_targets = normalized.duplicated(
        ["target_id", "reference_date"],
        keep=False,
    )
    if duplicate_targets.any():
        duplicates = normalized.loc[
            duplicate_targets,
            ["target_id", "reference_date"],
        ]
        msg = (
            "Duplicate target rows found for target_id and reference_date: "
            f"{duplicates.to_dict(orient='records')}"
        )
        raise ValueError(msg)

    cpi_targets = normalized["target_id"] == "cpi_mom"
    reference_month_end = (
        normalized.loc[cpi_targets, "reference_date"] + pd.offsets.MonthEnd(0)
    )
    invalid_release_dates = (
        normalized.loc[cpi_targets, "release_date"] <= reference_month_end
    )
    if invalid_release_dates.any():
        invalid_rows = normalized.loc[cpi_targets].loc[
            invalid_release_dates,
            ["target_id", "reference_date", "release_date"],
        ]
        msg = (
            "Monthly CPI target release_date must be after the reference month "
            f"end: {invalid_rows.to_dict(orient='records')}"
        )
        raise ValueError(msg)

    normalized["reference_date"] = normalized["reference_date"].dt.date
    normalized["release_date"] = normalized["release_date"].dt.date
    return normalized.sort_values(["target_id", "reference_date"]).reset_index(
        drop=True,
    )


def create_empty_feature_dataset(targets: pd.DataFrame) -> pd.DataFrame:
    """Create a model-dataset shell without real feature columns."""
    validated = validate_target_frame(targets)
    dataset = pd.DataFrame(
        {
            "forecast_timestamp": validated["release_date"],
            "target_id": validated["target_id"],
            "target_reference_date": validated["reference_date"],
            "target_release_date": validated["release_date"],
            "target_value": validated["target_value"],
        },
        columns=FEATURE_DATASET_BASE_COLUMNS,
    )
    return dataset
