"""Target construction utilities for macro forecasting."""

import pandas as pd

TARGET_COLUMNS = [
    "target_id",
    "target_name",
    "series_id",
    "source",
    "reference_date",
    "release_date",
    "target_value",
    "target_units",
]


class TargetConstructionError(ValueError):
    """Raised when target construction would be ambiguous or biased."""


def _require_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        msg = f"Missing required observation columns: {missing}"
        raise TargetConstructionError(msg)


def build_cpi_mom_target(
    observations: pd.DataFrame,
    series_id: str = "CUSR0000SA0",
    source: str = "bls",
    strict_release_dates: bool = True,
) -> pd.DataFrame:
    """Build headline CPI month-over-month inflation targets."""
    _require_columns(
        observations,
        ["series_id", "source", "date", "value", "release_date"],
    )
    filtered = observations.loc[
        (observations["series_id"] == series_id) & (observations["source"] == source),
        ["series_id", "source", "date", "value", "release_date"],
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    filtered["date"] = pd.to_datetime(filtered["date"], errors="raise")
    raw_release_dates = filtered["release_date"].copy()
    filtered["release_date"] = pd.to_datetime(
        filtered["release_date"],
        errors="coerce",
    )
    invalid_release_dates = raw_release_dates.notna() & filtered["release_date"].isna()
    if invalid_release_dates.any():
        msg = "release_date contains invalid date values."
        raise TargetConstructionError(msg)
    filtered["value"] = pd.to_numeric(filtered["value"], errors="raise")

    if strict_release_dates and filtered["release_date"].isna().any():
        msg = (
            "CPI target construction requires release_date for every observation "
            "in strict mode."
        )
        raise TargetConstructionError(msg)

    duplicate_keys = filtered.duplicated(["date", "release_date"], keep=False)
    if duplicate_keys.any():
        duplicates = filtered.loc[duplicate_keys, ["date", "release_date"]]
        msg = (
            "Duplicate CPI observations found for the same date and release_date: "
            f"{duplicates.to_dict(orient='records')}"
        )
        raise TargetConstructionError(msg)

    filtered = filtered.sort_values(["date", "release_date"]).reset_index(drop=True)
    filtered["target_value"] = 100 * (
        filtered["value"] / filtered["value"].shift(1) - 1
    )
    filtered = filtered.dropna(subset=["target_value"]).copy()

    targets = pd.DataFrame(
        {
            "target_id": "cpi_mom",
            "target_name": "Headline CPI month-over-month inflation",
            "series_id": filtered["series_id"],
            "source": filtered["source"],
            "reference_date": filtered["date"].dt.date,
            "release_date": filtered["release_date"].dt.date,
            "target_value": filtered["target_value"],
            "target_units": "percent",
        },
        columns=TARGET_COLUMNS,
    )
    return targets.reset_index(drop=True)
