"""CPI release-calendar helpers for BLS observations."""

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

REQUIRED_CALENDAR_COLUMNS = (
    "release_name",
    "reference_period",
    "release_date",
    "release_time",
)


class CalendarCoverageError(ValueError):
    """Raised when CPI observations lack required release-calendar coverage."""


@dataclass(frozen=True)
class CPIReleaseCalendar:
    """Container for official CPI release calendar mappings."""

    calendar: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str | Path) -> "CPIReleaseCalendar":
        """Load CPI release calendar mappings from a CSV file."""
        return cls(load_cpi_release_calendar(path))

    def map_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Map CPI observation reference months to release dates."""
        return map_cpi_release_dates(observations, self.calendar)


def _require_calendar_columns(calendar: pd.DataFrame) -> None:
    missing = [
        column for column in REQUIRED_CALENDAR_COLUMNS if column not in calendar.columns
    ]
    if missing:
        msg = f"CPI release calendar missing required columns: {missing}"
        raise ValueError(msg)


def normalize_reference_period(value: str | date | pd.Timestamp) -> str:
    """Normalize a date-like CPI reference period to YYYY-MM format."""
    if pd.isna(value):
        msg = "reference_period cannot be missing."
        raise ValueError(msg)

    if isinstance(value, str):
        value = value.strip()
        if not value:
            msg = "reference_period cannot be blank."
            raise ValueError(msg)

    timestamp = pd.to_datetime(value, errors="raise")
    return timestamp.strftime("%Y-%m")


def validate_cpi_release_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize CPI release calendar rows."""
    _require_calendar_columns(calendar)
    normalized = calendar.loc[:, REQUIRED_CALENDAR_COLUMNS].copy()
    if normalized.empty:
        msg = "CPI release calendar cannot be empty."
        raise ValueError(msg)
    normalized["reference_period"] = normalized["reference_period"].map(
        normalize_reference_period,
    )
    normalized["release_date"] = pd.to_datetime(
        normalized["release_date"],
        errors="raise",
    ).dt.date
    if normalized["release_time"].isna().any():
        msg = "CPI release calendar release_time cannot be missing."
        raise ValueError(msg)
    normalized["release_time"] = normalized["release_time"].astype(str).str.strip()
    if (normalized["release_time"] == "").any():
        msg = "CPI release calendar release_time cannot be blank."
        raise ValueError(msg)

    duplicated = normalized["reference_period"].duplicated(keep=False)
    if duplicated.any():
        periods = sorted(normalized.loc[duplicated, "reference_period"].unique())
        msg = f"CPI release calendar has duplicate reference_period rows: {periods}"
        raise ValueError(msg)

    reference_starts = pd.to_datetime(
        normalized["reference_period"] + "-01",
        errors="raise",
    )
    release_dates = pd.to_datetime(normalized["release_date"], errors="raise")
    reference_ends = reference_starts + pd.offsets.MonthEnd(0)

    before_month_start = release_dates <= reference_starts
    if before_month_start.any():
        periods = sorted(normalized.loc[before_month_start, "reference_period"])
        msg = (
            "CPI release_date must be after the reference month starts for "
            f"periods: {periods}"
        )
        raise ValueError(msg)

    before_or_on_month_end = release_dates <= reference_ends
    if before_or_on_month_end.any():
        periods = sorted(normalized.loc[before_or_on_month_end, "reference_period"])
        msg = (
            "CPI release_date must be after the reference month ends for "
            f"point-in-time safety. Invalid periods: {periods}"
        )
        raise ValueError(msg)

    return normalized.sort_values("reference_period").reset_index(drop=True)


def load_cpi_release_calendar(path: str | Path) -> pd.DataFrame:
    """Load a local CPI release calendar CSV with validated date columns."""
    calendar_path = Path(path)
    if not calendar_path.exists():
        msg = (
            f"CPI release calendar file not found: {calendar_path}. "
            "Provide an official BLS release calendar CSV before mapping dates."
        )
        raise FileNotFoundError(msg)

    calendar = pd.read_csv(calendar_path)
    return validate_cpi_release_calendar(calendar)


def assert_calendar_coverage(
    observations: pd.DataFrame,
    calendar: pd.DataFrame,
    strict: bool = True,
) -> None:
    """Assert every observation reference month exists in the CPI calendar."""
    if "date" not in observations.columns:
        msg = "Observations must include a date column for calendar coverage checks."
        raise ValueError(msg)

    validated_calendar = validate_cpi_release_calendar(calendar)
    observation_dates = pd.to_datetime(observations["date"], errors="raise")
    if observation_dates.isna().any():
        msg = "Observation dates cannot be missing for calendar coverage checks."
        raise ValueError(msg)
    observation_periods = set(observation_dates.dt.strftime("%Y-%m"))
    calendar_periods = set(validated_calendar["reference_period"])
    missing_periods = sorted(observation_periods.difference(calendar_periods))
    if not missing_periods:
        return

    msg = f"CPI release calendar missing reference periods: {missing_periods}"
    if strict:
        raise CalendarCoverageError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def map_cpi_release_dates(
    observations: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Fill CPI observation release dates from reference-period calendar matches."""
    validated_calendar = validate_cpi_release_calendar(calendar)
    if "date" not in observations.columns or "release_date" not in observations.columns:
        msg = "Observations must include date and release_date columns."
        raise ValueError(msg)
    assert_calendar_coverage(observations, validated_calendar, strict=True)

    mapped = observations.copy()
    mapped["release_date"] = mapped["release_date"].astype(object)
    reference_periods = pd.to_datetime(mapped["date"], errors="raise").dt.strftime(
        "%Y-%m",
    )
    release_map = validated_calendar.set_index("reference_period")["release_date"]
    release_dates = reference_periods.map(release_map)
    mapped["release_date"] = release_dates
    return mapped
