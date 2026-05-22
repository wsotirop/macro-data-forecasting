"""CPI release-calendar helpers for BLS observations."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_CALENDAR_COLUMNS = (
    "release_name",
    "reference_period",
    "release_date",
    "release_time",
)


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
    _require_calendar_columns(calendar)
    calendar = calendar.loc[:, REQUIRED_CALENDAR_COLUMNS].copy()
    calendar["reference_period"] = calendar["reference_period"].astype(str)
    calendar["release_date"] = pd.to_datetime(
        calendar["release_date"],
        errors="raise",
    ).dt.date
    calendar["release_time"] = calendar["release_time"].astype(str)
    return calendar


def map_cpi_release_dates(
    observations: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Fill CPI observation release dates from reference-period calendar matches."""
    _require_calendar_columns(calendar)
    if "date" not in observations.columns or "release_date" not in observations.columns:
        msg = "Observations must include date and release_date columns."
        raise ValueError(msg)

    mapped = observations.copy()
    mapped["release_date"] = mapped["release_date"].astype(object)
    reference_periods = pd.to_datetime(mapped["date"], errors="raise").dt.strftime(
        "%Y-%m",
    )
    release_map = calendar.set_index("reference_period")["release_date"]
    release_dates = reference_periods.map(release_map)
    matched = release_dates.notna()
    mapped.loc[matched, "release_date"] = release_dates.loc[matched]
    return mapped
