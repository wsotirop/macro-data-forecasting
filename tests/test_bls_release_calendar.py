"""Tests for CPI release calendar validation and coverage."""

from datetime import date

import pandas as pd
import pytest

from macro_data_forecasting.cli import main
from macro_data_forecasting.sources.bls_release_calendar import (
    CalendarCoverageError,
    assert_calendar_coverage,
    load_cpi_release_calendar,
    map_cpi_release_dates,
    normalize_reference_period,
    validate_cpi_release_calendar,
)

SAMPLE_CALENDAR_PATH = "data/reference/cpi_release_calendar_sample.csv"


def _valid_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "release_name": ["Consumer Price Index", "Consumer Price Index"],
            "reference_period": ["2026-03", "2026-04"],
            "release_date": ["2026-04-10", "2026-05-12"],
            "release_time": ["08:30", "08:30"],
        },
    )


def test_normalize_reference_period_from_string_date_and_timestamp() -> None:
    """Verify reference periods normalize to YYYY-MM."""
    assert normalize_reference_period("2026-04") == "2026-04"
    assert normalize_reference_period(date(2026, 4, 1)) == "2026-04"
    assert normalize_reference_period(pd.Timestamp("2026-04-15")) == "2026-04"


def test_validate_cpi_release_calendar_succeeds_on_sample() -> None:
    """Verify the sample CPI release calendar validates."""
    calendar = load_cpi_release_calendar(SAMPLE_CALENDAR_PATH)

    assert list(calendar["reference_period"]) == [
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
    ]
    assert calendar.loc[3, "release_date"].isoformat() == "2026-05-12"


def test_validate_cpi_release_calendar_duplicate_period_raises() -> None:
    """Verify duplicate reference periods are rejected."""
    calendar = _valid_calendar()
    duplicate = pd.concat([calendar, calendar.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate reference_period"):
        validate_cpi_release_calendar(duplicate)


def test_validate_cpi_release_calendar_missing_column_raises() -> None:
    """Verify calendars without required columns are rejected."""
    calendar = _valid_calendar().drop(columns=["release_time"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_cpi_release_calendar(calendar)


def test_validate_cpi_release_calendar_before_month_end_raises() -> None:
    """Verify release dates before month end are rejected."""
    calendar = pd.DataFrame(
        {
            "release_name": ["Consumer Price Index"],
            "reference_period": ["2026-04"],
            "release_date": ["2026-04-15"],
            "release_time": ["08:30"],
        },
    )

    with pytest.raises(ValueError, match="after the reference month ends"):
        validate_cpi_release_calendar(calendar)


def test_assert_calendar_coverage_passes_when_observations_are_covered() -> None:
    """Verify coverage passes when all observation months are in the calendar."""
    observations = pd.DataFrame({"date": ["2026-03-01", "2026-04-01"]})

    assert assert_calendar_coverage(observations, _valid_calendar()) is None


def test_assert_calendar_coverage_raises_when_months_are_missing() -> None:
    """Verify strict coverage raises when observation months are missing."""
    observations = pd.DataFrame({"date": ["2026-02-01", "2026-04-01"]})

    with pytest.raises(CalendarCoverageError, match="2026-02"):
        assert_calendar_coverage(observations, _valid_calendar())


def test_map_cpi_release_dates_uses_validated_calendar() -> None:
    """Verify release dates are filled from a validated calendar."""
    observations = pd.DataFrame(
        {
            "series_id": ["CUSR0000SA0", "CUSR0000SA0"],
            "date": ["2026-03-01", "2026-04-01"],
            "value": [319.0, 320.0],
            "source": ["bls", "bls"],
            "release_date": [pd.NaT, pd.NaT],
            "fetched_at": [pd.Timestamp("2026-05-22T12:00:00Z")] * 2,
        },
    )

    mapped = map_cpi_release_dates(observations, _valid_calendar())

    assert mapped.loc[0, "release_date"].isoformat() == "2026-04-10"
    assert mapped.loc[1, "release_date"].isoformat() == "2026-05-12"


def test_cli_validate_cpi_calendar_succeeds_on_sample(capsys) -> None:
    """Verify the CLI validates the sample CPI release calendar."""
    result = main(
        [
            "validate-cpi-calendar",
            "--calendar",
            SAMPLE_CALENDAR_PATH,
            "--start-period",
            "2026-04",
            "--end-period",
            "2026-04",
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Calendar rows: 4" in output
    assert "Reference period range: 2026-01 to 2026-04" in output
    assert "Release date range: 2026-02-12 to 2026-05-12" in output
