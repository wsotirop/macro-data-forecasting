"""Tests for BLS CPI ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import func, select

from macro_data_forecasting.database import (
    get_engine,
    insert_observations,
    macro_observations,
)
from macro_data_forecasting.sources.bls import NORMALIZED_COLUMNS, BlsClient
from macro_data_forecasting.sources.bls_release_calendar import (
    load_cpi_release_calendar,
    map_cpi_release_dates,
)


@dataclass
class FakeBlsResponse:
    """Minimal BLS response test double."""

    payload: dict[str, Any]
    status_code: int = 200
    text: str = ""

    def json(self) -> dict[str, Any]:
        """Return the configured JSON payload."""
        return self.payload


class FakeBlsSession:
    """Minimal BLS session test double."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the payload returned by future POST requests."""
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        json: dict[str, Any],
        timeout: float,
    ) -> FakeBlsResponse:
        """Record request arguments and return a fake response."""
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeBlsResponse(self.payload)


@pytest.fixture
def bls_payload() -> dict[str, Any]:
    """Return a representative BLS Public Data API payload."""
    return {
        "status": "REQUEST_SUCCEEDED",
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": "CUSR0000SA0",
                    "data": [
                        {
                            "year": "2026",
                            "period": "M04",
                            "periodName": "April",
                            "value": "319.086",
                        },
                        {
                            "year": "2026",
                            "period": "M03",
                            "periodName": "March",
                            "value": "not_available",
                        },
                    ],
                },
            ],
        },
    }


def test_bls_client_parses_monthly_observations(
    bls_payload: dict[str, Any],
) -> None:
    """Verify BLS monthly observations normalize into storage columns."""
    session = FakeBlsSession(bls_payload)
    client = BlsClient(session=session, backoff_seconds=0)

    frame = client.fetch_series_observations(
        series_id="CUSR0000SA0",
        start_year=2026,
        end_year=2026,
    )

    assert list(frame.columns) == NORMALIZED_COLUMNS
    assert frame.loc[0, "series_id"] == "CUSR0000SA0"
    assert frame.loc[0, "date"].isoformat() == "2026-04-01"
    assert frame.loc[0, "value"] == pytest.approx(319.086)
    assert frame.loc[0, "source"] == "bls"
    assert pd.isna(frame.loc[0, "release_date"])
    assert pd.notna(frame.loc[0, "fetched_at"])
    assert session.calls[0]["json"]["seriesid"] == ["CUSR0000SA0"]
    assert session.calls[0]["json"]["startyear"] == "2026"


def test_bls_period_m01_becomes_monthly_reference_date() -> None:
    """Verify BLS M01 period fields become first-of-month dates."""
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": "CUSR0000SA0",
                    "data": [{"year": "2026", "period": "M01", "value": "317.671"}],
                },
            ],
        },
    }
    client = BlsClient(session=FakeBlsSession(payload), backoff_seconds=0)

    frame = client.fetch_series_observations("CUSR0000SA0", 2026, 2026)

    assert frame.loc[0, "date"].isoformat() == "2026-01-01"


def test_bls_missing_or_invalid_values_become_nan(
    bls_payload: dict[str, Any],
) -> None:
    """Verify BLS invalid numeric values are converted to NaN."""
    client = BlsClient(session=FakeBlsSession(bls_payload), backoff_seconds=0)

    frame = client.fetch_series_observations("CUSR0000SA0", 2026, 2026)

    assert pd.isna(frame.loc[1, "value"])


def test_bls_validate_requires_normalized_columns(
    bls_payload: dict[str, Any],
) -> None:
    """Verify validation rejects frames without normalized columns."""
    client = BlsClient(session=FakeBlsSession(bls_payload), backoff_seconds=0)

    with pytest.raises(ValueError, match="Missing required"):
        client.validate(pd.DataFrame({"series_id": ["CUSR0000SA0"]}))


def test_cpi_release_calendar_loads_sample() -> None:
    """Verify the sample CPI release calendar loads with parsed dates."""
    calendar = load_cpi_release_calendar(
        "data/reference/cpi_release_calendar_sample.csv",
    )

    assert "2026-04" in set(calendar["reference_period"])
    assert calendar.loc[0, "release_date"].isoformat() == "2026-05-12"


def test_cpi_release_calendar_maps_release_dates(
    bls_payload: dict[str, Any],
) -> None:
    """Verify CPI release dates are filled by reference month."""
    client = BlsClient(session=FakeBlsSession(bls_payload), backoff_seconds=0)
    observations = client.fetch_series_observations("CUSR0000SA0", 2026, 2026)
    calendar = load_cpi_release_calendar(
        "data/reference/cpi_release_calendar_sample.csv",
    )

    mapped = map_cpi_release_dates(observations, calendar)

    assert mapped.loc[0, "release_date"].isoformat() == "2026-05-12"
    assert mapped.loc[1, "release_date"].isoformat() == "2026-04-10"


def test_bls_observations_insert_into_sqlite(
    bls_payload: dict[str, Any],
    tmp_path,
) -> None:
    """Verify mapped BLS observations insert into a temporary SQLite database."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    client = BlsClient(session=FakeBlsSession(bls_payload), backoff_seconds=0)
    observations = client.fetch_series_observations("CUSR0000SA0", 2026, 2026)
    calendar = load_cpi_release_calendar(
        "data/reference/cpi_release_calendar_sample.csv",
    )
    mapped = map_cpi_release_dates(observations, calendar)

    row_count = insert_observations(client.validate(mapped), database_url=database_url)
    engine = get_engine(database_url)
    with engine.connect() as connection:
        stored_count = connection.scalar(
            select(func.count()).select_from(macro_observations),
        )
    engine.dispose()

    assert row_count == 2
    assert stored_count == 2
