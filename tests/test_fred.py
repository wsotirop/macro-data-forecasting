"""Tests for FRED/ALFRED ingestion."""

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
from macro_data_forecasting.sources.fred import NORMALIZED_COLUMNS, FredClient


@dataclass
class FakeResponse:
    """Minimal requests.Response test double."""

    payload: dict[str, Any]
    status_code: int = 200
    text: str = ""

    def json(self) -> dict[str, Any]:
        """Return the configured JSON payload."""
        return self.payload


class FakeSession:
    """Minimal requests.Session test double."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the payload returned by future GET requests."""
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: dict[str, Any],
        timeout: float,
    ) -> FakeResponse:
        """Record request arguments and return a fake response."""
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payload)


@pytest.fixture
def fred_payload() -> dict[str, Any]:
    """Return a representative FRED observations API payload."""
    return {
        "realtime_start": "2020-02-13",
        "realtime_end": "2020-02-13",
        "observations": [
            {
                "realtime_start": "2020-02-13",
                "realtime_end": "2020-03-10",
                "date": "2020-01-01",
                "value": "258.678",
            },
            {
                "realtime_start": "2020-03-11",
                "realtime_end": "9999-12-31",
                "date": "2020-02-01",
                "value": ".",
            },
        ],
    }


def test_fred_client_parses_observations(fred_payload: dict[str, Any]) -> None:
    """Verify FRED observations are normalized into storage columns."""
    session = FakeSession(fred_payload)
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    frame = client.fetch_series_observations(
        series_id="CPIAUCSL",
        observation_start="2020-01-01",
    )

    assert list(frame.columns) == NORMALIZED_COLUMNS
    assert frame.loc[0, "series_id"] == "CPIAUCSL"
    assert frame.loc[0, "date"].isoformat() == "2020-01-01"
    assert frame.loc[0, "value"] == pytest.approx(258.678)
    assert frame.loc[0, "source"] == "fred"
    assert frame.loc[0, "release_date"].isoformat() == "2020-02-13"
    assert pd.notna(frame.loc[0, "fetched_at"])
    assert session.calls[0]["params"]["series_id"] == "CPIAUCSL"
    assert session.calls[0]["params"]["observation_start"] == "2020-01-01"


def test_fred_missing_dot_values_become_nan(fred_payload: dict[str, Any]) -> None:
    """Verify FRED's '.' missing values are converted to NaN."""
    client = FredClient(
        api_key="test-key",
        session=FakeSession(fred_payload),
        backoff_seconds=0,
    )

    frame = client.fetch_series_observations(series_id="CPIAUCSL")

    assert pd.isna(frame.loc[1, "value"])


def test_fred_validate_requires_normalized_columns(
    fred_payload: dict[str, Any],
) -> None:
    """Verify validation rejects frames without normalized columns."""
    client = FredClient(
        api_key="test-key",
        session=FakeSession(fred_payload),
        backoff_seconds=0,
    )

    with pytest.raises(ValueError, match="Missing required"):
        client.validate(pd.DataFrame({"series_id": ["CPIAUCSL"]}))


def test_insert_observations_inserts_into_sqlite(tmp_path) -> None:
    """Verify normalized observations insert into a temporary SQLite database."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    frame = pd.DataFrame(
        {
            "series_id": ["CPIAUCSL", "CPIAUCSL"],
            "date": ["2020-01-01", "2020-02-01"],
            "value": [258.678, None],
            "source": ["fred", "fred"],
            "release_date": ["2020-02-13", "2020-03-11"],
            "fetched_at": ["2026-05-21T21:00:00Z", "2026-05-21T21:00:00Z"],
        },
    )

    row_count = insert_observations(frame, database_url=database_url)
    engine = get_engine(database_url)
    with engine.connect() as connection:
        stored_count = connection.scalar(
            select(func.count()).select_from(macro_observations),
        )
    engine.dispose()

    assert row_count == 2
    assert stored_count == 2


def test_fred_fetch_store_workflow_with_mocked_response(
    fred_payload: dict[str, Any],
    tmp_path,
) -> None:
    """Verify mocked FRED fetches can be validated and stored."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    client = FredClient(
        api_key="test-key",
        session=FakeSession(fred_payload),
        backoff_seconds=0,
    )

    observations = client.fetch_series_observations(series_id="CPIAUCSL")
    inserted = client.store(observations, database_url=database_url)

    assert inserted == {"rows_seen": 2, "inserted": 2, "updated": 0, "skipped": 0}


def test_repeated_fred_store_does_not_duplicate_rows(
    fred_payload: dict[str, Any],
    tmp_path,
) -> None:
    """Verify repeated FRED stores upsert instead of duplicating rows."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    client = FredClient(
        api_key="test-key",
        session=FakeSession(fred_payload),
        backoff_seconds=0,
    )
    observations = client.fetch_series_observations(series_id="CPIAUCSL")

    client.store(observations, database_url=database_url)
    counts = client.store(observations, database_url=database_url)

    engine = get_engine(database_url)
    with engine.connect() as connection:
        stored_count = connection.scalar(
            select(func.count()).select_from(macro_observations),
        )
    engine.dispose()

    assert counts == {"rows_seen": 2, "inserted": 0, "updated": 0, "skipped": 2}
    assert stored_count == 2
