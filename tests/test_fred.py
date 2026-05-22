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
from macro_data_forecasting.sources.fred import (
    NORMALIZED_COLUMNS,
    FredApiError,
    FredClient,
)


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


class SequenceSession:
    """Requests.Session test double returning one response per GET."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        """Store ordered fake responses."""
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: dict[str, Any],
        timeout: float,
    ) -> FakeResponse:
        """Record request arguments and return the next fake response."""
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.responses.pop(0)


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


def test_fred_initial_release_mode_sends_output_type_4(
    fred_payload: dict[str, Any],
) -> None:
    """Verify initial-release mode requests output_type=4 and full realtime window."""
    session = FakeSession(fred_payload)
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    client.fetch_series_observations(
        series_id="UNRATE",
        vintage_mode="initial_release",
    )

    params = session.calls[0]["params"]
    assert params["output_type"] == 4
    assert params["realtime_start"] == "1776-07-04"
    assert params["realtime_end"] == "9999-12-31"


def test_fred_initial_release_preserves_explicit_realtime_window(
    fred_payload: dict[str, Any],
) -> None:
    """Verify initial-release mode preserves user-provided realtime bounds."""
    session = FakeSession(fred_payload)
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    client.fetch_series_observations(
        series_id="UNRATE",
        realtime_start="2018-01-01",
        realtime_end="2020-01-01",
        vintage_mode="initial_release",
    )

    params = session.calls[0]["params"]
    assert params["output_type"] == 4
    assert params["realtime_start"] == "2018-01-01"
    assert params["realtime_end"] == "2020-01-01"


def test_fred_initial_release_uses_observation_realtime_start(
    fred_payload: dict[str, Any],
) -> None:
    """Verify initial-release mode maps release_date from row realtime_start."""
    client = FredClient(
        api_key="test-key",
        session=FakeSession(fred_payload),
        backoff_seconds=0,
    )

    frame = client.fetch_series_observations(
        series_id="UNRATE",
        vintage_mode="initial_release",
    )

    assert frame.loc[0, "release_date"].isoformat() == "2020-02-13"
    assert frame.loc[1, "release_date"].isoformat() == "2020-03-11"


def test_fred_initial_release_raises_without_realtime_start() -> None:
    """Verify initial-release mode rejects responses without vintage metadata."""
    payload = {
        "observations": [
            {"date": "2020-01-01", "value": "3.5"},
        ],
    }
    client = FredClient(
        api_key="test-key",
        session=FakeSession(payload),
        backoff_seconds=0,
    )

    with pytest.raises(FredApiError, match="initial_release.*realtime_start"):
        client.fetch_series_observations(
            series_id="UNRATE",
            vintage_mode="initial_release",
        )


def test_fred_current_mode_preserves_existing_behavior(
    fred_payload: dict[str, Any],
) -> None:
    """Verify current mode keeps default request behavior and release mapping."""
    session = FakeSession(fred_payload)
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    frame = client.fetch_series_observations(series_id="UNRATE")

    assert "output_type" not in session.calls[0]["params"]
    assert frame.loc[0, "release_date"].isoformat() == "2020-02-13"


def test_fred_realtime_period_mode_sends_output_type_1(
    fred_payload: dict[str, Any],
) -> None:
    """Verify realtime-period mode requests FRED output_type=1."""
    session = FakeSession(fred_payload)
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    client.fetch_series_observations(
        series_id="UNRATE",
        realtime_start="2018-01-01",
        realtime_end="2020-01-01",
        vintage_mode="realtime_period",
    )

    params = session.calls[0]["params"]
    assert params["output_type"] == 1
    assert params["realtime_start"] == "2018-01-01"
    assert params["realtime_end"] == "2020-01-01"


def test_fred_chunked_initial_release_calls_api_multiple_times() -> None:
    """Verify chunked initial-release requests split the real-time window."""
    session = SequenceSession(
        [
            FakeResponse(
                {
                    "observations": [
                        {
                            "realtime_start": "2020-02-01",
                            "realtime_end": "2020-02-01",
                            "date": "2020-01-01",
                            "value": "3.5",
                        },
                    ],
                },
            ),
            FakeResponse(
                {
                    "observations": [
                        {
                            "realtime_start": "2022-02-01",
                            "realtime_end": "2022-02-01",
                            "date": "2022-01-01",
                            "value": "4.0",
                        },
                    ],
                },
            ),
        ],
    )
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    frame = client.fetch_series_observations(
        series_id="UNRATE",
        observation_start="2020-01-01",
        realtime_start="2020-01-01",
        realtime_end="2023-12-31",
        vintage_mode="initial_release",
        chunk_realtime=True,
        realtime_chunk_years=2,
    )

    assert len(session.calls) == 2
    assert len(frame) == 2
    assert session.calls[0]["params"]["realtime_start"] == "2020-01-01"
    assert session.calls[0]["params"]["realtime_end"] == "2021-12-31"
    assert session.calls[1]["params"]["realtime_start"] == "2022-01-01"
    assert session.calls[1]["params"]["realtime_end"] == "2023-12-31"


def test_fred_chunked_initial_release_default_end_uses_max_final_chunk() -> None:
    """Verify default chunked initial-release mode ends final chunk at max date."""
    session = SequenceSession(
        [
            FakeResponse(
                {
                    "observations": [
                        {
                            "realtime_start": "2025-02-01",
                            "realtime_end": "2025-02-01",
                            "date": "2025-01-01",
                            "value": "4.1",
                        },
                    ],
                },
            ),
        ],
    )
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    client.fetch_series_observations(
        series_id="DGS2",
        observation_start="2025-01-01",
        vintage_mode="initial_release",
        chunk_realtime=True,
        realtime_chunk_years=100,
    )

    params = session.calls[0]["params"]
    assert params["realtime_start"] == "2025-01-01"
    assert params["realtime_end"] == "9999-12-31"


def test_fred_chunked_initial_release_does_not_use_local_date_as_final_end() -> None:
    """Verify the final default chunk does not end on the local system date."""
    session = SequenceSession([FakeResponse({"observations": []})])
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    client.fetch_series_observations(
        series_id="DGS10",
        observation_start="2025-01-01",
        vintage_mode="initial_release",
        chunk_realtime=True,
        realtime_chunk_years=100,
    )

    assert session.calls[0]["params"]["realtime_end"] == "9999-12-31"


def test_fred_chunked_initial_release_drops_exact_duplicates() -> None:
    """Verify exact duplicate normalized observations are removed after chunks."""
    observation = {
        "realtime_start": "2020-02-01",
        "realtime_end": "2020-02-01",
        "date": "2020-01-01",
        "value": "3.5",
    }
    session = SequenceSession(
        [
            FakeResponse({"observations": [observation]}),
            FakeResponse({"observations": [observation]}),
        ],
    )
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    frame = client.fetch_series_observations(
        series_id="UNRATE",
        observation_start="2020-01-01",
        realtime_start="2020-01-01",
        realtime_end="2021-12-31",
        vintage_mode="initial_release",
        chunk_realtime=True,
        realtime_chunk_years=1,
    )

    assert len(frame) == 1


def test_fred_chunked_initial_release_passes_observation_start_to_every_chunk() -> None:
    """Verify observation_start is passed through on each real-time chunk."""
    session = SequenceSession(
        [
            FakeResponse({"observations": []}),
            FakeResponse({"observations": []}),
        ],
    )
    client = FredClient(api_key="test-key", session=session, backoff_seconds=0)

    client.fetch_series_observations(
        series_id="UNRATE",
        observation_start="2020-01-01",
        realtime_start="2020-01-01",
        realtime_end="2021-12-31",
        vintage_mode="initial_release",
        chunk_realtime=True,
        realtime_chunk_years=1,
    )

    assert {call["params"]["observation_start"] for call in session.calls} == {
        "2020-01-01",
    }


def test_fred_chunked_initial_release_failed_chunk_raises_with_window() -> None:
    """Verify failed chunks are surfaced with the chunk window."""
    session = SequenceSession(
        [
            FakeResponse({"observations": []}),
            FakeResponse({}, status_code=400, text="Bad chunk"),
        ],
    )
    client = FredClient(
        api_key="test-key",
        session=session,
        backoff_seconds=0,
        max_retries=1,
    )

    with pytest.raises(FredApiError, match="2021-01-01 to 2021-12-31"):
        client.fetch_series_observations(
            series_id="UNRATE",
            observation_start="2020-01-01",
            realtime_start="2020-01-01",
            realtime_end="2021-12-31",
            vintage_mode="initial_release",
            chunk_realtime=True,
            realtime_chunk_years=1,
        )


def test_fred_chunked_initial_release_after_today_error_has_guidance() -> None:
    """Verify server-date realtime_end errors include operator guidance."""
    session = SequenceSession(
        [
            FakeResponse(
                {},
                status_code=400,
                text=(
                    "Variable realtime_end can not be after today's date "
                    "(2026-05-21)."
                ),
            ),
        ],
    )
    client = FredClient(
        api_key="test-key",
        session=session,
        backoff_seconds=0,
        max_retries=1,
    )

    with pytest.raises(FredApiError, match="Pass a valid --realtime-end"):
        client.fetch_series_observations(
            series_id="DGS2",
            observation_start="2025-01-01",
            realtime_start="2025-01-01",
            realtime_end="2026-05-22",
            vintage_mode="initial_release",
            chunk_realtime=True,
            realtime_chunk_years=5,
        )


def test_fred_chunked_realtime_requires_initial_release(
    fred_payload: dict[str, Any],
) -> None:
    """Verify chunking is rejected outside initial-release mode."""
    client = FredClient(
        api_key="test-key",
        session=FakeSession(fred_payload),
        backoff_seconds=0,
    )

    with pytest.raises(FredApiError, match="chunk_realtime"):
        client.fetch_series_observations(series_id="UNRATE", chunk_realtime=True)


def test_fred_invalid_output_type_raises(fred_payload: dict[str, Any]) -> None:
    """Verify unsupported FRED output_type values are rejected."""
    client = FredClient(
        api_key="test-key",
        session=FakeSession(fred_payload),
        backoff_seconds=0,
    )

    with pytest.raises(FredApiError, match="Unsupported FRED output_type"):
        client.fetch_series_observations(series_id="UNRATE", output_type=5)


def test_fred_invalid_vintage_mode_raises(fred_payload: dict[str, Any]) -> None:
    """Verify unsupported FRED vintage modes are rejected."""
    client = FredClient(
        api_key="test-key",
        session=FakeSession(fred_payload),
        backoff_seconds=0,
    )

    with pytest.raises(FredApiError, match="Unsupported FRED vintage_mode"):
        client.fetch_series_observations(series_id="UNRATE", vintage_mode="bad-mode")


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
