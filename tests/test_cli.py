"""Tests for command-line ingestion workflows."""

import pandas as pd
import pytest
from sqlalchemy import func, select

import macro_data_forecasting.cli as cli
from macro_data_forecasting.database import (
    get_engine,
    ingestion_runs,
    macro_observations,
    upsert_observations,
)
from macro_data_forecasting.sources.bls_release_calendar import CalendarCoverageError


class FakeFredClient:
    """Fake FRED client for CLI tests."""

    def fetch_series_observations(self, **kwargs) -> pd.DataFrame:
        """Return one normalized FRED observation."""
        return pd.DataFrame(
            {
                "series_id": [kwargs["series_id"]],
                "date": ["2020-01-01"],
                "value": [258.678],
                "source": ["fred"],
                "release_date": ["2020-02-13"],
                "fetched_at": ["2026-05-21T21:00:00Z"],
            },
        )

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return already-normalized test observations."""
        return data

    def store(
        self,
        data: pd.DataFrame,
        database_url: str | None = None,
    ) -> dict[str, int]:
        """Store test observations through the real upsert helper."""
        return upsert_observations(data, database_url=database_url)


class FakeBlsClient:
    """Fake BLS client for CLI tests."""

    def fetch_series_observations(self, **kwargs) -> pd.DataFrame:
        """Return one BLS observation outside the sample calendar coverage."""
        return pd.DataFrame(
            {
                "series_id": [kwargs["series_id"]],
                "date": ["2026-05-01"],
                "value": [320.0],
                "source": ["bls"],
                "release_date": [pd.NaT],
                "fetched_at": ["2026-05-21T21:00:00Z"],
            },
        )

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return already-normalized test observations."""
        return data

    def store(
        self,
        data: pd.DataFrame,
        database_url: str | None = None,
    ) -> dict[str, int]:
        """Fail if strict-calendar preflight does not stop storage."""
        raise AssertionError("store should not be called")


def test_cli_fetch_fred_records_ingestion_run(monkeypatch, tmp_path, capsys) -> None:
    """Verify fetch-fred uses upsert storage and records a succeeded run."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    monkeypatch.setattr(cli, "FredClient", FakeFredClient)

    result = cli.main(
        [
            "fetch-fred",
            "--series-id",
            "CPIAUCSL",
            "--start",
            "2020-01-01",
            "--database-url",
            database_url,
        ],
    )

    output = capsys.readouterr().out
    engine = get_engine(database_url)
    with engine.connect() as connection:
        run = connection.execute(select(ingestion_runs)).mappings().one()
        stored_count = connection.scalar(
            select(func.count()).select_from(macro_observations),
        )
    engine.dispose()

    assert result == 0
    assert "Rows seen: 1" in output
    assert "Inserted: 1" in output
    assert run["status"] == "succeeded"
    assert stored_count == 1


def test_cli_fetch_bls_strict_calendar_fails_before_storage(
    monkeypatch,
    tmp_path,
) -> None:
    """Verify strict BLS calendar coverage failures are recorded before storage."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    monkeypatch.setattr(cli, "BlsClient", FakeBlsClient)

    with pytest.raises(CalendarCoverageError, match="2026-05"):
        cli.main(
            [
                "fetch-bls",
                "--series-id",
                "CUSR0000SA0",
                "--start-year",
                "2026",
                "--end-year",
                "2026",
                "--release-calendar",
                "data/reference/cpi_release_calendar_sample.csv",
                "--strict-calendar",
                "--database-url",
                database_url,
            ],
        )

    engine = get_engine(database_url)
    with engine.connect() as connection:
        run = connection.execute(select(ingestion_runs)).mappings().one()
        stored_count = connection.scalar(
            select(func.count()).select_from(macro_observations),
        )
    engine.dispose()

    assert run["status"] == "failed"
    assert "2026-05" in run["error_message"]
    assert stored_count == 0
