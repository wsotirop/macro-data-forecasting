"""Tests for SQLAlchemy database initialization."""

import json

import pandas as pd
import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError

from macro_data_forecasting.database import (
    fail_ingestion_run,
    finish_ingestion_run,
    get_engine,
    get_ingestion_run,
    ingestion_runs,
    initialize_database,
    insert_observations,
    list_ingestion_runs,
    load_observations,
    macro_observations,
    start_ingestion_run,
    summarize_observation_coverage,
    upsert_observations,
)


def test_initialize_database_creates_macro_observations(tmp_path) -> None:
    """Verify the SQLite schema contains ingestion tables."""
    database_path = tmp_path / "macro_data.sqlite"
    engine = initialize_database(f"sqlite:///{database_path.as_posix()}")

    inspector = inspect(engine)

    assert inspector.has_table("macro_observations")
    assert inspector.has_table("ingestion_runs")
    assert "release_date_key" in {
        column["name"] for column in inspector.get_columns("macro_observations")
    }
    engine.dispose()


def _observation_frame(value: float = 258.678) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["CPIAUCSL"],
            "date": ["2020-01-01"],
            "value": [value],
            "source": ["fred"],
            "release_date": ["2020-02-13"],
            "fetched_at": ["2026-05-21T21:00:00Z"],
        },
    )


def test_upsert_observations_inserts_new_rows(tmp_path) -> None:
    """Verify upsert inserts unseen observation keys."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"

    counts = upsert_observations(_observation_frame(), database_url=database_url)

    assert counts == {"rows_seen": 1, "inserted": 1, "updated": 0, "skipped": 0}


def test_upsert_observations_does_not_duplicate_repeated_rows(tmp_path) -> None:
    """Verify repeated upserts skip identical observation rows."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    frame = _observation_frame()

    upsert_observations(frame, database_url=database_url)
    counts = upsert_observations(frame, database_url=database_url)

    engine = get_engine(database_url)
    with engine.connect() as connection:
        stored_count = connection.scalar(
            select(func.count()).select_from(macro_observations),
        )
    engine.dispose()

    assert counts == {"rows_seen": 1, "inserted": 0, "updated": 0, "skipped": 1}
    assert stored_count == 1


def test_macro_observations_unique_constraint_rejects_duplicate_keys(
    tmp_path,
) -> None:
    """Verify the table rejects duplicate point-in-time observation keys."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    frame = _observation_frame()

    insert_observations(frame, database_url=database_url)

    with pytest.raises(IntegrityError):
        insert_observations(frame, database_url=database_url)


def test_upsert_observations_updates_existing_row_when_value_changes(tmp_path) -> None:
    """Verify upsert updates an existing key when the value changes."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"

    upsert_observations(_observation_frame(258.678), database_url=database_url)
    counts = upsert_observations(_observation_frame(259.123), database_url=database_url)

    engine = get_engine(database_url)
    with engine.connect() as connection:
        value = connection.scalar(select(macro_observations.c.value))
    engine.dispose()

    assert counts == {"rows_seen": 1, "inserted": 0, "updated": 1, "skipped": 0}
    assert value == 259.123


def test_upsert_observations_handles_missing_release_date_idempotently(
    tmp_path,
) -> None:
    """Verify missing release dates use a deterministic uniqueness key."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    frame = pd.DataFrame(
        {
            "series_id": ["CUSR0000SA0"],
            "date": ["2026-04-01"],
            "value": [319.086],
            "source": ["bls"],
            "release_date": [pd.NaT],
            "fetched_at": ["2026-05-21T21:00:00Z"],
        },
    )

    upsert_observations(frame, database_url=database_url)
    counts = upsert_observations(frame, database_url=database_url)

    engine = get_engine(database_url)
    with engine.connect() as connection:
        stored_count = connection.scalar(
            select(func.count()).select_from(macro_observations),
        )
        key = connection.scalar(select(macro_observations.c.release_date_key))
    engine.dispose()

    assert counts == {"rows_seen": 1, "inserted": 0, "updated": 0, "skipped": 1}
    assert stored_count == 1
    assert key == "__MISSING__"


def test_ingestion_run_success_records_status_and_parameters(tmp_path) -> None:
    """Verify successful ingestion run metadata is recorded."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    run_id = start_ingestion_run(
        source="fred",
        series_id="CPIAUCSL",
        parameters={"observation_start": "2020-01-01"},
        database_url=database_url,
    )

    finish_ingestion_run(
        run_id,
        {"rows_seen": 1, "inserted": 1, "updated": 0, "skipped": 0},
        database_url=database_url,
    )

    engine = get_engine(database_url)
    with engine.connect() as connection:
        row = connection.execute(select(ingestion_runs)).mappings().one()
    engine.dispose()

    assert row["status"] == "succeeded"
    assert row["rows_seen"] == 1
    assert json.loads(row["parameters_json"]) == {"observation_start": "2020-01-01"}


def test_ingestion_run_failure_records_error(tmp_path) -> None:
    """Verify failed ingestion run metadata captures the error message."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    run_id = start_ingestion_run(
        source="bls",
        series_id="CUSR0000SA0",
        parameters={"start_year": 2026},
        database_url=database_url,
    )

    fail_ingestion_run(run_id, "calendar coverage missing", database_url=database_url)

    engine = get_engine(database_url)
    with engine.connect() as connection:
        row = connection.execute(select(ingestion_runs)).mappings().one()
    engine.dispose()

    assert row["status"] == "failed"
    assert row["error_message"] == "calendar coverage missing"


def test_list_ingestion_runs_returns_recent_runs(tmp_path) -> None:
    """Verify ingestion runs are listed newest first."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    first = start_ingestion_run("fred", "CPIAUCSL", {}, database_url=database_url)
    second = start_ingestion_run("bls", "CUSR0000SA0", {}, database_url=database_url)

    runs = list_ingestion_runs(database_url=database_url, limit=10)

    assert list(runs["id"]) == [second, first]


def test_list_ingestion_runs_filters_by_source_and_status(tmp_path) -> None:
    """Verify ingestion run listing supports source and status filters."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    fred_run = start_ingestion_run("fred", "CPIAUCSL", {}, database_url=database_url)
    finish_ingestion_run(
        fred_run,
        {"rows_seen": 1, "inserted": 1, "updated": 0, "skipped": 0},
        database_url=database_url,
    )
    bls_run = start_ingestion_run("bls", "CUSR0000SA0", {}, database_url=database_url)
    fail_ingestion_run(bls_run, "missing calendar", database_url=database_url)

    runs = list_ingestion_runs(
        database_url=database_url,
        source="bls",
        status="failed",
    )

    assert list(runs["id"]) == [bls_run]
    assert list(runs["status"]) == ["failed"]


def test_get_ingestion_run_returns_details(tmp_path) -> None:
    """Verify one ingestion run can be loaded by ID."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    run_id = start_ingestion_run(
        "fred",
        "CPIAUCSL",
        {"observation_start": "2020-01-01"},
        database_url=database_url,
    )

    run = get_ingestion_run(run_id, database_url=database_url)

    assert run is not None
    assert run["id"] == run_id
    assert run["source"] == "fred"
    assert json.loads(run["parameters_json"]) == {"observation_start": "2020-01-01"}


def test_get_ingestion_run_returns_none_when_missing(tmp_path) -> None:
    """Verify missing ingestion run IDs return None."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"

    assert get_ingestion_run(999, database_url=database_url) is None


def test_summarize_observation_coverage_returns_counts_and_dates(tmp_path) -> None:
    """Verify observation coverage summarizes source and series ranges."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    frame = pd.DataFrame(
        {
            "series_id": ["CPIAUCSL", "CPIAUCSL"],
            "date": ["2020-01-01", "2020-02-01"],
            "value": [258.678, 259.007],
            "source": ["fred", "fred"],
            "release_date": ["2020-02-13", "2020-03-11"],
            "fetched_at": ["2026-05-21T21:00:00Z", "2026-05-21T21:00:00Z"],
        },
    )
    upsert_observations(frame, database_url=database_url)

    coverage = summarize_observation_coverage(database_url=database_url)
    row = coverage.iloc[0]

    assert row["source"] == "fred"
    assert row["series_id"] == "CPIAUCSL"
    assert row["row_count"] == 2
    assert row["min_date"].isoformat() == "2020-01-01"
    assert row["max_date"].isoformat() == "2020-02-01"
    assert row["min_release_date"].isoformat() == "2020-02-13"
    assert row["max_release_date"].isoformat() == "2020-03-11"
    assert row["missing_release_date_count"] == 0


def test_summarize_observation_coverage_counts_missing_release_dates(
    tmp_path,
) -> None:
    """Verify coverage counts observations without release dates."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    frame = pd.DataFrame(
        {
            "series_id": ["CUSR0000SA0", "CUSR0000SA0"],
            "date": ["2026-04-01", "2026-05-01"],
            "value": [319.086, 320.0],
            "source": ["bls", "bls"],
            "release_date": [pd.NaT, pd.NaT],
            "fetched_at": ["2026-05-21T21:00:00Z", "2026-05-21T21:00:00Z"],
        },
    )
    upsert_observations(frame, database_url=database_url)

    coverage = summarize_observation_coverage(
        database_url=database_url,
        source="bls",
        series_id="CUSR0000SA0",
    )
    row = coverage.iloc[0]

    assert row["row_count"] == 2
    assert row["missing_release_date_count"] == 2
    assert pd.isna(row["min_release_date"])
    assert pd.isna(row["max_release_date"])


def test_load_observations_returns_filtered_rows(tmp_path) -> None:
    """Verify stored observations can be loaded by source and series."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    frame = pd.DataFrame(
        {
            "series_id": ["CPIAUCSL", "CUSR0000SA0"],
            "date": ["2020-01-01", "2026-04-01"],
            "value": [258.678, 319.086],
            "source": ["fred", "bls"],
            "release_date": ["2020-02-13", "2026-05-12"],
            "fetched_at": ["2026-05-21T21:00:00Z", "2026-05-21T21:00:00Z"],
        },
    )
    upsert_observations(frame, database_url=database_url)

    observations = load_observations(
        database_url=database_url,
        source="bls",
        series_id="CUSR0000SA0",
    )

    assert len(observations) == 1
    assert observations.loc[0, "series_id"] == "CUSR0000SA0"
    assert observations.loc[0, "source"] == "bls"
    assert observations.loc[0, "date"].isoformat() == "2026-04-01"
