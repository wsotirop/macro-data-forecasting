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
    ingestion_runs,
    initialize_database,
    insert_observations,
    macro_observations,
    start_ingestion_run,
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
