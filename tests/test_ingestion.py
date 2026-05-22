"""Tests for ingestion orchestration helpers."""

import json

import pandas as pd
import pytest
from sqlalchemy import select

from macro_data_forecasting.database import (
    get_engine,
    ingestion_runs,
    upsert_observations,
)
from macro_data_forecasting.sources.ingestion import run_ingestion


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["CPIAUCSL"],
            "date": ["2020-01-01"],
            "value": [258.678],
            "source": ["fred"],
            "release_date": ["2020-02-13"],
            "fetched_at": ["2026-05-21T21:00:00Z"],
        },
    )


def test_run_ingestion_records_success(tmp_path) -> None:
    """Verify run_ingestion records succeeded metadata."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"

    result = run_ingestion(
        source="fred",
        series_id="CPIAUCSL",
        fetch_fn=_frame,
        validate_fn=lambda frame: frame,
        store_fn=lambda frame: upsert_observations(frame, database_url=database_url),
        parameters={"series_id": "CPIAUCSL"},
        database_url=database_url,
    )

    engine = get_engine(database_url)
    with engine.connect() as connection:
        row = connection.execute(select(ingestion_runs)).mappings().one()
    engine.dispose()

    assert result["status"] == "succeeded"
    assert result["inserted"] == 1
    assert row["status"] == "succeeded"
    assert json.loads(row["parameters_json"]) == {"series_id": "CPIAUCSL"}


def test_run_ingestion_records_failure(tmp_path) -> None:
    """Verify run_ingestion records failed metadata before re-raising."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"

    def fail_fetch() -> pd.DataFrame:
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        run_ingestion(
            source="fred",
            series_id="CPIAUCSL",
            fetch_fn=fail_fetch,
            validate_fn=lambda frame: frame,
            store_fn=lambda frame: upsert_observations(
                frame,
                database_url=database_url,
            ),
            parameters={"series_id": "CPIAUCSL"},
            database_url=database_url,
        )

    engine = get_engine(database_url)
    with engine.connect() as connection:
        row = connection.execute(select(ingestion_runs)).mappings().one()
    engine.dispose()

    assert row["status"] == "failed"
    assert row["error_message"] == "boom"
