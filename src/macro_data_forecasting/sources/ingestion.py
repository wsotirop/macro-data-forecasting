"""Reusable ingestion run orchestration helpers."""

from collections.abc import Callable
from typing import Any

import pandas as pd

from macro_data_forecasting.database import (
    fail_ingestion_run,
    finish_ingestion_run,
    start_ingestion_run,
)


def run_ingestion(
    source: str,
    series_id: str,
    fetch_fn: Callable[[], pd.DataFrame],
    validate_fn: Callable[[pd.DataFrame], pd.DataFrame],
    store_fn: Callable[[pd.DataFrame], dict[str, int]],
    parameters: dict[str, Any],
    database_url: str | None = None,
) -> dict[str, Any]:
    """Run fetch, validate, store, and ingestion-run bookkeeping."""
    run_id = start_ingestion_run(
        source=source,
        series_id=series_id,
        parameters=parameters,
        database_url=database_url,
    )
    try:
        fetched = fetch_fn()
        validated = validate_fn(fetched)
        counts = store_fn(validated)
        finish_ingestion_run(run_id, counts, database_url=database_url)
    except Exception as exc:
        fail_ingestion_run(run_id, str(exc), database_url=database_url)
        raise

    return {"run_id": run_id, "status": "succeeded", **counts}
