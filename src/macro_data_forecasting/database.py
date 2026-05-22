"""Database schema and initialization helpers."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    case,
    create_engine,
    func,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine, make_url

from macro_data_forecasting.config import get_settings

INGESTION_RUN_COLUMNS = (
    "id",
    "source",
    "series_id",
    "status",
    "started_at",
    "finished_at",
    "rows_seen",
    "inserted",
    "updated",
    "skipped",
    "error_message",
    "parameters_json",
)
MISSING_RELEASE_DATE_KEY = "__MISSING__"
OBSERVATION_COVERAGE_COLUMNS = (
    "source",
    "series_id",
    "row_count",
    "min_date",
    "max_date",
    "min_release_date",
    "max_release_date",
    "missing_release_date_count",
)
REQUIRED_OBSERVATION_COLUMNS = (
    "series_id",
    "date",
    "value",
    "source",
    "release_date",
    "fetched_at",
)

metadata = MetaData()

macro_observations = Table(
    "macro_observations",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("series_id", String(128), nullable=False),
    Column("date", Date, nullable=False),
    Column("value", Float, nullable=True),
    Column("source", String(64), nullable=False),
    Column("release_date", Date, nullable=True),
    # SQLite allows duplicate NULLs in unique indexes. Store a deterministic
    # key alongside the nullable release_date so missing-release-date rows are
    # still idempotent instead of accumulating duplicates on repeated ingestion.
    Column("release_date_key", String(32), nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
)

Index(
    "uq_macro_observations_point_in_time",
    macro_observations.c.series_id,
    macro_observations.c.source,
    macro_observations.c.date,
    macro_observations.c.release_date_key,
    unique=True,
)

ingestion_runs = Table(
    "ingestion_runs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("source", String(64), nullable=False),
    Column("series_id", String(128), nullable=False),
    Column("status", String(32), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("rows_seen", Integer, nullable=False, default=0),
    Column("inserted", Integer, nullable=False, default=0),
    Column("updated", Integer, nullable=False, default=0),
    Column("skipped", Integer, nullable=False, default=0),
    Column("error_message", Text, nullable=True),
    Column("parameters_json", Text, nullable=True),
)


def get_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for the configured database URL."""
    resolved_url = database_url or get_settings().database_url
    url = make_url(resolved_url)
    if url.drivername == "sqlite" and url.database not in (None, "", ":memory:"):
        database_path = Path(url.database)
        database_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(resolved_url)


def initialize_database(database_url: str | None = None) -> Engine:
    """Create database tables and return the SQLAlchemy engine."""
    engine = get_engine(database_url)
    metadata.create_all(engine)
    _ensure_sqlite_schema(engine)
    return engine


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _ensure_sqlite_schema(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "macro_observations" not in table_names:
        return

    column_names = {
        column["name"] for column in inspector.get_columns("macro_observations")
    }
    with engine.begin() as connection:
        if "release_date_key" not in column_names:
            connection.execute(
                text(
                    "ALTER TABLE macro_observations "
                    f"ADD COLUMN release_date_key VARCHAR(32) NOT NULL "
                    f"DEFAULT '{MISSING_RELEASE_DATE_KEY}'",
                ),
            )
            connection.execute(
                text(
                    "UPDATE macro_observations "
                    "SET release_date_key = COALESCE(CAST(release_date AS TEXT), "
                    f"'{MISSING_RELEASE_DATE_KEY}')",
                ),
            )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_macro_observations_point_in_time "
                "ON macro_observations "
                "(series_id, source, date, release_date_key)",
            ),
        )


def _require_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        msg = f"Missing required observation columns: {missing}"
        raise ValueError(msg)


def _release_date_keys(release_dates: pd.Series) -> pd.Series:
    return release_dates.map(
        lambda value: MISSING_RELEASE_DATE_KEY
        if pd.isna(value)
        else value.isoformat(),
    )


def _records_from_observations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    _require_columns(frame, REQUIRED_OBSERVATION_COLUMNS)
    observations = frame.loc[:, REQUIRED_OBSERVATION_COLUMNS].copy()
    observations["date"] = pd.to_datetime(observations["date"], errors="raise").dt.date
    release_dates = pd.to_datetime(
        observations["release_date"],
        errors="coerce",
    )
    invalid_release_dates = observations["release_date"].notna() & release_dates.isna()
    if invalid_release_dates.any():
        msg = "release_date contains invalid date values."
        raise ValueError(msg)
    observations["release_date"] = release_dates.dt.date
    observations["release_date_key"] = _release_date_keys(observations["release_date"])
    observations["fetched_at"] = pd.to_datetime(
        observations["fetched_at"],
        errors="raise",
        utc=True,
    ).dt.tz_convert(None)
    observations["value"] = pd.to_numeric(observations["value"], errors="raise")
    observations = observations.astype(object).where(pd.notna(observations), None)
    return observations.to_dict(orient="records")


def insert_observations(df: pd.DataFrame, database_url: str | None = None) -> int:
    """Insert normalized macro observations and return the inserted row count."""
    records = _records_from_observations(df)
    if not records:
        return 0

    engine = initialize_database(database_url)
    with engine.begin() as connection:
        result = connection.execute(macro_observations.insert(), records)
    engine.dispose()
    return int(result.rowcount)


def _observation_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["series_id"],
        record["source"],
        record["date"],
        record["release_date_key"],
    )


def _values_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return left == right


def upsert_observations(
    df: pd.DataFrame,
    database_url: str | None = None,
) -> dict[str, int]:
    """Insert or update normalized observations without creating duplicates."""
    records = _records_from_observations(df)
    counts = {"rows_seen": len(records), "inserted": 0, "updated": 0, "skipped": 0}
    if not records:
        return counts

    engine = initialize_database(database_url)
    with engine.begin() as connection:
        for record in records:
            existing = connection.execute(
                select(macro_observations).where(
                    macro_observations.c.series_id == record["series_id"],
                    macro_observations.c.source == record["source"],
                    macro_observations.c.date == record["date"],
                    macro_observations.c.release_date_key
                    == record["release_date_key"],
                ),
            ).mappings().one_or_none()

            if existing is None:
                connection.execute(macro_observations.insert().values(record))
                counts["inserted"] += 1
                continue

            value_changed = not _values_equal(existing["value"], record["value"])
            fetched_at_changed = not _values_equal(
                existing["fetched_at"],
                record["fetched_at"],
            )
            if value_changed or fetched_at_changed:
                connection.execute(
                    update(macro_observations)
                    .where(macro_observations.c.id == existing["id"])
                    .values(
                        value=record["value"],
                        fetched_at=record["fetched_at"],
                    ),
                )
                counts["updated"] += 1
            else:
                counts["skipped"] += 1

    engine.dispose()
    return counts


def start_ingestion_run(
    source: str,
    series_id: str,
    parameters: dict[str, Any] | None = None,
    database_url: str | None = None,
) -> int:
    """Create an ingestion run row and return its run ID."""
    engine = initialize_database(database_url)
    with engine.begin() as connection:
        result = connection.execute(
            ingestion_runs.insert().values(
                source=source,
                series_id=series_id,
                status="started",
                started_at=_utc_now(),
                finished_at=None,
                rows_seen=0,
                inserted=0,
                updated=0,
                skipped=0,
                error_message=None,
                parameters_json=json.dumps(
                    parameters or {},
                    sort_keys=True,
                    default=str,
                ),
            ),
        )
        run_id = int(result.inserted_primary_key[0])
    engine.dispose()
    return run_id


def finish_ingestion_run(
    run_id: int,
    counts: dict[str, int],
    database_url: str | None = None,
) -> None:
    """Mark an ingestion run as succeeded with final row counts."""
    engine = initialize_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(ingestion_runs)
            .where(ingestion_runs.c.id == run_id)
            .values(
                status="succeeded",
                finished_at=_utc_now(),
                rows_seen=counts.get("rows_seen", 0),
                inserted=counts.get("inserted", 0),
                updated=counts.get("updated", 0),
                skipped=counts.get("skipped", 0),
                error_message=None,
            ),
        )
    engine.dispose()


def fail_ingestion_run(
    run_id: int,
    error_message: str,
    database_url: str | None = None,
) -> None:
    """Mark an ingestion run as failed with an error message."""
    engine = initialize_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            update(ingestion_runs)
            .where(ingestion_runs.c.id == run_id)
            .values(
                status="failed",
                finished_at=_utc_now(),
                error_message=error_message,
            ),
        )
    engine.dispose()


def list_ingestion_runs(
    database_url: str | None = None,
    limit: int = 10,
    source: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    """Return recent ingestion runs sorted newest first."""
    if limit < 1:
        msg = "limit must be at least 1."
        raise ValueError(msg)

    statement = select(ingestion_runs)
    if source is not None:
        statement = statement.where(ingestion_runs.c.source == source)
    if status is not None:
        statement = statement.where(ingestion_runs.c.status == status)
    statement = statement.order_by(
        ingestion_runs.c.started_at.desc(),
        ingestion_runs.c.id.desc(),
    ).limit(limit)

    engine = initialize_database(database_url)
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    engine.dispose()
    return pd.DataFrame(rows, columns=INGESTION_RUN_COLUMNS)


def get_ingestion_run(
    run_id: int,
    database_url: str | None = None,
) -> dict[str, Any] | None:
    """Return one ingestion run by ID, or None when it is missing."""
    engine = initialize_database(database_url)
    with engine.connect() as connection:
        row = connection.execute(
            select(ingestion_runs).where(ingestion_runs.c.id == run_id),
        ).mappings().one_or_none()
    engine.dispose()
    return dict(row) if row is not None else None


def summarize_observation_coverage(
    database_url: str | None = None,
    source: str | None = None,
    series_id: str | None = None,
) -> pd.DataFrame:
    """Return source and series observation coverage summary rows."""
    missing_release_dates = func.sum(
        case((macro_observations.c.release_date.is_(None), 1), else_=0),
    ).label("missing_release_date_count")
    statement = select(
        macro_observations.c.source,
        macro_observations.c.series_id,
        func.count().label("row_count"),
        func.min(macro_observations.c.date).label("min_date"),
        func.max(macro_observations.c.date).label("max_date"),
        func.min(macro_observations.c.release_date).label("min_release_date"),
        func.max(macro_observations.c.release_date).label("max_release_date"),
        missing_release_dates,
    )
    if source is not None:
        statement = statement.where(macro_observations.c.source == source)
    if series_id is not None:
        statement = statement.where(macro_observations.c.series_id == series_id)
    statement = statement.group_by(
        macro_observations.c.source,
        macro_observations.c.series_id,
    ).order_by(
        macro_observations.c.source,
        macro_observations.c.series_id,
    )

    engine = initialize_database(database_url)
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    engine.dispose()
    return pd.DataFrame(rows, columns=OBSERVATION_COVERAGE_COLUMNS)
