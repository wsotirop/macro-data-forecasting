"""Database schema and initialization helpers."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.engine import Engine, make_url

from macro_data_forecasting.config import get_settings

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
    Column("fetched_at", DateTime(timezone=True), nullable=False),
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
    return engine


def _require_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        msg = f"Missing required observation columns: {missing}"
        raise ValueError(msg)


def _records_from_observations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    _require_columns(frame, REQUIRED_OBSERVATION_COLUMNS)
    observations = frame.loc[:, REQUIRED_OBSERVATION_COLUMNS].copy()
    observations["date"] = pd.to_datetime(observations["date"], errors="raise").dt.date
    observations["release_date"] = pd.to_datetime(
        observations["release_date"],
        errors="raise",
    ).dt.date
    observations["fetched_at"] = pd.to_datetime(
        observations["fetched_at"],
        errors="raise",
        utc=True,
    )
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
