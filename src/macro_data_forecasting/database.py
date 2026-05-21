"""Database schema and initialization helpers."""

from pathlib import Path

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

metadata = MetaData()

macro_observations = Table(
    "macro_observations",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("series_id", String(128), nullable=False),
    Column("date", Date, nullable=False),
    Column("value", Float, nullable=True),
    Column("source", String(64), nullable=False),
    Column("release_date", Date, nullable=False),
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
