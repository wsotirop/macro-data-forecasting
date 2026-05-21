"""Tests for SQLAlchemy database initialization."""

from sqlalchemy import inspect

from macro_data_forecasting.database import initialize_database


def test_initialize_database_creates_macro_observations(tmp_path) -> None:
    """Verify the SQLite schema contains the macro_observations table."""
    database_path = tmp_path / "macro_data.sqlite"
    engine = initialize_database(f"sqlite:///{database_path.as_posix()}")

    inspector = inspect(engine)

    assert inspector.has_table("macro_observations")
    engine.dispose()
