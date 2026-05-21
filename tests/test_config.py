"""Tests for configuration loading."""

from pathlib import Path

from macro_data_forecasting.config import get_settings


def test_get_settings_returns_default_dirs(monkeypatch) -> None:
    """Verify default directory settings are relative scaffold paths."""
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("REPORTS_DIR", raising=False)

    settings = get_settings()

    assert settings.data_dir == Path("data")
    assert settings.reports_dir == Path("reports")
