"""Logging configuration helpers."""

import logging

from macro_data_forecasting.config import get_settings


def configure_logging(log_level: str | None = None) -> None:
    """Configure root logging for command-line and batch workflows."""
    level_name = (log_level or get_settings().log_level).upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        msg = f"Invalid log level: {level_name}"
        raise ValueError(msg)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
