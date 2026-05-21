"""Runtime configuration for macro data forecasting."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    fred_api_key: str | None = None
    bls_api_key: str | None = None
    database_url: str = "sqlite:///data/macro_data.sqlite"
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Return application settings loaded from `.env` and environment variables."""
    load_dotenv()
    return Settings(
        fred_api_key=os.getenv("FRED_API_KEY") or None,
        bls_api_key=os.getenv("BLS_API_KEY") or None,
        database_url=os.getenv("DATABASE_URL", "sqlite:///data/macro_data.sqlite"),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        reports_dir=Path(os.getenv("REPORTS_DIR", "reports")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
