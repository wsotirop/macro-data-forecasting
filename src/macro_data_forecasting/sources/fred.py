"""Placeholder FRED source client."""

import pandas as pd


class FredClient:
    """Placeholder client for future FRED data ingestion."""

    def fetch(self) -> pd.DataFrame:
        """Fetch FRED observations in Stage 2."""
        raise NotImplementedError("TODO: implement FRED API fetching in Stage 2.")

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate FRED observations before storage."""
        raise NotImplementedError("TODO: implement FRED validation in Stage 2.")

    def store(self, data: pd.DataFrame) -> None:
        """Store FRED observations with release-date metadata."""
        raise NotImplementedError("TODO: implement FRED storage in Stage 2.")
