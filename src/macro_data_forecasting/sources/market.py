"""Placeholder market source client."""

import pandas as pd


class MarketClient:
    """Placeholder client for future market data ingestion."""

    def fetch(self) -> pd.DataFrame:
        """Fetch market observations in Stage 2."""
        raise NotImplementedError("TODO: implement market data fetching in Stage 2.")

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate market observations before storage."""
        raise NotImplementedError("TODO: implement market validation in Stage 2.")

    def store(self, data: pd.DataFrame) -> None:
        """Store market observations with release-date metadata."""
        raise NotImplementedError("TODO: implement market storage in Stage 2.")
