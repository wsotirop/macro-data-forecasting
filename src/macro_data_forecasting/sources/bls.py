"""Placeholder BLS source client."""

import pandas as pd


class BlsClient:
    """Placeholder client for future BLS data ingestion."""

    def fetch(self) -> pd.DataFrame:
        """Fetch BLS observations in Stage 2."""
        raise NotImplementedError("TODO: implement BLS API fetching in Stage 2.")

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate BLS observations before storage."""
        raise NotImplementedError("TODO: implement BLS validation in Stage 2.")

    def store(self, data: pd.DataFrame) -> None:
        """Store BLS observations with release-date metadata."""
        raise NotImplementedError("TODO: implement BLS storage in Stage 2.")
