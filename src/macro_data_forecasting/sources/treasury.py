"""Placeholder Treasury source client."""

import pandas as pd


class TreasuryClient:
    """Placeholder client for future Treasury data ingestion."""

    def fetch(self) -> pd.DataFrame:
        """Fetch Treasury observations in Stage 2."""
        raise NotImplementedError("TODO: implement Treasury fetching in Stage 2.")

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate Treasury observations before storage."""
        raise NotImplementedError("TODO: implement Treasury validation in Stage 2.")

    def store(self, data: pd.DataFrame) -> None:
        """Store Treasury observations with release-date metadata."""
        raise NotImplementedError("TODO: implement Treasury storage in Stage 2.")
