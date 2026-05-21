"""Common interface for source clients."""

from typing import Any, Protocol

import pandas as pd


class SourceClient(Protocol):
    """Protocol for staged macro data source clients."""

    def fetch(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Fetch observations from the source."""
        ...

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate fetched observations before storage."""
        ...

    def store(self, data: pd.DataFrame) -> None:
        """Store validated observations."""
        ...
