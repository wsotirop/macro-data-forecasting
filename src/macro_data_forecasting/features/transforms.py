"""Feature transformation placeholders."""

import pandas as pd


def lag_by_release_date(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Lag features according to release-date availability."""
    # Transformations must preserve point-in-time availability and avoid
    # leaking data that was not released at the forecast timestamp.
    raise NotImplementedError(
        "TODO: implement release-date-aware transforms in Stage 3."
    )
