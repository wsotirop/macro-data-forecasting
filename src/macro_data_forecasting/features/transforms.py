"""Feature transformation placeholders."""

import pandas as pd


def lag_by_release_date(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Lag features according to release-date availability."""
    # Transformations must preserve point-in-time availability and avoid
    # leaking data that was not released at the forecast timestamp.
    raise NotImplementedError(
        "TODO: implement release-date-aware transforms in Stage 3."
    )


def percent_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Return percent change over the requested number of periods."""
    return series.astype(float).pct_change(periods=periods) * 100


def difference(series: pd.Series, periods: int = 1) -> pd.Series:
    """Return arithmetic difference over the requested number of periods."""
    return series.astype(float).diff(periods=periods)


def rolling_z_score(series: pd.Series, window: int) -> pd.Series:
    """Return rolling z-scores using a full rolling window."""
    if window < 2:
        msg = "window must be at least 2."
        raise ValueError(msg)
    values = series.astype(float)
    rolling_mean = values.rolling(window=window, min_periods=window).mean()
    rolling_std = values.rolling(window=window, min_periods=window).std()
    return (values - rolling_mean) / rolling_std
