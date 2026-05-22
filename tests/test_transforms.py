"""Tests for feature transformation helpers."""

import pandas as pd
import pytest

from macro_data_forecasting.features.transforms import (
    difference,
    percent_change,
    rolling_z_score,
)


def test_percent_change_works() -> None:
    """Verify percent change returns percentage units."""
    result = percent_change(pd.Series([100.0, 105.0, 110.25]))

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(5.0)
    assert result.iloc[2] == pytest.approx(5.0)


def test_difference_works() -> None:
    """Verify arithmetic differences are computed."""
    result = difference(pd.Series([100.0, 105.0, 103.0]))

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(5.0)
    assert result.iloc[2] == pytest.approx(-2.0)


def test_rolling_z_score_works_and_handles_insufficient_windows() -> None:
    """Verify rolling z-scores require full windows."""
    result = rolling_z_score(pd.Series([1.0, 2.0, 3.0]), window=3)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(1.0)
