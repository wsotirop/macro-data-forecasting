"""Time-series validation placeholders."""

from typing import Any

import pandas as pd


def expanding_window_walk_forward(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Run expanding-window walk-forward validation for macro forecasts."""
    # Macro forecasting validation must use chronological walk-forward splits.
    # Do not use k-fold cross-validation for time-series model evaluation.
    raise NotImplementedError("TODO: implement walk-forward validation in Stage 4.")
