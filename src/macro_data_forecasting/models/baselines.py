"""Baseline model placeholders."""

from typing import Any

import pandas as pd


def fit_naive_baseline(features: pd.DataFrame, target: pd.Series) -> Any:
    """Fit a simple baseline model in a later modeling stage."""
    raise NotImplementedError("TODO: implement baseline models in Stage 4.")
