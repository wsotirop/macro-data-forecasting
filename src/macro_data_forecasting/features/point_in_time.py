"""Point-in-time feature dataset construction placeholders."""

import pandas as pd


def build_point_in_time_dataset(observations: pd.DataFrame) -> pd.DataFrame:
    """Build features using only observations available by `release_date`."""
    # Stage 3 will construct feature matrices from release_date availability,
    # not final revised values or reference dates alone.
    raise NotImplementedError("TODO: implement point-in-time features in Stage 3.")
