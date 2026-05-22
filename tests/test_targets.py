"""Tests for CPI target construction."""

import pandas as pd
import pytest

from macro_data_forecasting.features.targets import (
    TARGET_COLUMNS,
    TargetConstructionError,
    build_cpi_mom_target,
)


def _cpi_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["CUSR0000SA0", "CUSR0000SA0", "CUSR0000SA0"],
            "source": ["bls", "bls", "bls"],
            "date": ["2026-01-01", "2026-02-01", "2026-03-01"],
            "value": [100.0, 101.0, 103.02],
            "release_date": ["2026-02-12", "2026-03-11", "2026-04-10"],
        },
    )


def test_build_cpi_mom_target_computes_values() -> None:
    """Verify CPI MoM target values are percent changes."""
    targets = build_cpi_mom_target(_cpi_observations())

    assert targets.loc[0, "target_value"] == pytest.approx(1.0)
    assert targets.loc[1, "target_value"] == pytest.approx(2.0)


def test_build_cpi_mom_target_drops_first_row() -> None:
    """Verify first CPI row is dropped because MoM cannot be computed."""
    targets = build_cpi_mom_target(_cpi_observations())

    assert len(targets) == 2
    assert targets.loc[0, "reference_date"].isoformat() == "2026-02-01"


def test_build_cpi_mom_target_missing_release_date_raises_strict() -> None:
    """Verify strict mode rejects missing CPI release dates."""
    observations = _cpi_observations()
    observations.loc[1, "release_date"] = pd.NaT

    with pytest.raises(TargetConstructionError, match="requires release_date"):
        build_cpi_mom_target(observations)


def test_build_cpi_mom_target_missing_release_date_allowed_non_strict() -> None:
    """Verify non-strict mode can build targets with missing release dates."""
    observations = _cpi_observations()
    observations.loc[1, "release_date"] = pd.NaT

    targets = build_cpi_mom_target(observations, strict_release_dates=False)

    assert pd.isna(targets.loc[0, "release_date"])


def test_build_cpi_mom_target_duplicate_rows_raise() -> None:
    """Verify duplicate date and release-date rows are rejected."""
    observations = pd.concat(
        [_cpi_observations(), _cpi_observations().iloc[[1]]],
        ignore_index=True,
    )

    with pytest.raises(TargetConstructionError, match="Duplicate CPI observations"):
        build_cpi_mom_target(observations)


def test_build_cpi_mom_target_output_columns_exact() -> None:
    """Verify CPI target output columns match the contract."""
    targets = build_cpi_mom_target(_cpi_observations())

    assert list(targets.columns) == TARGET_COLUMNS
