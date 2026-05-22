"""Tests for point-in-time feature matrix construction."""

import pandas as pd
import pytest

from macro_data_forecasting.features.feature_matrix import (
    FeatureMatrixError,
    add_lagged_target_features,
    build_point_in_time_feature_matrix,
)


def _target_shell() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_timestamp": ["2026-03-11", "2026-04-10"],
            "target_id": ["cpi_mom", "cpi_mom"],
            "target_reference_date": ["2026-02-01", "2026-03-01"],
            "target_release_date": ["2026-03-11", "2026-04-10"],
            "target_value": [1.0, 2.0],
        },
    )


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["UNRATE", "UNRATE", "UNRATE", "FEDFUNDS"],
            "source": ["fred", "fred", "fred", "fred"],
            "date": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-03-01"],
            "value": [4.0, 4.1, 4.2, 4.5],
            "release_date": ["2026-02-06", "2026-03-06", "2026-04-04", "2026-04-30"],
        },
    )


def test_feature_matrix_uses_release_date_not_reference_date() -> None:
    """Verify future-released observations are excluded from feature values."""
    matrix = build_point_in_time_feature_matrix(
        _target_shell(),
        _observations(),
        feature_series=["UNRATE", "FEDFUNDS"],
    )

    assert matrix.loc[0, "feature_UNRATE_latest"] == pytest.approx(4.1)
    assert matrix.loc[1, "feature_UNRATE_latest"] == pytest.approx(4.2)
    assert pd.isna(matrix.loc[1, "feature_FEDFUNDS_latest"])


def test_feature_matrix_selects_latest_available_observation() -> None:
    """Verify latest release date and date are selected per forecast timestamp."""
    matrix = build_point_in_time_feature_matrix(
        _target_shell(),
        _observations(),
        feature_series=["UNRATE"],
    )

    assert list(matrix["feature_UNRATE_latest"]) == [4.1, 4.2]


def test_feature_matrix_respects_feature_series_filtering() -> None:
    """Verify only requested series IDs become feature columns."""
    matrix = build_point_in_time_feature_matrix(
        _target_shell(),
        _observations(),
        feature_series=["UNRATE"],
    )

    assert "feature_UNRATE_latest" in matrix.columns
    assert "feature_FEDFUNDS_latest" not in matrix.columns


def test_feature_matrix_missing_release_date_raises_strict() -> None:
    """Verify missing observation release dates are rejected in strict mode."""
    observations = _observations()
    observations.loc[0, "release_date"] = pd.NaT

    with pytest.raises(FeatureMatrixError, match="release_date cannot be missing"):
        build_point_in_time_feature_matrix(
            _target_shell(),
            observations,
            feature_series=["UNRATE"],
        )


def test_feature_matrix_outputs_expected_feature_columns() -> None:
    """Verify feature column names follow the series latest convention."""
    matrix = build_point_in_time_feature_matrix(
        _target_shell(),
        _observations(),
        feature_series=["UNRATE", "FEDFUNDS"],
    )

    assert list(matrix.columns) == [
        "forecast_timestamp",
        "target_id",
        "target_reference_date",
        "target_release_date",
        "target_value",
        "feature_UNRATE_latest",
        "feature_FEDFUNDS_latest",
    ]


def test_add_lagged_target_features_creates_lag_columns() -> None:
    """Verify lagged target features are created by target period."""
    lagged = add_lagged_target_features(_target_shell(), lags=[1])

    assert "feature_target_lag_1" in lagged.columns
    assert pd.isna(lagged.loc[0, "feature_target_lag_1"])
    assert lagged.loc[1, "feature_target_lag_1"] == pytest.approx(1.0)


def test_lagged_target_features_do_not_use_current_target() -> None:
    """Verify current target value is not copied into its own lag feature."""
    lagged = add_lagged_target_features(_target_shell(), lags=[1])

    assert lagged.loc[1, "feature_target_lag_1"] != lagged.loc[1, "target_value"]


def test_add_lagged_target_features_multiple_lags() -> None:
    """Verify multiple requested lags are added."""
    targets = pd.DataFrame(
        {
            "forecast_timestamp": ["2026-03-11", "2026-04-10", "2026-05-12"],
            "target_id": ["cpi_mom", "cpi_mom", "cpi_mom"],
            "target_reference_date": ["2026-02-01", "2026-03-01", "2026-04-01"],
            "target_release_date": ["2026-03-11", "2026-04-10", "2026-05-12"],
            "target_value": [1.0, 2.0, 3.0],
        },
    )

    lagged = add_lagged_target_features(targets, lags=[1, 2])

    assert lagged.loc[2, "feature_target_lag_1"] == pytest.approx(2.0)
    assert lagged.loc[2, "feature_target_lag_2"] == pytest.approx(1.0)
