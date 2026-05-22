"""Tests for baseline forecasting models."""

import pandas as pd
import pytest

from macro_data_forecasting.models.baselines import (
    fit_predict_lightgbm,
    fit_predict_ridge,
    naive_last_value_predictions,
)


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_timestamp": ["2026-03-11", "2026-04-10", "2026-05-12"],
            "target_value": [1.0, 2.0, 3.0],
            "feature_UNRATE_latest": [4.0, 4.1, 4.2],
        },
    )


def test_naive_predictor_uses_previous_target_only() -> None:
    """Verify naive predictions use only the previous target value."""
    predictions = naive_last_value_predictions(_dataset())

    assert predictions.iloc[1] == pytest.approx(1.0)
    assert predictions.iloc[2] == pytest.approx(2.0)


def test_naive_predictor_first_prediction_is_nan() -> None:
    """Verify first naive prediction is missing."""
    predictions = naive_last_value_predictions(_dataset())

    assert pd.isna(predictions.iloc[0])


def test_naive_predictor_does_not_use_current_target() -> None:
    """Verify naive prediction is not the current row target."""
    predictions = naive_last_value_predictions(_dataset())

    assert predictions.iloc[1] != _dataset().loc[1, "target_value"]


def test_fit_predict_ridge_returns_expected_shape() -> None:
    """Verify ridge baseline returns one prediction per test row."""
    train = pd.DataFrame(
        {
            "target_value": [1.0, 2.0, 3.0],
            "feature_x": [1.0, 2.0, 3.0],
        },
    )
    test = pd.DataFrame({"target_value": [4.0], "feature_x": [4.0]})

    predictions = fit_predict_ridge(train, test, ["feature_x"])

    assert predictions.shape == (1,)


def test_fit_predict_ridge_raises_with_no_features() -> None:
    """Verify ridge baseline requires feature columns."""
    train = pd.DataFrame({"target_value": [1.0, 2.0]})
    test = pd.DataFrame({"target_value": [3.0]})

    with pytest.raises(ValueError, match="at least one feature"):
        fit_predict_ridge(train, test, [])


def test_fit_predict_ridge_handles_nan_features_with_imputation() -> None:
    """Verify ridge baseline imputes missing feature values."""
    train = pd.DataFrame(
        {
            "target_value": [1.0, 2.0, 3.0],
            "feature_x": [1.0, None, 3.0],
        },
    )
    test = pd.DataFrame({"target_value": [4.0], "feature_x": [None]})

    predictions = fit_predict_ridge(train, test, ["feature_x"])

    assert predictions.shape == (1,)
    assert pd.notna(predictions[0])


def test_fit_predict_lightgbm_returns_expected_shape() -> None:
    """Verify LightGBM baseline returns one prediction per test row."""
    train = pd.DataFrame(
        {
            "target_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_x": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )
    test = pd.DataFrame({"target_value": [6.0], "feature_x": [6.0]})

    predictions = fit_predict_lightgbm(train, test, ["feature_x"])

    assert predictions.shape == (1,)
    assert pd.notna(predictions[0])


def test_fit_predict_lightgbm_raises_with_no_features() -> None:
    """Verify LightGBM baseline requires feature columns."""
    train = pd.DataFrame({"target_value": [1.0, 2.0]})
    test = pd.DataFrame({"target_value": [3.0]})

    with pytest.raises(ValueError, match="at least one feature"):
        fit_predict_lightgbm(train, test, [])


def test_fit_predict_lightgbm_handles_nan_features_with_imputation() -> None:
    """Verify LightGBM baseline imputes missing feature values."""
    train = pd.DataFrame(
        {
            "target_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_x": [1.0, None, 3.0, 4.0, 5.0],
        },
    )
    test = pd.DataFrame({"target_value": [6.0], "feature_x": [None]})

    predictions = fit_predict_lightgbm(train, test, ["feature_x"])

    assert predictions.shape == (1,)
    assert pd.notna(predictions[0])


def test_fit_predict_lightgbm_predictions_are_numeric() -> None:
    """Verify LightGBM predictions are numeric."""
    train = pd.DataFrame(
        {
            "target_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_x": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )
    test = pd.DataFrame({"target_value": [6.0], "feature_x": [6.0]})

    predictions = fit_predict_lightgbm(train, test, ["feature_x"])

    assert isinstance(float(predictions[0]), float)
