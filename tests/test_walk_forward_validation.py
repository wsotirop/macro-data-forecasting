"""Tests for walk-forward validation."""

import pandas as pd
import pytest

import macro_data_forecasting.models.validation as validation


def _feature_dataset(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_timestamp": pd.date_range("2026-01-01", periods=rows, freq="MS"),
            "target_id": ["cpi_mom"] * rows,
            "target_reference_date": pd.date_range(
                "2025-12-01",
                periods=rows,
                freq="MS",
            ),
            "target_release_date": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="MS",
            ),
            "target_value": [float(index) for index in range(rows)],
            "feature_UNRATE_latest": [float(index) for index in range(rows)],
        },
    )


def test_naive_walk_forward_produces_correct_fold_count() -> None:
    """Verify naive walk-forward forecast count follows min_train_size."""
    forecasts = validation.walk_forward_validate(
        _feature_dataset(8),
        model_name="naive_last_value",
        min_train_size=3,
    )

    assert len(forecasts) == 5
    assert list(forecasts["fold_number"]) == [1, 2, 3, 4, 5]


def test_ridge_walk_forward_produces_correct_fold_count() -> None:
    """Verify ridge walk-forward forecast count follows min_train_size."""
    forecasts = validation.walk_forward_validate(
        _feature_dataset(8),
        model_name="ridge",
        min_train_size=3,
    )

    assert len(forecasts) == 5
    assert (forecasts["model_name"] == "ridge").all()


def test_lightgbm_walk_forward_produces_correct_fold_count() -> None:
    """Verify LightGBM walk-forward forecast count follows min_train_size."""
    forecasts = validation.walk_forward_validate(
        _feature_dataset(8),
        model_name="lightgbm",
        min_train_size=3,
    )

    assert len(forecasts) == 5
    assert (forecasts["model_name"] == "lightgbm").all()


def test_each_fold_trains_only_on_prior_rows(monkeypatch) -> None:
    """Verify ridge training slices exclude the current forecast row."""
    seen_slices: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spy_fit_predict(
        train: pd.DataFrame,
        test: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "target_value",
    ):
        seen_slices.append(
            (
                train["forecast_timestamp"].max(),
                test["forecast_timestamp"].iloc[0],
            ),
        )
        return [train[target_column].iloc[-1]]

    monkeypatch.setattr(validation, "fit_predict_ridge", spy_fit_predict)

    validation.walk_forward_validate(_feature_dataset(6), "ridge", min_train_size=3)

    assert seen_slices
    assert all(train_max < test_time for train_max, test_time in seen_slices)


def test_ridge_drops_all_nan_training_features_per_fold(monkeypatch) -> None:
    """Verify ridge validation excludes all-NaN training columns by fold."""
    dataset = _feature_dataset(6)
    dataset["feature_late_start"] = [None, None, None, None, 1.0, 2.0]
    seen_features: list[list[str]] = []

    def spy_fit_predict(
        train: pd.DataFrame,
        test: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "target_value",
    ):
        seen_features.append(feature_columns)
        return [train[target_column].iloc[-1]]

    monkeypatch.setattr(validation, "fit_predict_ridge", spy_fit_predict)

    validation.walk_forward_validate(dataset, "ridge", min_train_size=3)

    assert seen_features
    assert "feature_late_start" not in seen_features[0]
    assert "feature_UNRATE_latest" in seen_features[0]


def test_lightgbm_trains_only_on_prior_rows(monkeypatch) -> None:
    """Verify LightGBM training slices exclude current and future rows."""
    seen_slices: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spy_fit_predict(
        train: pd.DataFrame,
        test: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "target_value",
    ):
        seen_slices.append(
            (
                train["forecast_timestamp"].max(),
                test["forecast_timestamp"].iloc[0],
            ),
        )
        return [train[target_column].iloc[-1]]

    monkeypatch.setattr(validation, "fit_predict_lightgbm", spy_fit_predict)

    validation.walk_forward_validate(_feature_dataset(6), "lightgbm", min_train_size=3)

    assert seen_slices
    assert all(train_max < test_time for train_max, test_time in seen_slices)


def test_lightgbm_drops_all_nan_training_features_per_fold(monkeypatch) -> None:
    """Verify LightGBM validation excludes all-NaN training columns by fold."""
    dataset = _feature_dataset(6)
    dataset["feature_late_start"] = [None, None, None, None, 1.0, 2.0]
    seen_features: list[list[str]] = []

    def spy_fit_predict(
        train: pd.DataFrame,
        test: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "target_value",
    ):
        seen_features.append(feature_columns)
        return [train[target_column].iloc[-1]]

    monkeypatch.setattr(validation, "fit_predict_lightgbm", spy_fit_predict)

    validation.walk_forward_validate(dataset, "lightgbm", min_train_size=3)

    assert seen_features
    assert "feature_late_start" not in seen_features[0]
    assert "feature_UNRATE_latest" in seen_features[0]


def test_walk_forward_insufficient_rows_raises() -> None:
    """Verify validation rejects datasets without enough rows."""
    with pytest.raises(ValueError, match="Insufficient rows"):
        validation.walk_forward_validate(
            _feature_dataset(3),
            model_name="naive_last_value",
            min_train_size=3,
        )


def test_ridge_missing_feature_columns_raises() -> None:
    """Verify ridge validation requires feature columns."""
    dataset = _feature_dataset(6).drop(columns=["feature_UNRATE_latest"])

    with pytest.raises(ValueError, match="No feature columns"):
        validation.walk_forward_validate(dataset, model_name="ridge", min_train_size=3)
