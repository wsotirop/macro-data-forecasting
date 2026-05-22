"""Baseline model implementations."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def naive_last_value_predictions(dataset: pd.DataFrame) -> pd.Series:
    """Return previous target value predictions sorted by forecast timestamp."""
    if "forecast_timestamp" not in dataset.columns or "target_value" not in dataset:
        msg = "dataset must include forecast_timestamp and target_value columns."
        raise ValueError(msg)
    sorted_dataset = dataset.sort_values("forecast_timestamp").reset_index(drop=True)
    return sorted_dataset["target_value"].shift(1)


def fit_predict_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_value",
) -> np.ndarray:
    """Fit a ridge baseline on train rows and predict test rows."""
    train_features, test_features, train_targets = _validate_supervised_inputs(
        model_label="Ridge",
        train=train,
        test=test,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ],
    )
    pipeline.fit(train_features, train_targets)
    return pipeline.predict(test_features)


def fit_predict_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_value",
) -> np.ndarray:
    """Fit a fixed-default LightGBM baseline and predict test rows."""
    train_features, test_features, train_targets = _validate_supervised_inputs(
        model_label="LightGBM",
        train=train,
        test=test,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "lightgbm",
                LGBMRegressor(
                    objective="regression",
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=5,
                    random_state=42,
                    verbosity=-1,
                ),
            ),
        ],
    )
    pipeline.fit(train_features, train_targets)
    return pipeline.predict(test_features)


def _validate_supervised_inputs(
    model_label: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if not feature_columns:
        msg = f"{model_label} baseline requires at least one feature column."
        raise ValueError(msg)
    missing_features = [
        column
        for column in feature_columns
        if column not in train.columns or column not in test.columns
    ]
    if missing_features:
        msg = (
            f"{model_label} feature columns are missing from train/test data: "
            f"{missing_features}"
        )
        raise ValueError(msg)
    if target_column not in train.columns:
        msg = f"Training data missing target column: {target_column}"
        raise ValueError(msg)

    train_targets = pd.to_numeric(train[target_column], errors="coerce")
    valid_train = train_targets.notna()
    if valid_train.sum() < 2:
        msg = (
            f"{model_label} baseline requires at least two training rows "
            "with targets."
        )
        raise ValueError(msg)

    train_features = train.loc[valid_train, feature_columns]
    if train_features.dropna(axis=1, how="all").empty:
        msg = (
            f"{model_label} baseline has no usable feature columns after "
            "dropping all-NaN columns."
        )
        raise ValueError(msg)
    test_features = test.loc[:, feature_columns]
    return train_features, test_features, train_targets.loc[valid_train]
