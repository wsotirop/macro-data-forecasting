"""Tests for point-in-time dataset contract helpers."""

import pandas as pd
import pytest

from macro_data_forecasting.features.dataset_contract import (
    FEATURE_DATASET_BASE_COLUMNS,
    create_empty_feature_dataset,
    validate_target_frame,
)


def _target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": ["cpi_mom", "cpi_mom"],
            "target_name": [
                "Headline CPI month-over-month inflation",
                "Headline CPI month-over-month inflation",
            ],
            "series_id": ["CUSR0000SA0", "CUSR0000SA0"],
            "source": ["bls", "bls"],
            "reference_date": ["2026-02-01", "2026-03-01"],
            "release_date": ["2026-03-11", "2026-04-10"],
            "target_value": [1.0, 2.0],
            "target_units": ["percent", "percent"],
        },
    )


def test_validate_target_frame_passes_valid_targets() -> None:
    """Verify valid target frames normalize and sort."""
    validated = validate_target_frame(_target_frame())

    assert len(validated) == 2
    assert validated.loc[0, "reference_date"].isoformat() == "2026-02-01"


def test_validate_target_frame_rejects_missing_required_columns() -> None:
    """Verify missing target columns are rejected."""
    targets = _target_frame().drop(columns=["target_units"])

    with pytest.raises(ValueError, match="Missing required target columns"):
        validate_target_frame(targets)


def test_validate_target_frame_rejects_duplicate_targets() -> None:
    """Verify duplicate target_id and reference_date rows are rejected."""
    targets = pd.concat([_target_frame(), _target_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate target rows"):
        validate_target_frame(targets)


def test_validate_target_frame_rejects_release_before_month_end() -> None:
    """Verify CPI release dates must follow the reference month end."""
    targets = _target_frame()
    targets.loc[0, "release_date"] = "2026-02-28"

    with pytest.raises(ValueError, match="after the reference month end"):
        validate_target_frame(targets)


def test_validate_target_frame_rejects_missing_release_date() -> None:
    """Verify target release dates cannot be missing."""
    targets = _target_frame()
    targets.loc[0, "release_date"] = pd.NaT

    with pytest.raises(ValueError, match="release_date cannot be missing"):
        validate_target_frame(targets)


def test_create_empty_feature_dataset_creates_shell_columns() -> None:
    """Verify the dataset shell has only base target columns."""
    dataset = create_empty_feature_dataset(_target_frame())

    assert list(dataset.columns) == FEATURE_DATASET_BASE_COLUMNS
    assert dataset.loc[0, "forecast_timestamp"].isoformat() == "2026-03-11"
    assert dataset.loc[0, "target_reference_date"].isoformat() == "2026-02-01"
