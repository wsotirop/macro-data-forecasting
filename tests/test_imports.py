"""Import tests for the Stage 1 scaffold."""


def test_main_modules_import() -> None:
    """Verify core package modules are importable."""
    import macro_data_forecasting.config  # noqa: F401
    import macro_data_forecasting.database  # noqa: F401
    import macro_data_forecasting.evaluation.metrics  # noqa: F401
    import macro_data_forecasting.features.point_in_time  # noqa: F401
    import macro_data_forecasting.features.transforms  # noqa: F401
    import macro_data_forecasting.logging_config  # noqa: F401
    import macro_data_forecasting.models.baselines  # noqa: F401
    import macro_data_forecasting.models.validation  # noqa: F401
    import macro_data_forecasting.reports.generate_report  # noqa: F401
    import macro_data_forecasting.sources.base  # noqa: F401
    import macro_data_forecasting.sources.bls  # noqa: F401
    import macro_data_forecasting.sources.bls_release_calendar  # noqa: F401
    import macro_data_forecasting.sources.fred  # noqa: F401
    import macro_data_forecasting.sources.ingestion  # noqa: F401
    import macro_data_forecasting.sources.market  # noqa: F401
    import macro_data_forecasting.sources.treasury  # noqa: F401


def test_metrics_compute_expected_values() -> None:
    """Verify basic metric implementations."""
    import pytest

    from macro_data_forecasting.evaluation.metrics import (
        directional_accuracy,
        mean_absolute_error,
        root_mean_squared_error,
    )

    y_true = [1.0, -2.0, 3.0]
    y_pred = [1.5, -1.0, -2.0]

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2.1666666667)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(2.9580398915)
    assert directional_accuracy(y_true, y_pred) == pytest.approx(2 / 3)
