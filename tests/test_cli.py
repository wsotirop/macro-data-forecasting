"""Tests for command-line ingestion workflows."""

import pandas as pd
import pytest
from sqlalchemy import func, select

import macro_data_forecasting.cli as cli
from macro_data_forecasting.database import (
    finish_ingestion_run,
    get_engine,
    ingestion_runs,
    macro_observations,
    start_ingestion_run,
    upsert_observations,
)
from macro_data_forecasting.features.targets import TargetConstructionError
from macro_data_forecasting.sources.bls_release_calendar import CalendarCoverageError


class FakeFredClient:
    """Fake FRED client for CLI tests."""

    def fetch_series_observations(self, **kwargs) -> pd.DataFrame:
        """Return one normalized FRED observation."""
        return pd.DataFrame(
            {
                "series_id": [kwargs["series_id"]],
                "date": ["2020-01-01"],
                "value": [258.678],
                "source": ["fred"],
                "release_date": ["2020-02-13"],
                "fetched_at": ["2026-05-21T21:00:00Z"],
            },
        )

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return already-normalized test observations."""
        return data

    def store(
        self,
        data: pd.DataFrame,
        database_url: str | None = None,
    ) -> dict[str, int]:
        """Store test observations through the real upsert helper."""
        return upsert_observations(data, database_url=database_url)


class FakeBlsClient:
    """Fake BLS client for CLI tests."""

    def fetch_series_observations(self, **kwargs) -> pd.DataFrame:
        """Return one BLS observation outside the sample calendar coverage."""
        return pd.DataFrame(
            {
                "series_id": [kwargs["series_id"]],
                "date": ["2026-05-01"],
                "value": [320.0],
                "source": ["bls"],
                "release_date": [pd.NaT],
                "fetched_at": ["2026-05-21T21:00:00Z"],
            },
        )

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return already-normalized test observations."""
        return data

    def store(
        self,
        data: pd.DataFrame,
        database_url: str | None = None,
    ) -> dict[str, int]:
        """Fail if strict-calendar preflight does not stop storage."""
        raise AssertionError("store should not be called")


def test_cli_fetch_fred_records_ingestion_run(monkeypatch, tmp_path, capsys) -> None:
    """Verify fetch-fred uses upsert storage and records a succeeded run."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    monkeypatch.setattr(cli, "FredClient", FakeFredClient)

    result = cli.main(
        [
            "fetch-fred",
            "--series-id",
            "CPIAUCSL",
            "--start",
            "2020-01-01",
            "--database-url",
            database_url,
        ],
    )

    output = capsys.readouterr().out
    engine = get_engine(database_url)
    with engine.connect() as connection:
        run = connection.execute(select(ingestion_runs)).mappings().one()
        stored_count = connection.scalar(
            select(func.count()).select_from(macro_observations),
        )
    engine.dispose()

    assert result == 0
    assert "Rows seen: 1" in output
    assert "Inserted: 1" in output
    assert run["status"] == "succeeded"
    assert stored_count == 1


def test_cli_fetch_bls_strict_calendar_fails_before_storage(
    monkeypatch,
    tmp_path,
) -> None:
    """Verify strict BLS calendar coverage failures are recorded before storage."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    monkeypatch.setattr(cli, "BlsClient", FakeBlsClient)

    with pytest.raises(CalendarCoverageError, match="2026-05"):
        cli.main(
            [
                "fetch-bls",
                "--series-id",
                "CUSR0000SA0",
                "--start-year",
                "2026",
                "--end-year",
                "2026",
                "--release-calendar",
                "data/reference/cpi_release_calendar_sample.csv",
                "--strict-calendar",
                "--database-url",
                database_url,
            ],
        )

    engine = get_engine(database_url)
    with engine.connect() as connection:
        run = connection.execute(select(ingestion_runs)).mappings().one()
        stored_count = connection.scalar(
            select(func.count()).select_from(macro_observations),
        )
    engine.dispose()

    assert run["status"] == "failed"
    assert "2026-05" in run["error_message"]
    assert stored_count == 0


def test_cli_list_runs_outputs_table(tmp_path, capsys) -> None:
    """Verify list-runs prints recent ingestion run rows."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    run_id = start_ingestion_run(
        "fred",
        "CPIAUCSL",
        {"observation_start": "2020-01-01"},
        database_url=database_url,
    )
    finish_ingestion_run(
        run_id,
        {"rows_seen": 1, "inserted": 1, "updated": 0, "skipped": 0},
        database_url=database_url,
    )

    result = cli.main(["list-runs", "--database-url", database_url])

    output = capsys.readouterr().out
    assert result == 0
    assert "CPIAUCSL" in output
    assert "succeeded" in output
    assert "rows_seen" in output


def test_cli_show_run_outputs_details(tmp_path, capsys) -> None:
    """Verify show-run prints full ingestion run details."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    run_id = start_ingestion_run(
        "bls",
        "CUSR0000SA0",
        {"start_year": 2026},
        database_url=database_url,
    )

    result = cli.main(
        [
            "show-run",
            "--run-id",
            str(run_id),
            "--database-url",
            database_url,
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert f"id: {run_id}" in output
    assert "source: bls" in output
    assert "parameters_json" in output


def test_cli_coverage_outputs_summary(tmp_path, capsys) -> None:
    """Verify coverage prints stored observation summary rows."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    upsert_observations(
        pd.DataFrame(
            {
                "series_id": ["CPIAUCSL", "CPIAUCSL"],
                "date": ["2020-01-01", "2020-02-01"],
                "value": [258.678, 259.007],
                "source": ["fred", "fred"],
                "release_date": ["2020-02-13", "2020-03-11"],
                "fetched_at": ["2026-05-21T21:00:00Z", "2026-05-21T21:00:00Z"],
            },
        ),
        database_url=database_url,
    )

    result = cli.main(
        [
            "coverage",
            "--source",
            "fred",
            "--database-url",
            database_url,
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "CPIAUCSL" in output
    assert "row_count" in output
    assert "2020-01-01" in output
    assert "missing_release_date_count" in output


def _seed_cpi_observations(
    database_url: str,
    missing_release_date: bool = False,
) -> None:
    release_dates = ["2026-02-12", "2026-03-11", "2026-04-10"]
    if missing_release_date:
        release_dates[1] = pd.NaT
    upsert_observations(
        pd.DataFrame(
            {
                "series_id": ["CUSR0000SA0", "CUSR0000SA0", "CUSR0000SA0"],
                "date": ["2026-01-01", "2026-02-01", "2026-03-01"],
                "value": [100.0, 101.0, 103.02],
                "source": ["bls", "bls", "bls"],
                "release_date": release_dates,
                "fetched_at": ["2026-05-21T21:00:00Z"] * 3,
            },
        ),
        database_url=database_url,
    )


def _seed_feature_observations(database_url: str) -> None:
    upsert_observations(
        pd.DataFrame(
            {
                "series_id": ["UNRATE", "UNRATE", "FEDFUNDS"],
                "date": ["2026-01-01", "2026-02-01", "2026-03-01"],
                "value": [4.0, 4.1, 4.5],
                "source": ["fred", "fred", "fred"],
                "release_date": ["2026-02-06", "2026-03-06", "2026-04-30"],
                "fetched_at": ["2026-05-21T21:00:00Z"] * 3,
            },
        ),
        database_url=database_url,
    )


def test_cli_build_cpi_target_prints_shell(tmp_path, capsys) -> None:
    """Verify build-cpi-target builds a shell from stored observations."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    _seed_cpi_observations(database_url)

    result = cli.main(
        [
            "build-cpi-target",
            "--series-id",
            "CUSR0000SA0",
            "--database-url",
            database_url,
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "cpi_mom" in output
    assert "Rows: 2" in output


def test_cli_build_cpi_target_fails_missing_release_dates(tmp_path) -> None:
    """Verify strict CLI target construction rejects missing release dates."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    _seed_cpi_observations(database_url, missing_release_date=True)

    with pytest.raises(TargetConstructionError, match="requires release_date"):
        cli.main(
            [
                "build-cpi-target",
                "--series-id",
                "CUSR0000SA0",
                "--database-url",
                database_url,
            ],
        )


def test_cli_build_cpi_target_allows_missing_release_dates_when_requested(
    tmp_path,
    capsys,
) -> None:
    """Verify diagnostic target shell can be built with missing release dates."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    _seed_cpi_observations(database_url, missing_release_date=True)

    result = cli.main(
        [
            "build-cpi-target",
            "--series-id",
            "CUSR0000SA0",
            "--allow-missing-release-dates",
            "--database-url",
            database_url,
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Rows: 2" in output


def test_cli_build_cpi_target_writes_output_csv(tmp_path) -> None:
    """Verify build-cpi-target writes CSV output when requested."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    output_path = tmp_path / "cpi_target_shell.csv"
    _seed_cpi_observations(database_url)

    result = cli.main(
        [
            "build-cpi-target",
            "--series-id",
            "CUSR0000SA0",
            "--output",
            str(output_path),
            "--database-url",
            database_url,
        ],
    )

    written = pd.read_csv(output_path)
    assert result == 0
    assert output_path.exists()
    assert len(written) == 2
    assert list(written.columns) == [
        "forecast_timestamp",
        "target_id",
        "target_reference_date",
        "target_release_date",
        "target_value",
    ]


def test_cli_build_feature_matrix_prints_preview(tmp_path, capsys) -> None:
    """Verify build-feature-matrix builds point-in-time feature columns."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    _seed_cpi_observations(database_url)
    _seed_feature_observations(database_url)

    result = cli.main(
        [
            "build-feature-matrix",
            "--target-series-id",
            "CUSR0000SA0",
            "--features",
            "UNRATE",
            "FEDFUNDS",
            "--database-url",
            database_url,
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "feature_UNRATE_latest" in output
    assert "feature_FEDFUNDS_latest" in output
    assert "Shape: (2, 7)" in output


def test_cli_build_feature_matrix_adds_lagged_target(tmp_path, capsys) -> None:
    """Verify build-feature-matrix can include lagged target features."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    _seed_cpi_observations(database_url)
    _seed_feature_observations(database_url)

    result = cli.main(
        [
            "build-feature-matrix",
            "--target-series-id",
            "CUSR0000SA0",
            "--features",
            "UNRATE",
            "--include-lagged-target",
            "--database-url",
            database_url,
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "feature_target_lag_1" in output


def test_cli_build_feature_matrix_writes_output_csv(tmp_path) -> None:
    """Verify build-feature-matrix writes CSV output when requested."""
    database_url = f"sqlite:///{(tmp_path / 'macro.sqlite').as_posix()}"
    output_path = tmp_path / "cpi_feature_matrix.csv"
    _seed_cpi_observations(database_url)
    _seed_feature_observations(database_url)

    result = cli.main(
        [
            "build-feature-matrix",
            "--target-series-id",
            "CUSR0000SA0",
            "--features",
            "UNRATE",
            "--output",
            str(output_path),
            "--database-url",
            database_url,
        ],
    )

    written = pd.read_csv(output_path)
    assert result == 0
    assert output_path.exists()
    assert "feature_UNRATE_latest" in written.columns
    assert len(written) == 2


def _validation_dataset(rows: int = 8) -> pd.DataFrame:
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


def test_cli_validate_model_works_on_feature_matrix_csv(tmp_path, capsys) -> None:
    """Verify validate-model prints metrics for a feature matrix CSV."""
    dataset_path = tmp_path / "feature_matrix.csv"
    _validation_dataset().to_csv(dataset_path, index=False)

    result = cli.main(
        [
            "validate-model",
            "--dataset",
            str(dataset_path),
            "--model",
            "naive_last_value",
            "--min-train-size",
            "3",
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Model: naive_last_value" in output
    assert "rmse:" in output


def test_cli_validate_model_writes_forecasts_csv(tmp_path) -> None:
    """Verify validate-model writes forecast-level predictions."""
    dataset_path = tmp_path / "feature_matrix.csv"
    output_path = tmp_path / "forecasts.csv"
    _validation_dataset().to_csv(dataset_path, index=False)

    result = cli.main(
        [
            "validate-model",
            "--dataset",
            str(dataset_path),
            "--model",
            "naive_last_value",
            "--min-train-size",
            "3",
            "--output",
            str(output_path),
        ],
    )

    written = pd.read_csv(output_path)
    assert result == 0
    assert output_path.exists()
    assert len(written) == 5
    assert "prediction" in written.columns


def test_cli_validate_model_ridge_reports_naive_comparison(tmp_path, capsys) -> None:
    """Verify ridge validation also prints naive comparison."""
    dataset_path = tmp_path / "feature_matrix.csv"
    _validation_dataset().to_csv(dataset_path, index=False)

    result = cli.main(
        [
            "validate-model",
            "--dataset",
            str(dataset_path),
            "--model",
            "ridge",
            "--min-train-size",
            "3",
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Naive comparison:" in output
    assert "model_beats_naive_rmse" in output


def test_cli_compare_models_prints_metrics_table(tmp_path, capsys) -> None:
    """Verify compare-models prints comparison metrics for requested models."""
    dataset_path = tmp_path / "feature_matrix.csv"
    _validation_dataset().to_csv(dataset_path, index=False)

    result = cli.main(
        [
            "compare-models",
            "--dataset",
            str(dataset_path),
            "--models",
            "naive_last_value",
            "ridge",
            "--min-train-size",
            "3",
        ],
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "model_name" in output
    assert "naive_last_value" in output
    assert "ridge" in output
    assert "rmse_vs_naive" in output
    assert "naive on RMSE" in output


def test_cli_compare_models_writes_outputs(tmp_path) -> None:
    """Verify compare-models writes forecast and metrics CSV outputs."""
    dataset_path = tmp_path / "feature_matrix.csv"
    output_dir = tmp_path / "reports"
    _validation_dataset().to_csv(dataset_path, index=False)

    result = cli.main(
        [
            "compare-models",
            "--dataset",
            str(dataset_path),
            "--models",
            "naive_last_value",
            "ridge",
            "--min-train-size",
            "3",
            "--output-dir",
            str(output_dir),
            "--prefix",
            "cpi",
        ],
    )

    forecasts = output_dir / "cpi_forecasts.csv"
    metrics = output_dir / "cpi_metrics.csv"
    assert result == 0
    assert forecasts.exists()
    assert metrics.exists()
    assert "prediction" in pd.read_csv(forecasts).columns
    assert "rmse_vs_naive" in pd.read_csv(metrics).columns
