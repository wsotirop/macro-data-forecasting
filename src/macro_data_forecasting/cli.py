"""Command-line interface for macro data forecasting workflows."""

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from macro_data_forecasting.database import (
    get_ingestion_run,
    list_ingestion_runs,
    load_observations,
    summarize_observation_coverage,
)
from macro_data_forecasting.evaluation.metrics import (
    compare_to_naive,
    evaluate_forecasts,
)
from macro_data_forecasting.features.dataset_contract import (
    FEATURE_DATASET_BASE_COLUMNS,
    create_empty_feature_dataset,
    validate_target_frame,
)
from macro_data_forecasting.features.feature_matrix import (
    add_lagged_target_features,
    build_point_in_time_feature_matrix,
)
from macro_data_forecasting.features.targets import build_cpi_mom_target
from macro_data_forecasting.models.comparison import (
    run_model_comparison,
    save_model_comparison_outputs,
)
from macro_data_forecasting.models.validation import walk_forward_validate
from macro_data_forecasting.reports.generate_report import (
    generate_model_comparison_report,
)
from macro_data_forecasting.sources.bls import BlsClient
from macro_data_forecasting.sources.bls_release_calendar import (
    assert_calendar_coverage,
    load_cpi_release_calendar,
    map_cpi_release_dates,
    normalize_reference_period,
)
from macro_data_forecasting.sources.fred import FredClient
from macro_data_forecasting.sources.ingestion import run_ingestion


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="macro-data-forecasting")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_fred = subparsers.add_parser("fetch-fred", help="Fetch and store FRED data")
    fetch_fred.add_argument("--series-id", required=True)
    fetch_fred.add_argument("--start", dest="observation_start")
    fetch_fred.add_argument("--end", dest="observation_end")
    fetch_fred.add_argument("--realtime-start")
    fetch_fred.add_argument("--realtime-end")
    fetch_fred.add_argument("--frequency")
    fetch_fred.add_argument("--aggregation-method")
    fetch_fred.add_argument("--database-url")

    fetch_bls = subparsers.add_parser("fetch-bls", help="Fetch and store BLS data")
    fetch_bls.add_argument("--series-id", required=True)
    fetch_bls.add_argument("--start-year", type=int, required=True)
    fetch_bls.add_argument("--end-year", type=int, required=True)
    fetch_bls.add_argument("--annual-average", action="store_true")
    fetch_bls.add_argument("--release-calendar", type=Path)
    fetch_bls.add_argument("--strict-calendar", action="store_true")
    fetch_bls.add_argument("--database-url")

    validate_cpi_calendar = subparsers.add_parser(
        "validate-cpi-calendar",
        help="Validate a local CPI release calendar",
    )
    validate_cpi_calendar.add_argument("--calendar", type=Path, required=True)
    validate_cpi_calendar.add_argument("--start-period")
    validate_cpi_calendar.add_argument("--end-period")

    list_runs = subparsers.add_parser("list-runs", help="List ingestion runs")
    list_runs.add_argument("--limit", type=int, default=10)
    list_runs.add_argument("--source")
    list_runs.add_argument("--status")
    list_runs.add_argument("--database-url")

    show_run = subparsers.add_parser("show-run", help="Show one ingestion run")
    show_run.add_argument("--run-id", type=int, required=True)
    show_run.add_argument("--database-url")

    coverage = subparsers.add_parser(
        "coverage",
        help="Summarize stored observation coverage",
    )
    coverage.add_argument("--source")
    coverage.add_argument("--series-id")
    coverage.add_argument("--database-url")

    build_cpi_target = subparsers.add_parser(
        "build-cpi-target",
        help="Build headline CPI MoM target shell",
    )
    build_cpi_target.add_argument("--series-id", default="CUSR0000SA0")
    build_cpi_target.add_argument("--source", default="bls")
    build_cpi_target.add_argument("--output", type=Path)
    build_cpi_target.add_argument("--allow-missing-release-dates", action="store_true")
    build_cpi_target.add_argument("--database-url")

    build_feature_matrix = subparsers.add_parser(
        "build-feature-matrix",
        help="Build a point-in-time CPI feature matrix",
    )
    build_feature_matrix.add_argument("--target-series-id", default="CUSR0000SA0")
    build_feature_matrix.add_argument("--target-source", default="bls")
    build_feature_matrix.add_argument("--features", nargs="+")
    build_feature_matrix.add_argument("--output", type=Path)
    build_feature_matrix.add_argument("--include-lagged-target", action="store_true")
    build_feature_matrix.add_argument(
        "--allow-missing-release-dates",
        action="store_true",
    )
    build_feature_matrix.add_argument("--database-url")

    validate_model = subparsers.add_parser(
        "validate-model",
        help="Run walk-forward validation for a baseline model",
    )
    validate_model.add_argument("--dataset", type=Path, required=True)
    validate_model.add_argument(
        "--model",
        choices=["naive_last_value", "ridge", "lightgbm"],
        required=True,
    )
    validate_model.add_argument("--min-train-size", type=int, default=24)
    validate_model.add_argument("--output", type=Path)

    compare_models = subparsers.add_parser(
        "compare-models",
        help="Run multiple walk-forward models on one feature matrix",
    )
    compare_models.add_argument("--dataset", type=Path, required=True)
    compare_models.add_argument(
        "--models",
        nargs="+",
        default=["naive_last_value", "ridge"],
    )
    compare_models.add_argument("--min-train-size", type=int, default=24)
    compare_models.add_argument("--output-dir", type=Path)
    compare_models.add_argument("--prefix", default="model_comparison")
    compare_models.add_argument("--report", type=Path)

    generate_report = subparsers.add_parser(
        "generate-report",
        help="Generate a markdown model comparison report from saved CSV outputs",
    )
    generate_report.add_argument("--metrics", type=Path, required=True)
    generate_report.add_argument("--forecasts", type=Path, required=True)
    generate_report.add_argument("--output", type=Path, required=True)
    generate_report.add_argument("--title", default="Macro Data Forecasting Report")
    generate_report.add_argument("--dataset", type=Path)
    generate_report.add_argument("--notes")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch-fred":
        client = FredClient()
        parameters = {
            "series_id": args.series_id,
            "observation_start": args.observation_start,
            "observation_end": args.observation_end,
            "realtime_start": args.realtime_start,
            "realtime_end": args.realtime_end,
            "frequency": args.frequency,
            "aggregation_method": args.aggregation_method,
        }
        result = run_ingestion(
            source="fred",
            series_id=args.series_id,
            fetch_fn=lambda: client.fetch_series_observations(**parameters),
            validate_fn=client.validate,
            store_fn=lambda data: client.store(data, database_url=args.database_url),
            parameters=parameters,
            database_url=args.database_url,
        )
        _print_ingestion_result(result, args.series_id)
        return 0

    if args.command == "fetch-bls":
        client = BlsClient()
        parameters = {
            "series_id": args.series_id,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "annual_average": args.annual_average,
            "release_calendar": str(args.release_calendar)
            if args.release_calendar is not None
            else None,
            "strict_calendar": args.strict_calendar,
        }

        def fetch_bls_observations() -> pd.DataFrame:
            observations = client.fetch_series_observations(
                series_id=args.series_id,
                start_year=args.start_year,
                end_year=args.end_year,
                annual_average=args.annual_average,
            )
            if args.strict_calendar and args.release_calendar is None:
                msg = "--strict-calendar requires --release-calendar."
                raise ValueError(msg)
            if args.release_calendar is not None:
                calendar = load_cpi_release_calendar(args.release_calendar)
                assert_calendar_coverage(
                    observations,
                    calendar,
                    strict=args.strict_calendar,
                )
                observations = map_cpi_release_dates(observations, calendar)
            else:
                warnings.warn(
                    "No CPI release calendar provided; BLS release_date values "
                    "will remain missing and will use the database missing-date "
                    "idempotency key.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return observations

        result = run_ingestion(
            source="bls",
            series_id=args.series_id,
            fetch_fn=fetch_bls_observations,
            validate_fn=client.validate,
            store_fn=lambda data: client.store(data, database_url=args.database_url),
            parameters=parameters,
            database_url=args.database_url,
        )
        _print_ingestion_result(result, args.series_id)
        return 0

    if args.command == "validate-cpi-calendar":
        calendar = load_cpi_release_calendar(args.calendar)
        if bool(args.start_period) != bool(args.end_period):
            parser.error("--start-period and --end-period must be provided together.")
        if args.start_period and args.end_period:
            start_period = normalize_reference_period(args.start_period)
            end_period = normalize_reference_period(args.end_period)
            if start_period > end_period:
                parser.error("--start-period must be before or equal to --end-period.")
            period_range = pd.period_range(start_period, end_period, freq="M")
            observations = pd.DataFrame(
                {
                    "date": [
                        period.to_timestamp().date() for period in period_range
                    ],
                },
            )
            assert_calendar_coverage(observations, calendar, strict=True)

        print(f"Calendar rows: {len(calendar)}")
        print(
            "Reference period range: "
            f"{calendar['reference_period'].min()} to "
            f"{calendar['reference_period'].max()}",
        )
        print(
            "Release date range: "
            f"{calendar['release_date'].min().isoformat()} to "
            f"{calendar['release_date'].max().isoformat()}",
        )
        return 0

    if args.command == "list-runs":
        runs = list_ingestion_runs(
            database_url=args.database_url,
            limit=args.limit,
            source=args.source,
            status=args.status,
        )
        display_columns = [
            "id",
            "source",
            "series_id",
            "status",
            "started_at",
            "finished_at",
            "rows_seen",
            "inserted",
            "updated",
            "skipped",
        ]
        _print_dataframe(runs.loc[:, display_columns], "No ingestion runs found.")
        return 0

    if args.command == "show-run":
        run = get_ingestion_run(args.run_id, database_url=args.database_url)
        if run is None:
            msg = f"Ingestion run not found: {args.run_id}"
            raise SystemExit(msg)
        _print_mapping(run)
        return 0

    if args.command == "coverage":
        coverage_summary = summarize_observation_coverage(
            database_url=args.database_url,
            source=args.source,
            series_id=args.series_id,
        )
        _print_dataframe(coverage_summary, "No observation coverage found.")
        return 0

    if args.command == "build-cpi-target":
        observations = load_observations(
            database_url=args.database_url,
            source=args.source,
            series_id=args.series_id,
        )
        targets = build_cpi_mom_target(
            observations,
            series_id=args.series_id,
            source=args.source,
            strict_release_dates=not args.allow_missing_release_dates,
        )
        if args.allow_missing_release_dates:
            dataset = _create_relaxed_feature_dataset(targets)
        else:
            validated_targets = validate_target_frame(targets)
            dataset = create_empty_feature_dataset(validated_targets)

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(args.output, index=False)
            print(f"Wrote {len(dataset)} rows to {args.output}")
        else:
            _print_dataframe(dataset.head(10), "No CPI target rows built.")
            print(f"Rows: {len(dataset)}")
        return 0

    if args.command == "build-feature-matrix":
        observations = load_observations(database_url=args.database_url)
        targets = build_cpi_mom_target(
            observations,
            series_id=args.target_series_id,
            source=args.target_source,
            strict_release_dates=not args.allow_missing_release_dates,
        )
        validated_targets = validate_target_frame(targets)
        target_shell = create_empty_feature_dataset(validated_targets)
        feature_matrix = build_point_in_time_feature_matrix(
            target_shell,
            observations,
            feature_series=args.features,
            strict_release_dates=not args.allow_missing_release_dates,
        )
        if args.include_lagged_target:
            feature_matrix = add_lagged_target_features(feature_matrix)

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            feature_matrix.to_csv(args.output, index=False)
            print(f"Wrote {len(feature_matrix)} rows to {args.output}")
        else:
            _print_dataframe(feature_matrix.head(10), "No feature matrix rows built.")
            print(f"Shape: {feature_matrix.shape}")
        return 0

    if args.command == "validate-model":
        dataset = pd.read_csv(args.dataset)
        forecasts = walk_forward_validate(
            dataset,
            model_name=args.model,
            min_train_size=args.min_train_size,
        )
        metrics = evaluate_forecasts(forecasts)
        print(f"Model: {args.model}")
        _print_metrics(metrics)

        if args.model != "naive_last_value":
            naive_forecasts = walk_forward_validate(
                dataset,
                model_name="naive_last_value",
                min_train_size=args.min_train_size,
            )
            comparison = compare_to_naive(forecasts, naive_forecasts)
            print("Naive comparison:")
            _print_metrics(comparison)
            if comparison["model_beats_naive_rmse"] == 1.0:
                print(f"{args.model} beats naive on RMSE.")
            else:
                print(f"{args.model} does not beat naive on RMSE.")
            if comparison["model_beats_naive_mae"] == 1.0:
                print(f"{args.model} beats naive on MAE.")
            else:
                print(f"{args.model} does not beat naive on MAE.")

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            forecasts.to_csv(args.output, index=False)
            print(f"Wrote {len(forecasts)} forecasts to {args.output}")
        return 0

    if args.command == "compare-models":
        dataset = pd.read_csv(args.dataset)
        forecasts, metrics = run_model_comparison(
            dataset,
            models=args.models,
            min_train_size=args.min_train_size,
        )
        _print_dataframe(metrics, "No model metrics produced.")
        _print_naive_comparisons(metrics)

        if args.output_dir is not None:
            paths = save_model_comparison_outputs(
                forecasts,
                metrics,
                output_dir=args.output_dir,
                prefix=args.prefix,
            )
            print(f"Wrote forecasts to {paths['forecasts']}")
            print(f"Wrote metrics to {paths['metrics']}")
        if args.report is not None:
            report_path = generate_model_comparison_report(
                metrics=metrics,
                forecasts=forecasts,
                output_path=args.report,
                dataset_path=args.dataset,
            )
            print(f"Wrote report to {report_path}")
        return 0

    if args.command == "generate-report":
        metrics = pd.read_csv(args.metrics)
        forecasts = pd.read_csv(args.forecasts)
        report_path = generate_model_comparison_report(
            metrics=metrics,
            forecasts=forecasts,
            output_path=args.output,
            title=args.title,
            dataset_path=args.dataset,
            notes=args.notes,
        )
        print(f"Wrote report to {report_path}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _print_dataframe(frame: pd.DataFrame, empty_message: str) -> None:
    if frame.empty:
        print(empty_message)
        return
    print(frame.to_string(index=False))


def _print_mapping(values: dict[str, object]) -> None:
    for key, value in values.items():
        print(f"{key}: {value}")


def _print_metrics(metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        print(f"{key}: {value}")


def _print_naive_comparisons(metrics: pd.DataFrame) -> None:
    benchmarked = metrics.loc[metrics["model_name"] != "naive_last_value"]
    for _index, row in benchmarked.iterrows():
        model_name = row["model_name"]
        if row["beats_naive_rmse"] == 1.0:
            print(f"{model_name} beats naive on RMSE.")
        else:
            print(f"{model_name} does not beat naive on RMSE.")
        if row["beats_naive_mae"] == 1.0:
            print(f"{model_name} beats naive on MAE.")
        else:
            print(f"{model_name} does not beat naive on MAE.")


def _create_relaxed_feature_dataset(targets: pd.DataFrame) -> pd.DataFrame:
    dataset = pd.DataFrame(
        {
            "forecast_timestamp": targets["release_date"],
            "target_id": targets["target_id"],
            "target_reference_date": targets["reference_date"],
            "target_release_date": targets["release_date"],
            "target_value": targets["target_value"],
        },
        columns=FEATURE_DATASET_BASE_COLUMNS,
    )
    return dataset


def _print_ingestion_result(result: dict[str, int], series_id: str) -> None:
    print(f"Ingestion run {result['run_id']} for {series_id}: {result['status']}")
    print(f"Rows seen: {result['rows_seen']}")
    print(f"Inserted: {result['inserted']}")
    print(f"Updated: {result['updated']}")
    print(f"Skipped: {result['skipped']}")


if __name__ == "__main__":
    raise SystemExit(main())
