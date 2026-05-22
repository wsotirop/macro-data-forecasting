"""Command-line interface for macro data forecasting workflows."""

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

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

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _print_ingestion_result(result: dict[str, int], series_id: str) -> None:
    print(f"Ingestion run {result['run_id']} for {series_id}: {result['status']}")
    print(f"Rows seen: {result['rows_seen']}")
    print(f"Inserted: {result['inserted']}")
    print(f"Updated: {result['updated']}")
    print(f"Skipped: {result['skipped']}")


if __name__ == "__main__":
    raise SystemExit(main())
