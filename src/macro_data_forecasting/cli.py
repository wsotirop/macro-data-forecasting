"""Command-line interface for macro data forecasting workflows."""

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

from macro_data_forecasting.sources.bls import BlsClient
from macro_data_forecasting.sources.bls_release_calendar import (
    load_cpi_release_calendar,
    map_cpi_release_dates,
)
from macro_data_forecasting.sources.fred import FredClient


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
    fetch_bls.add_argument("--database-url")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch-fred":
        client = FredClient()
        observations = client.fetch_series_observations(
            series_id=args.series_id,
            observation_start=args.observation_start,
            observation_end=args.observation_end,
            realtime_start=args.realtime_start,
            realtime_end=args.realtime_end,
            frequency=args.frequency,
            aggregation_method=args.aggregation_method,
        )
        row_count = client.store(observations, database_url=args.database_url)
        print(f"Inserted {row_count} rows for {args.series_id}.")
        return 0

    if args.command == "fetch-bls":
        client = BlsClient()
        observations = client.fetch_series_observations(
            series_id=args.series_id,
            start_year=args.start_year,
            end_year=args.end_year,
            annual_average=args.annual_average,
        )
        if args.release_calendar is not None:
            calendar = load_cpi_release_calendar(args.release_calendar)
            observations = map_cpi_release_dates(observations, calendar)
        else:
            warnings.warn(
                "No CPI release calendar provided; BLS release_date values will "
                "remain missing. Provide --release-calendar for point-in-time CPI "
                "storage.",
                RuntimeWarning,
                stacklevel=2,
            )
        row_count = client.store(observations, database_url=args.database_url)
        print(f"Inserted {row_count} rows for {args.series_id}.")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
