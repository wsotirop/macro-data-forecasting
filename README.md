# Macro Data Forecasting

Macro Data Forecasting is a reusable Python framework for point-in-time macroeconomic forecasting and nowcasting. The first planned target is U.S. headline CPI month-over-month inflation, with the architecture intended to support additional macro releases and market reaction targets over time.

## Motivation

Macroeconomic forecasting workflows are easy to contaminate with data that was revised, released late, or unavailable at the forecast timestamp. This project is designed around point-in-time data availability so models can be evaluated under realistic information sets.

> Warning: this repository is designed to avoid lookahead bias. Features should be built from `release_date` availability, not from final revised data or reference periods alone.

## Planned Architecture

- `sources`: data-source clients for economic, government, and market data.
- `database`: SQLAlchemy schema and database initialization.
- `features`: point-in-time feature construction and transformations.
- `models`: baselines, model interfaces, and walk-forward validation.
- `evaluation`: forecast metrics and diagnostics.
- `reports`: generated markdown and chart outputs.

## Roadmap

- Stage 1: Scaffold
- Stage 2A: FRED/ALFRED ingestion
- Stage 2B: BLS CPI/Core CPI ingestion
- Stage 2C: CPI release calendar validation
- Stage 2D: Idempotent ingestion and run metadata
- Stage 2E: Ingestion audit and coverage CLI
- Stage 3A: CPI target construction and dataset contract
- Stage 3: Point-in-time feature engineering
- Stage 4: Modeling and walk-forward validation
- Stage 5: Automated reporting

## Ingestion Status

Stage 2A adds the first real ingestion path for FRED/ALFRED series. The current client fetches FRED `series/observations`, normalizes observations, validates the point-in-time columns, and stores rows in the local SQLAlchemy database.

Stage 2B adds BLS CPI/Core CPI ingestion for monthly CPI series and a separate CPI release-calendar mapping utility. BLS observations can be fetched without an API key, but exact `release_date` values require a local official release calendar CSV.

Stage 2C adds CPI release-calendar validation and coverage checks. Calendars are checked for schema, duplicate reference months, valid release dates, valid release times, and complete coverage over requested CPI observation periods.

Stage 2D adds idempotent upsert behavior and ingestion-run metadata. Re-running the same ingestion is safe: existing observations are matched by `series_id`, `source`, `date`, and `release_date`, then inserted, updated, or skipped without creating duplicate point-in-time rows.

Stage 2E adds audit commands for inspecting ingestion history and stored observation coverage from the command line.

Stage 3A adds the first target-construction contract: U.S. headline CPI month-over-month inflation. It builds a target shell from normalized CPI observations, validates target release dates, and creates an empty model-dataset shell.

No Treasury, market-data, full feature-engineering, or modeling logic is implemented yet.

## Point-In-Time Columns

Every stored observation distinguishes:

- `date`: the economic or reference date for the observation.
- `release_date`: the best available date when the value became observable.
- `fetched_at`: the timestamp when this system fetched the value.

For FRED/ALFRED, the observations endpoint provides realtime vintage metadata. Until exact release calendars are integrated, this project uses each observation's `realtime_start` vintage date as the best available `release_date` approximation. This preserves vintage awareness but should not be treated as a perfect official release timestamp for every series.

For BLS CPI data, `date` is the CPI reference month derived from BLS `year` and `period` fields. The BLS Public Data API does not guarantee exact release dates for every observation in the observation payload, so CPI `release_date` should be mapped from an official CPI release calendar. If no release calendar is supplied, BLS rows may keep `release_date` missing rather than silently approximating it.

Feature engineering and backtesting should require complete CPI release-calendar coverage before using CPI observations. Missing `release_date` values are not acceptable for strict point-in-time CPI work.

For rows where `release_date` is intentionally missing, such as unmapped BLS observations, the database stores a deterministic internal missing-date key so repeated ingestion still remains idempotent. This is only a storage safety mechanism; it is not a release-date approximation.

Each CLI ingestion command records an `ingestion_runs` row with source, series, parameters, status, timestamps, row counts, and any error message. Failed ingestion runs are recorded and the exception is still surfaced.

Auditability matters because reproducible macro forecasting depends on knowing which data was fetched, when it was fetched, which parameters were used, whether the run succeeded, and what observation coverage is available before building point-in-time features.

## CPI Target Construction

The initial target is headline CPI month-over-month inflation:

```text
target_value = 100 * (CPI_t / CPI_t-1 - 1)
```

Targets are labeled by CPI reference month and official CPI `release_date`. In strict mode, missing release dates are rejected rather than filled or approximated.

Stage 3A only creates the dataset shell:

- `forecast_timestamp`
- `target_id`
- `target_reference_date`
- `target_release_date`
- `target_value`

For now, `forecast_timestamp` is set to `target_release_date`. Real point-in-time feature joins are not implemented yet and will come in a later stage.

## First Target

The initial forecasting target will be U.S. headline CPI month-over-month inflation.

Future target ideas include Core CPI, PCE, NFP, unemployment, ISM, FOMC/rates decisions, and Treasury yield reactions.

## Getting Started

Install dependencies with `uv`:

```powershell
uv sync
```

Run tests:

```powershell
uv run pytest
```

Run linting:

```powershell
uv run ruff check .
```

Create a local environment file when needed:

```powershell
Copy-Item .env.example .env
```

Set your FRED API key in `.env`:

```text
FRED_API_KEY=your_key_here
```

Fetch and store CPI observations:

```powershell
uv run python -m macro_data_forecasting.cli fetch-fred --series-id CPIAUCSL --start 2010-01-01
```

Fetch and store BLS headline CPI observations:

```powershell
uv run python -m macro_data_forecasting.cli fetch-bls --series-id CUSR0000SA0 --start-year 2020 --end-year 2024
```

Fetch BLS CPI and map release dates from a local calendar:

```powershell
uv run python -m macro_data_forecasting.cli fetch-bls --series-id CUSR0000SA0 --start-year 2026 --end-year 2026 --release-calendar data/reference/cpi_release_calendar_sample.csv
```

Require complete CPI calendar coverage before storing:

```powershell
uv run python -m macro_data_forecasting.cli fetch-bls --series-id CUSR0000SA0 --start-year 2026 --end-year 2026 --release-calendar data/reference/cpi_release_calendar_sample.csv --strict-calendar
```

Validate a CPI release calendar:

```powershell
uv run python -m macro_data_forecasting.cli validate-cpi-calendar --calendar data/reference/cpi_release_calendar_sample.csv --start-period 2026-04 --end-period 2026-04
```

Running the same fetch command repeatedly should not create duplicate observations. The CLI reports rows seen, inserted, updated, skipped, and ingestion run status.

List recent ingestion runs:

```powershell
uv run python -m macro_data_forecasting.cli list-runs
```

Show one ingestion run:

```powershell
uv run python -m macro_data_forecasting.cli show-run --run-id 1
```

Summarize stored observation coverage:

```powershell
uv run python -m macro_data_forecasting.cli coverage
```

Build the CPI target dataset shell:

```powershell
uv run python -m macro_data_forecasting.cli build-cpi-target --series-id CUSR0000SA0 --output data/processed/cpi_target_shell.csv
```

The included `data/reference/cpi_release_calendar_sample.csv` is only for tests and examples. It is not a complete historical CPI release calendar and should not be used as the production source for point-in-time CPI backtests.

Do not commit `.env` or API keys.
