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
- Stage 2B: Additional data ingestion
- Stage 3: Point-in-time feature engineering
- Stage 4: Modeling and walk-forward validation
- Stage 5: Automated reporting

## Ingestion Status

Stage 2A adds the first real ingestion path for FRED/ALFRED series. The current client fetches FRED `series/observations`, normalizes observations, validates the point-in-time columns, and stores rows in the local SQLAlchemy database.

No BLS, Treasury, market-data, feature-engineering, or modeling logic is implemented yet.

## Point-In-Time Columns

Every stored observation distinguishes:

- `date`: the economic or reference date for the observation.
- `release_date`: the best available date when the value became observable.
- `fetched_at`: the timestamp when this system fetched the value.

For FRED/ALFRED, the observations endpoint provides realtime vintage metadata. Until exact release calendars are integrated, this project uses each observation's `realtime_start` vintage date as the best available `release_date` approximation. This preserves vintage awareness but should not be treated as a perfect official release timestamp for every series.

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

Do not commit `.env` or API keys.
