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
- Stage 2: Data ingestion
- Stage 3: Point-in-time feature engineering
- Stage 4: Modeling and walk-forward validation
- Stage 5: Automated reporting

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

Do not commit `.env` or API keys.
