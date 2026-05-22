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
- Stage 2F: FRED/ALFRED initial-release vintage mode
- Stage 2G: Chunked FRED initial-release ingestion
- Stage 2H: Robust final window for chunked FRED ingestion
- Stage 3A: CPI target construction and dataset contract
- Stage 3B: Point-in-time feature matrix construction
- Stage 4A: Walk-forward validation with naive and ridge baselines
- Stage 4B: Model comparison runner and metrics table
- Stage 4C: Fixed-default LightGBM baseline
- Stage 5A: Automated markdown research reporting
- Stage 5B: Lightweight report plots
- Stage 3: Point-in-time feature engineering
- Stage 4: Modeling and walk-forward validation
- Stage 5: Automated reporting

## Ingestion Status

Stage 2A adds the first real ingestion path for FRED/ALFRED series. The current client fetches FRED `series/observations`, normalizes observations, validates the point-in-time columns, and stores rows in the local SQLAlchemy database.

Stage 2B adds BLS CPI/Core CPI ingestion for monthly CPI series and a separate CPI release-calendar mapping utility. BLS observations can be fetched without an API key, but exact `release_date` values require a local official release calendar CSV.

Stage 2C adds CPI release-calendar validation and coverage checks. Calendars are checked for schema, duplicate reference months, valid release dates, valid release times, and complete coverage over requested CPI observation periods.

Stage 2D adds idempotent upsert behavior and ingestion-run metadata. Re-running the same ingestion is safe: existing observations are matched by `series_id`, `source`, `date`, and `release_date`, then inserted, updated, or skipped without creating duplicate point-in-time rows.

Stage 2E adds audit commands for inspecting ingestion history and stored observation coverage from the command line.

Stage 2F adds explicit FRED/ALFRED vintage modes, including `initial_release`, which uses FRED `output_type=4` so historical rows use per-observation initial-release vintage dates when FRED provides them.

Stage 2G adds chunked FRED initial-release requests for high-vintage daily series that can exceed FRED's 2,000-vintage-date API limit over a full real-time window.

Stage 2H makes the final chunk robust by using FRED's allowed max real-time date `9999-12-31` for open-ended chunked requests instead of the local system date.

Stage 3A adds the first target-construction contract: U.S. headline CPI month-over-month inflation. It builds a target shell from normalized CPI observations, validates target release dates, and creates an empty model-dataset shell.

Stage 3B adds the first simple point-in-time feature matrix. Feature values are selected using `release_date <= forecast_timestamp`, never by reference date alone.

Stage 4A adds walk-forward validation plus naive last-value and ridge regression baselines.

Stage 4B adds a model comparison runner that evaluates multiple existing baselines on the same feature matrix, saves forecast-level outputs, and produces a metrics table benchmarked against naive.

Stage 4C adds a fixed-default LightGBM baseline to the same walk-forward validation and model comparison framework.

Stage 5A adds automated markdown research reporting from saved model comparison metrics and forecast outputs.

Stage 5B adds lightweight matplotlib plots for predictions vs actuals, forecast errors, and RMSE/MAE metric comparisons. Plots are generated from saved forecast and metric outputs; they do not rerun ingestion.

No Treasury, market-data, hyperparameter tuning, notebooks, or external report publishing logic is implemented yet.

## Point-In-Time Columns

Every stored observation distinguishes:

- `date`: the economic or reference date for the observation.
- `release_date`: the best available date when the value became observable.
- `fetched_at`: the timestamp when this system fetched the value.

For FRED/ALFRED, the observations endpoint provides realtime vintage metadata. The CLI supports three FRED vintage modes:

- `current`: the default compatibility mode. It fetches the current vintage snapshot. This can make all historical observations appear available on the same current vintage date, so it is not suitable for strict historical point-in-time feature construction.
- `initial_release`: uses FRED `output_type=4` and maps each observation's `realtime_start` to `release_date`. FRED defaults `realtime_start` and `realtime_end` to today's date when they are omitted, so this mode uses the full real-time window `1776-07-04` to `9999-12-31` by default. This is recommended for strict historical point-in-time macro features when FRED provides the required metadata.
- `realtime_period`: uses FRED `output_type=1` with user-provided realtime bounds for explicit realtime-period requests.

FRED `release_date` is still vintage metadata, not a complete official release calendar for every series. The project no longer treats a current-vintage snapshot as historical release timing unless current mode is explicitly used.

Daily or high-frequency FRED series can exceed the API's 2,000 vintage-date limit over a full initial-release real-time window. Use `--chunk-realtime` to split the real-time window into deterministic year-based chunks. Chunking is only supported with `--vintage-mode initial_release`; it still passes `observation_start` and `observation_end` through to every FRED request.

FRED's server date can lag the local system date. For default open-ended chunked initial-release requests, the final chunk uses `realtime_end=9999-12-31` because FRED explicitly allows the real-time max date. If a specific historical cutoff is needed, pass `--realtime-end` explicitly.

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

## Point-In-Time Features

The first feature matrix is intentionally simple. For each target row and requested feature series, the builder selects the latest observation whose `release_date` is less than or equal to the target `forecast_timestamp`. If an observation has a reference date before the target but was released after the forecast timestamp, it is excluded.

Feature columns are named:

```text
feature_{series_id}_latest
```

Optional lagged target features can be added as `feature_target_lag_1`, etc. These lags are computed from prior target rows only and do not use the current target value.

Feature diagnostics summarize missing values by feature, including first and last valid forecast timestamps. During walk-forward validation, ridge and LightGBM drop features that are entirely missing in the current training fold, without removing those columns from the saved dataset.

## Walk-Forward Validation

Macro model validation uses expanding-window walk-forward validation only. No k-fold cross-validation is used because it would mix future and past observations in a way that can create lookahead bias.

For each forecast row, the validator trains only on rows strictly before that row. Stage 4A includes:

- `naive_last_value`: predicts the previous known `target_value`.
- `ridge`: fits a scikit-learn pipeline with median imputation, standard scaling, and ridge regression.
- `lightgbm`: fits a fixed-default LightGBM regressor with median imputation.

The LightGBM baseline uses conservative fixed defaults and no hyperparameter tuning. If ridge or LightGBM does not beat the naive baseline, the CLI reports that plainly.

## Model Comparison

The comparison runner executes requested models through the same walk-forward validator and summarizes each model in a metrics table. `naive_last_value` is always used as the benchmark, even when it is not included in the requested output models.

Metrics include RMSE, MAE, directional accuracy, forecast count, RMSE/MAE improvement versus naive, and numeric flags for whether a model beats naive. Forecast comparisons are aligned by forecast timestamp, target id, and target reference date before benchmark metrics are computed.

LightGBM now plugs into the comparison framework as a fixed-default baseline.

## Automated Reports

Stage 5A generates reproducible markdown reports from model comparison metrics and forecast-level outputs. Reports include a summary, model metrics table, naive benchmark interpretation, forecast-output summary, methodology, limitations, and optional notes.

Stage 5B can add lightweight matplotlib plots to those reports:

- predictions vs actuals
- forecast errors
- RMSE comparison
- MAE comparison

Reports should state plainly when a model does not beat the naive baseline. They should not claim ridge or LightGBM beats naive unless the saved metrics show it.

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

Fetch and store FRED initial-release observations for point-in-time features:

```powershell
uv run python -m macro_data_forecasting.cli fetch-fred --series-id UNRATE --start 2010-01-01 --vintage-mode initial_release
```

You can override the initial-release real-time window when needed:

```powershell
uv run python -m macro_data_forecasting.cli fetch-fred --series-id UNRATE --start 2010-01-01 --vintage-mode initial_release --realtime-start 2018-01-01 --realtime-end 2020-01-01
```

Fetch high-vintage daily FRED series with chunked initial-release requests:

```powershell
uv run python -m macro_data_forecasting.cli fetch-fred --series-id DGS2 --start 2010-01-01 --vintage-mode initial_release --chunk-realtime --realtime-chunk-years 5
```

```powershell
uv run python -m macro_data_forecasting.cli fetch-fred --series-id DGS10 --start 2010-01-01 --vintage-mode initial_release --chunk-realtime --realtime-chunk-years 5
```

```powershell
uv run python -m macro_data_forecasting.cli fetch-fred --series-id T10Y2Y --start 2010-01-01 --vintage-mode initial_release --chunk-realtime --realtime-chunk-years 5
```

Fetch a FRED realtime period explicitly:

```powershell
uv run python -m macro_data_forecasting.cli fetch-fred --series-id UNRATE --start 2010-01-01 --vintage-mode realtime_period --realtime-start 2018-01-01 --realtime-end 2020-01-01
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

Build a simple point-in-time feature matrix:

```powershell
uv run python -m macro_data_forecasting.cli build-feature-matrix --target-series-id CUSR0000SA0 --features UNRATE FEDFUNDS DGS2 DGS10 T10Y2Y --output data/processed/cpi_feature_matrix.csv
```

Inspect feature missingness:

```powershell
uv run python -m macro_data_forecasting.cli feature-diagnostics --dataset data/processed/cpi_feature_matrix.csv --output reports/feature_missingness.csv
```

Run walk-forward validation:

```powershell
uv run python -m macro_data_forecasting.cli validate-model --dataset data/processed/cpi_feature_matrix.csv --model ridge --output reports/ridge_forecasts.csv
```

Run LightGBM walk-forward validation:

```powershell
uv run python -m macro_data_forecasting.cli validate-model --dataset data/processed/cpi_feature_matrix.csv --model lightgbm --output reports/lightgbm_forecasts.csv
```

Compare baseline models:

```powershell
uv run python -m macro_data_forecasting.cli compare-models --dataset data/processed/cpi_feature_matrix.csv --models naive_last_value ridge --output-dir reports
```

Compare all current baselines:

```powershell
uv run python -m macro_data_forecasting.cli compare-models --dataset data/processed/cpi_feature_matrix.csv --models naive_last_value ridge lightgbm --output-dir reports
```

Generate a markdown report from saved comparison outputs:

```powershell
uv run python -m macro_data_forecasting.cli generate-report --metrics reports/model_comparison_metrics.csv --forecasts reports/model_comparison_forecasts.csv --output reports/baseline_report.md
```

Generate a markdown report with plots from saved comparison outputs:

```powershell
uv run python -m macro_data_forecasting.cli generate-report --metrics reports/model_comparison_metrics.csv --forecasts reports/model_comparison_forecasts.csv --output reports/baseline_report.md --include-plots --plots-dir reports/plots
```

Run comparison and generate a report in one step:

```powershell
uv run python -m macro_data_forecasting.cli compare-models --dataset data/processed/cpi_feature_matrix.csv --models naive_last_value ridge lightgbm --output-dir reports --report reports/baseline_report.md
```

Run comparison and generate a plot-backed report in one step:

```powershell
uv run python -m macro_data_forecasting.cli compare-models --dataset data/processed/cpi_feature_matrix.csv --models naive_last_value ridge lightgbm --output-dir reports --report reports/baseline_report.md --include-plots --plots-dir reports/plots
```

The included `data/reference/cpi_release_calendar_sample.csv` is only for tests and examples. It is not a complete historical CPI release calendar and should not be used as the production source for point-in-time CPI backtests.

Do not commit `.env` or API keys.
