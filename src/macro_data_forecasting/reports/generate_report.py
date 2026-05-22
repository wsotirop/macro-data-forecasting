"""Markdown reporting for model comparison outputs."""

from datetime import UTC, datetime
from os import path as os_path
from pathlib import Path

import pandas as pd

from macro_data_forecasting.config import get_settings

METRICS_REPORT_COLUMNS = [
    "model_name",
    "n_forecasts",
    "rmse",
    "mae",
    "directional_accuracy",
    "beats_naive_rmse",
    "beats_naive_mae",
    "rmse_vs_naive",
    "mae_vs_naive",
]


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Return a GitHub-flavored markdown table without extra dependencies."""
    if df.empty:
        return "_No rows._"

    headers = [str(column) for column in df.columns]
    rows = [
        [_format_markdown_value(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    header_line = (
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in headers) + " |"
    )
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *row_lines])


def generate_model_comparison_report(
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    output_path: str | Path,
    title: str = "Macro Data Forecasting Report",
    dataset_path: str | Path | None = None,
    notes: str | None = None,
    plot_paths: dict[str, str | Path] | None = None,
) -> Path:
    """Generate a markdown report from model metrics and forecasts."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_metrics = _normalize_metrics(metrics)
    normalized_forecasts = _normalize_forecasts(forecasts)
    sections = [
        f"# {title}",
        "## Summary",
        _build_summary_section(
            normalized_metrics,
            normalized_forecasts,
            dataset_path=dataset_path,
        ),
        "## Model Comparison",
        dataframe_to_markdown_table(normalized_metrics.loc[:, METRICS_REPORT_COLUMNS]),
        "## Naive Benchmark Interpretation",
        _build_naive_interpretation(normalized_metrics),
    ]
    if plot_paths:
        sections.extend(["## Plots", _build_plots_section(plot_paths, report_path)])
    sections.extend(
        [
            "## Forecast Output Summary",
            _build_forecast_summary(normalized_forecasts),
            "## Methodology",
            _build_methodology_section(),
            "## Limitations",
            _build_limitations_section(),
        ],
    )
    if notes:
        sections.extend(["## Notes", str(notes)])

    report_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return report_path


def generate_markdown_report(
    title: str = "Macro Data Forecasting Report",
    output_path: Path | None = None,
) -> Path:
    """Write a simple markdown report placeholder and return its path."""
    settings = get_settings()
    report_path = output_path or settings.reports_dir / "stage_1_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# {title}\n\n"
        "This placeholder report confirms the reporting module is installed.\n",
        encoding="utf-8",
    )
    return report_path


def _normalize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in METRICS_REPORT_COLUMNS if column not in metrics]
    if missing:
        msg = f"Metrics missing required report columns: {missing}"
        raise ValueError(msg)
    normalized = metrics.copy()
    for column in METRICS_REPORT_COLUMNS:
        if column != "model_name":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def _normalize_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    required = ["forecast_timestamp", "actual", "prediction", "model_name"]
    missing = [column for column in required if column not in forecasts]
    if missing:
        msg = f"Forecasts missing required report columns: {missing}"
        raise ValueError(msg)

    normalized = forecasts.copy()
    normalized["forecast_timestamp"] = pd.to_datetime(
        normalized["forecast_timestamp"],
        errors="coerce",
    )
    normalized["actual"] = pd.to_numeric(normalized["actual"], errors="coerce")
    normalized["prediction"] = pd.to_numeric(
        normalized["prediction"],
        errors="coerce",
    )
    return normalized


def _build_summary_section(
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    dataset_path: str | Path | None,
) -> str:
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        f"- Generated timestamp: {generated_at}",
        f"- Number of models: {metrics['model_name'].nunique()}",
        f"- Number of forecasts: {len(forecasts)}",
    ]
    if dataset_path is not None:
        lines.insert(1, f"- Dataset path: {dataset_path}")
    if "target_id" in forecasts and not forecasts["target_id"].dropna().empty:
        target_ids = sorted(
            str(value) for value in forecasts["target_id"].dropna().unique()
        )
        lines.append(f"- Target id: {', '.join(target_ids)}")
    valid_timestamps = forecasts["forecast_timestamp"].dropna()
    if not valid_timestamps.empty:
        lines.append(
            "- Forecast date range: "
            f"{valid_timestamps.min().date().isoformat()} to "
            f"{valid_timestamps.max().date().isoformat()}",
        )
    return "\n".join(lines)


def _build_naive_interpretation(metrics: pd.DataFrame) -> str:
    benchmarked = metrics.loc[metrics["model_name"] != "naive_last_value"]
    if benchmarked.empty:
        return "No non-naive models were included for benchmark interpretation."

    lines: list[str] = []
    for row in benchmarked.itertuples(index=False):
        model_name = str(row.model_name)
        rmse_verb = "beats" if row.beats_naive_rmse == 1.0 else "does not beat"
        mae_verb = "beats" if row.beats_naive_mae == 1.0 else "does not beat"
        lines.append(f"- {model_name} {rmse_verb} the naive baseline on RMSE.")
        lines.append(f"- {model_name} {mae_verb} the naive baseline on MAE.")
    return "\n".join(lines)


def _build_plots_section(
    plot_paths: dict[str, str | Path],
    report_path: Path,
) -> str:
    labels = {
        "predictions_vs_actuals": "Predictions vs Actuals",
        "forecast_errors": "Forecast Errors",
        "rmse_comparison": "RMSE Comparison",
        "mae_comparison": "MAE Comparison",
    }
    lines: list[str] = []
    for plot_name, plot_path in plot_paths.items():
        label = labels.get(plot_name, plot_name.replace("_", " ").title())
        reference = _format_plot_reference(Path(plot_path), report_path)
        lines.append(f"![{label}]({reference})")
    return "\n\n".join(lines)


def _format_plot_reference(plot_path: Path, report_path: Path) -> str:
    try:
        relative_path = os_path.relpath(
            plot_path.resolve(),
            start=report_path.parent.resolve(),
        )
        return Path(relative_path).as_posix()
    except ValueError:
        return plot_path.as_posix()


def _build_forecast_summary(forecasts: pd.DataFrame) -> str:
    valid_timestamps = forecasts["forecast_timestamp"].dropna()
    lines: list[str] = []
    if valid_timestamps.empty:
        lines.append("- Forecast timestamp range: unavailable")
    else:
        lines.append(
            "- Forecast timestamp range: "
            f"{valid_timestamps.min().date().isoformat()} to "
            f"{valid_timestamps.max().date().isoformat()}",
        )

    count_by_model = (
        forecasts.groupby("model_name", dropna=False)
        .size()
        .reset_index(name="forecast_count")
    )
    lines.extend(
        ["", "Forecast count by model:", dataframe_to_markdown_table(count_by_model)],
    )

    actual_summary = pd.DataFrame(
        {
            "statistic": ["mean", "std"],
            "actual": [
                forecasts["actual"].mean(),
                forecasts["actual"].std(),
            ],
        },
    )
    prediction_summary = (
        forecasts.groupby("model_name", dropna=False)["prediction"]
        .agg(prediction_mean="mean", prediction_std="std")
        .reset_index()
    )
    lines.extend(
        [
            "",
            "Actual summary:",
            dataframe_to_markdown_table(actual_summary),
            "",
            "Prediction summary by model:",
            dataframe_to_markdown_table(prediction_summary),
        ],
    )
    return "\n".join(lines)


def _build_methodology_section() -> str:
    return "\n".join(
        [
            "- Validation uses expanding-window walk-forward validation only.",
            "- Each fold trains on rows strictly before the forecast row.",
            "- No k-fold validation is used for macro time-series forecasting.",
            "- Point-in-time features must satisfy "
            "`release_date <= forecast_timestamp`.",
            "- `naive_last_value` predicts the prior known target value.",
            "- `ridge` uses median imputation, standard scaling, and ridge regression.",
            "- `lightgbm` uses median imputation and a fixed-default "
            "LightGBM regressor.",
        ],
    )


def _build_limitations_section() -> str:
    return "\n".join(
        [
            "- Data quality depends on stored observations and release dates.",
            "- CPI point-in-time work requires a complete official CPI "
            "release calendar.",
            "- LightGBM is a fixed-default baseline and is not tuned.",
            "- The current feature matrix uses latest available values only.",
            "- Results may not beat the naive baseline.",
        ],
    )


def _format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
