"""Tests for markdown report generation."""

import pandas as pd

from macro_data_forecasting.reports.generate_report import (
    dataframe_to_markdown_table,
    generate_model_comparison_report,
)


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["naive_last_value", "ridge", "lightgbm"],
            "n_forecasts": [3, 3, 3],
            "rmse": [1.0, 1.2, 0.8],
            "mae": [0.8, 0.9, 0.6],
            "directional_accuracy": [0.5, 0.5, 1.0],
            "beats_naive_rmse": [0.0, 0.0, 1.0],
            "beats_naive_mae": [0.0, 0.0, 1.0],
            "rmse_vs_naive": [0.0, -0.2, 0.2],
            "mae_vs_naive": [0.0, -0.1, 0.2],
        },
    )


def _forecasts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, predictions in {
        "naive_last_value": [0.9, 1.8, 2.8],
        "ridge": [1.2, 2.3, 3.5],
        "lightgbm": [1.0, 2.0, 3.0],
    }.items():
        for index, prediction in enumerate(predictions):
            rows.append(
                {
                    "forecast_timestamp": pd.Timestamp("2026-01-01")
                    + pd.DateOffset(months=index),
                    "target_id": "cpi_mom",
                    "target_reference_date": pd.Timestamp("2025-12-01")
                    + pd.DateOffset(months=index),
                    "target_release_date": pd.Timestamp("2026-01-01")
                    + pd.DateOffset(months=index),
                    "actual": float(index + 1),
                    "prediction": prediction,
                    "model_name": model_name,
                    "fold_number": index + 1,
                },
            )
    return pd.DataFrame(rows)


def test_generate_model_comparison_report_writes_markdown_file(tmp_path) -> None:
    """Verify model comparison report writes to disk."""
    output_path = tmp_path / "baseline_report.md"

    report_path = generate_model_comparison_report(
        _metrics(),
        _forecasts(),
        output_path=output_path,
    )

    assert report_path == output_path
    assert output_path.exists()


def test_report_contains_title_and_expected_sections(tmp_path) -> None:
    """Verify report includes the main sections."""
    output_path = tmp_path / "baseline_report.md"

    generate_model_comparison_report(
        _metrics(),
        _forecasts(),
        output_path=output_path,
        title="CPI Baseline Report",
    )

    content = output_path.read_text(encoding="utf-8")
    assert "# CPI Baseline Report" in content
    assert "## Model Comparison" in content
    assert "## Naive Benchmark Interpretation" in content
    assert "## Methodology" in content
    assert "## Limitations" in content


def test_report_plainly_says_when_ridge_does_not_beat_naive(tmp_path) -> None:
    """Verify losing models are described plainly."""
    output_path = tmp_path / "baseline_report.md"

    generate_model_comparison_report(_metrics(), _forecasts(), output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "ridge does not beat the naive baseline on RMSE." in content
    assert "ridge does not beat the naive baseline on MAE." in content


def test_report_plainly_says_when_lightgbm_beats_naive(tmp_path) -> None:
    """Verify winning models are described only when metrics indicate it."""
    output_path = tmp_path / "baseline_report.md"

    generate_model_comparison_report(_metrics(), _forecasts(), output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "lightgbm beats the naive baseline on RMSE." in content
    assert "lightgbm beats the naive baseline on MAE." in content


def test_report_includes_notes_when_provided(tmp_path) -> None:
    """Verify optional notes are included."""
    output_path = tmp_path / "baseline_report.md"

    generate_model_comparison_report(
        _metrics(),
        _forecasts(),
        output_path,
        notes="Research note.",
    )

    content = output_path.read_text(encoding="utf-8")
    assert "## Notes" in content
    assert "Research note." in content


def test_dataframe_to_markdown_table_works_without_extra_dependencies() -> None:
    """Verify fallback markdown table formatting."""
    table = dataframe_to_markdown_table(
        pd.DataFrame({"model_name": ["ridge"], "rmse": [1.23456789]}),
    )

    assert "| model_name | rmse |" in table
    assert "| ridge | 1.23457 |" in table
