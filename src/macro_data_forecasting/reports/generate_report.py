"""Markdown report generation placeholder."""

from pathlib import Path

from macro_data_forecasting.config import get_settings


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
        "This placeholder report confirms the Stage 1 scaffold is installed.\n",
        encoding="utf-8",
    )
    return report_path
