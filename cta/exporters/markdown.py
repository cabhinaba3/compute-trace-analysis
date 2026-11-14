"""Exports a comparison report as Markdown."""

from __future__ import annotations

import pathlib

from cta.core import models


def _format_cell(value: object) -> str:
    """Formats one summary value as Markdown table cell content."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def export_markdown(
    report: models.ComparisonReport, path: pathlib.Path | None = None
) -> str:
    """Renders a report as Markdown.

    The text has a title naming the groups, the bottleneck diagnoses and
    gate violations as bullet lists, and one table per aspect with a
    non-empty summary (without the ``ci95_*`` columns).

    Args:
        report: The report to export.
        path: File to write the Markdown to; its parent directories are
            created. None only returns the text.

    Returns:
        The Markdown text.
    """
    heading = (
        "CTA Trace Analysis Report"
        if report.mode == "single"
        else "CTA Comparison Report"
    )
    group_labels = " vs ".join(group.label for group in report.groups)
    lines = [f"# {heading} ({group_labels})", ""]
    if report.diagnoses:
        lines.append("## Bottleneck Diagnostics")
        for shift in report.diagnoses:
            lines.append(f"- {shift.summary}")
        lines.append("")
    if report.violations:
        lines.append("## Regression Gate Violations")
        for violation in report.violations:
            lines.append(
                f"- **{violation.metric}** ({violation.aspect}): "
                f"{violation.message}"
            )
        lines.append("")

    for aspect in report.aspects.values():
        summary = aspect.summary
        if summary.is_empty():
            continue
        lines.append(f"## {aspect.title} (`{aspect.name}`)")
        columns = [
            column
            for column in summary.columns
            if not column.startswith("ci95_")
        ]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in summary.select(columns).iter_rows():
            cells = [_format_cell(value) for value in row]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    content = "\n".join(lines)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return content
