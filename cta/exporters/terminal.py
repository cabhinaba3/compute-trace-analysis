"""Prints a comparison report to the terminal with Rich."""

from __future__ import annotations

from typing import Any

from rich import console as rich_console
from rich import panel
from rich import table

from cta.core import models

# Summary columns left out of the terminal tables (besides every ci95_* one).
_HIDDEN_COLUMNS = ("p_mannwhitney", "cohens_d")


def _delta_color(delta_pct: float, speedup: Any) -> str:
    """Picks the Rich color of a ``*delta_pct`` cell.

    A float ``speedup`` (pair comparisons) is direction-aware: above 1
    means the candidate is better. Otherwise the change itself decides:
    more than +2% is red, less than -2% green.
    """
    if isinstance(speedup, float):
        if speedup > 1.02:
            return "green"
        return "red" if speedup < 0.98 else "white"
    if delta_pct > 2.0:
        return "red"
    return "green" if delta_pct < -2.0 else "white"


def _format_cell(column: str, value: Any, row: dict[str, Any]) -> str:
    """Formats one summary value as Rich table cell markup."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if column.endswith("delta_pct"):
            color = _delta_color(value, row.get("speedup"))
            return f"[{color}]{value:+.2f}%[/{color}]"
        if column == "speedup":
            return f"{value:.2f}x"
        return f"{value:.4g}"
    return str(value)


def render_terminal_report(
    report: models.ComparisonReport, console: rich_console.Console | None = None
) -> None:
    """Prints a report to the terminal.

    Prints a title panel, one line per bottleneck diagnosis (yellow when
    the regime shifted), one table per aspect with a non-empty summary
    (without CI, Mann-Whitney and Cohen's d columns) and the gate
    violations in red.

    Args:
        report: The report to print.
        console: Rich console to print to; None creates a new one.
    """
    console = console or rich_console.Console()
    group_labels = " vs ".join(
        f"[bold cyan]{group.label}[/]" for group in report.groups
    )
    title_kind = (
        "Analysis"
        if report.mode == "single"
        else f"Comparison ({report.mode.upper()})"
    )
    console.print(
        panel.Panel(f"CTA {title_kind}: {group_labels}", expand=False)
    )

    for shift in report.diagnoses:
        color = (
            "yellow"
            if shift.baseline_regime != shift.candidate_regime
            else "green"
        )
        console.print(f"[{color}]• {shift.summary}[/{color}]")

    for aspect in report.aspects.values():
        summary = aspect.summary
        if summary.is_empty():
            continue
        summary_table = table.Table(
            title=f"{aspect.title} [{aspect.name}]", show_lines=False
        )
        visible_columns = [
            column
            for column in summary.columns
            if not column.startswith("ci95_") and column not in _HIDDEN_COLUMNS
        ]
        for column in visible_columns:
            summary_table.add_column(
                column,
                justify="left" if column in ("metric", "unit") else "right",
            )
        for row in summary.select(visible_columns).iter_rows(named=True):
            cells = [
                _format_cell(column, value, row)
                for column, value in row.items()
            ]
            summary_table.add_row(*cells)
        console.print(summary_table)

    for violation in report.violations:
        console.print(f"[bold red]✗ {violation.message}[/bold red]")
