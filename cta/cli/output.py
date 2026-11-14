"""Console handles and report output shared by the CLI commands."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from rich import console as rich_console
import typer

if TYPE_CHECKING:
    from cta.core import models


console = rich_console.Console()


err_console = rich_console.Console(stderr=True)


FORMAT_HELP = (
    "Stdout format: 'rich' (tables) or 'json' (machine-readable, for piping "
    "to jq)."
)


def split_comma_separated(value: str | None) -> list[str] | None:
    """Splits a comma-separated option value into its non-empty items.

    Args:
        value: The raw option value, for example ``"a, b,,c"``.

    Returns:
        The stripped, non-empty items, or ``None`` when ``value`` is ``None``
        or empty.
    """
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def emit_report(
    report: models.ComparisonReport,
    output_format: str,
    out_dir: pathlib.Path | None,
    export: str,
) -> None:
    """Prints a report to stdout and optionally exports it to files.

    In ``json`` mode stdout carries only the JSON document, so the export
    status lines go to stderr.

    Args:
        report: The report to print and export.
        output_format: Stdout format, ``"rich"`` (tables) or ``"json"``.
        out_dir: Export directory; ``None`` skips the export.
        export: Comma-separated export formats.

    Raises:
        typer.BadParameter: If ``output_format`` is neither ``"rich"`` nor
            ``"json"``.
    """
    from cta.exporters import tabular
    from cta.exporters import terminal

    if output_format not in ("rich", "json"):
        raise typer.BadParameter(
            f"Unknown --format '{output_format}' (expected 'rich' or 'json')."
        )
    status_console = err_console if output_format == "json" else console
    if output_format == "json":
        typer.echo(tabular.report_to_json(report))
    else:
        terminal.render_terminal_report(report, console=console)
    if out_dir is not None:
        paths_by_format = report.export(
            out_dir, formats=split_comma_separated(export) or []
        )
        for export_format, export_path in paths_by_format.items():
            status_console.print(
                f"[green]✓ Exported {export_format}:[/green] {export_path}"
            )
