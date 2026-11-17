"""Console handles and report output shared by the CLI commands."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from rich import console as rich_console
from rich import markup as rich_markup
import typer

if TYPE_CHECKING:
    from cta import models


console = rich_console.Console()


err_console = rich_console.Console(stderr=True)


def print_error(message: object) -> None:
    """Prints an error on stderr.

    The text is printed as given: Rich would otherwise read a bracket such as
    the ``[sql]`` in an install hint as a style tag and drop it.

    Args:
        message: The error text or exception.
    """
    err_console.print(
        f"[bold red]error:[/bold red] {rich_markup.escape(str(message))}"
    )


def print_warning(message: object) -> None:
    """Prints a warning on stderr, as ``print_error`` prints an error.

    Args:
        message: The warning text.
    """
    err_console.print(
        f"[yellow]warning:[/yellow] {rich_markup.escape(str(message))}"
    )


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
    report: models.Report,
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
    from cta.export import tabular
    from cta.export import terminal

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
