"""The ``list`` command: lists the registered trace formats and aspects."""

from __future__ import annotations

from rich import table
import typer

from cta.cli import output


def list_command() -> None:
    """Lists the registered trace formats and comparison aspects.

    Plugins are included.
    """
    from cta import plugins

    formats_table = table.Table(title="Trace formats")
    formats_table.add_column("format")
    formats_table.add_column("adapter version")
    for name, adapter in sorted(plugins.ADAPTERS.items()):
        formats_table.add_row(name, adapter.version)
    output.console.print(formats_table)
    aspects_table = table.Table(title="Aspects")
    for column in ("name", "title", "requires"):
        aspects_table.add_column(column)
    for name, aspect in sorted(plugins.ASPECTS.items()):
        mode = "any of" if aspect.any_of_formats else "all of"
        aspects_table.add_row(
            name, aspect.title, f"{mode}: {', '.join(aspect.required_formats)}"
        )
    output.console.print(aspects_table)


def register(app: typer.Typer) -> None:
    """Adds ``cta list`` to the application."""
    app.command("list")(list_command)
