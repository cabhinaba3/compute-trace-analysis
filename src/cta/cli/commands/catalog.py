"""The ``list`` command: lists what the four plugin registries hold."""

from __future__ import annotations

from rich import table
import typer

from cta.cli import output


def list_command() -> None:
    """Lists the registered trace formats, aspects, export formats and plots.

    Plugins are included.
    """
    from cta import _device
    from cta import plugins
    from cta.export import writers
    from cta.viz import registry as viz_registry

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
    export_table = table.Table(title="Export formats (--export)")
    export_table.add_column("format")
    for name in sorted(writers.WRITERS):
        export_table.add_row(name)
    output.console.print(export_table)
    devices_table = table.Table(title="Devices (--device)")
    devices_table.add_column("device")
    devices_table.add_column("available")
    devices_table.add_row("cpu", "always")
    for name in ("cuda", "metal"):
        devices_table.add_row(
            name, "yes" if name in _device.backends() else "no"
        )
    output.console.print(devices_table)
    plots_table = table.Table(title="Plot kinds (cta plot --kind)")
    for column in ("kind", "group", "title"):
        plots_table.add_column(column)
    for kind in viz_registry.kind_catalog():
        plots_table.add_row(kind.name, kind.group, kind.title)
    output.console.print(plots_table)


def register(app: typer.Typer) -> None:
    """Adds ``cta list`` to the application."""
    app.command("list")(list_command)
