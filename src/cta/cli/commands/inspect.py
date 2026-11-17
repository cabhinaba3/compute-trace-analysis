"""The ``inspect`` command: shows what CTA finds in a trace or directory."""

from __future__ import annotations

import pathlib
from typing import Annotated

from rich import pretty
import typer

from cta.cli import output


def inspect_command(
    source: Annotated[
        str, typer.Argument(help="Trace file, glob, or RunBundle directory.")
    ],
    cache_dir: Annotated[
        pathlib.Path, typer.Option("--cache-dir")
    ] = pathlib.Path(".cta_cache"),
) -> None:
    """Shows the detected formats, the world size and the table schemas."""
    from cta import api

    info = api.inspect_trace(source, cache_dir=cache_dir)
    output.console.print(pretty.Pretty(info))


def register(app: typer.Typer) -> None:
    """Adds ``cta inspect`` to the application."""
    app.command("inspect")(inspect_command)
