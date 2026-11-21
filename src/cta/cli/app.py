"""The ``cta`` command-line application: wires every command together.

Each module in ``cta.cli.commands`` defines one command (or command group)
and a ``register(app)`` function; ``COMMANDS`` lists them in ``--help``
order. Heavy modules (polars, scipy, the profilers) are imported inside the
commands, so ``cta --help`` and ``cta --version`` stay fast.
"""

from __future__ import annotations

from typing import Annotated

import typer

import cta
from cta.cli.commands import analyze
from cta.cli.commands import cache
from cta.cli.commands import catalog
from cta.cli.commands import compare
from cta.cli.commands import inspect
from cta.cli.commands import plot
from cta.cli.commands import profile
from cta.cli.commands import simulate
from cta.cli.commands import synth
from cta.cli.commands import verify
from cta.cli.commands import whatif

COMMANDS = (
    cache,
    profile,
    compare,
    verify,
    simulate,
    analyze,
    plot,
    synth,
    inspect,
    catalog,
    whatif,
)

app = typer.Typer(
    name="cta",
    help="CTA: multi-trace, multi-aspect performance comparison.",
    no_args_is_help=True,
)


def _print_version_and_exit(value: bool) -> None:
    """Handles ``--version``: prints the package version and exits."""
    if value:
        typer.echo(f"cta {cta.__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version_and_exit,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """Declares the root-level options of ``cta`` (only ``--version``)."""


for command in COMMANDS:
    command.register(app)


def main() -> None:
    """Runs the ``cta`` command-line application."""
    app()


if __name__ == "__main__":
    main()
