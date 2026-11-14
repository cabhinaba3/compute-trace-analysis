"""The ``cta`` command-line application: wires every command group together."""

from __future__ import annotations

from typing import Annotated

import typer

import cta
from cta.cli import cache_commands
from cta.cli import compare_commands
from cta.cli import plot_command
from cta.cli import profile_commands
from cta.cli import trace_commands

# Heavy modules (polars, scipy, profilers) are imported inside each command
# so that `cta --help` and `cta --version` stay fast.
app = typer.Typer(
    name="cta",
    help=(
        "CTA v1.0: Multi-trace & multi-aspect performance comparison framework."
    ),
    no_args_is_help=True,
)


app.add_typer(cache_commands.cache_app, name="cache")


app.add_typer(profile_commands.profile_app, name="profile")


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


app.command("compare")(compare_commands.compare_command)
app.command("verify-equivalence")(compare_commands.verify_equivalence_command)
app.command("analyze")(compare_commands.analyze_command)
app.command("plot")(plot_command.plot_command)
app.command("synth")(trace_commands.synth_command)
app.command("inspect")(trace_commands.inspect_command)
app.command("list")(trace_commands.list_command)


def main() -> None:
    """Runs the ``cta`` command-line application."""
    app()


if __name__ == "__main__":
    main()
