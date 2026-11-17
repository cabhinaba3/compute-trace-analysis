"""The ``cta`` command-line application: wires every command group together."""

from __future__ import annotations

from typing import Annotated

import typer

import cta
from cta.cli.commands import cache
from cta.cli.commands import compare
from cta.cli.commands import plot
from cta.cli.commands import profile
from cta.cli.commands import traces

# Heavy modules (polars, scipy, profilers) are imported inside each command
# so that `cta --help` and `cta --version` stay fast.
app = typer.Typer(
    name="cta",
    help=(
        "CTA v1.0: Multi-trace & multi-aspect performance comparison framework."
    ),
    no_args_is_help=True,
)


app.add_typer(cache.cache_app, name="cache")


app.add_typer(profile.profile_app, name="profile")


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


app.command("compare")(compare.compare_command)
app.command("verify-equivalence")(compare.verify_equivalence_command)
app.command("analyze")(compare.analyze_command)
app.command("plot")(plot.plot_command)
app.command("synth")(traces.synth_command)
app.command("inspect")(traces.inspect_command)
app.command("list")(traces.list_command)


def main() -> None:
    """Runs the ``cta`` command-line application."""
    app()


if __name__ == "__main__":
    main()
