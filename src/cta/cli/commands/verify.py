"""The ``verify-equivalence`` command: checks a GOAL schedule against Chakra."""

from __future__ import annotations

import pathlib
from typing import Annotated

from rich import table
import typer

from cta.cli import output


def verify_equivalence_command(
    chakra_trace: Annotated[
        str, typer.Argument(help="Chakra-ET directory, glob, or .et file.")
    ],
    goal_schedule: Annotated[
        str, typer.Argument(help="GOAL .goal or .bin schedule file.")
    ],
    collective_algorithm: Annotated[
        str,
        typer.Option(
            "--collective-algorithm",
            help="Collective decomposition algorithm (ring, tree, direct).",
        ),
    ] = "ring",
    rel_tol: Annotated[
        float, typer.Option("--rel-tol", help="Relative error tolerance.")
    ] = 0.02,
    cache_dir: Annotated[
        pathlib.Path, typer.Option("--cache-dir")
    ] = pathlib.Path(".cta_cache"),
) -> None:
    """Checks that a Chakra ET and a GOAL trace describe the same run.

    Compares compute durations and the bytes of each decomposed collective.
    """
    from cta.aspects.distributed import equivalence

    verdict = equivalence.verify_chakra_goal_equivalence(
        chakra_trace,
        goal_schedule,
        collective_algorithm=collective_algorithm,
        rel_tol=rel_tol,
        cache_dir=cache_dir,
    )
    checks_table = table.Table(title="Chakra-ET <-> GOAL Equivalence Check")
    for column in ("check", "expected", "observed", "rel_error", "passed"):
        checks_table.add_column(
            column, justify="left" if column == "check" else "right"
        )
    for check in verdict.checks.iter_rows(named=True):
        status = (
            "[green]PASS[/green]"
            if check["passed"]
            else "[bold red]FAIL[/bold red]"
        )
        checks_table.add_row(
            str(check["check"]),
            f"{float(check['expected']):.4g}",
            f"{float(check['observed']):.4g}",
            f"{float(check['rel_error']) * 100:.2f}%",
            status,
        )
    output.console.print(checks_table)
    if not verdict.equivalent:
        raise typer.Exit(code=2)


def register(app: typer.Typer) -> None:
    """Adds ``cta verify-equivalence`` to the application."""
    app.command("verify-equivalence")(verify_equivalence_command)
