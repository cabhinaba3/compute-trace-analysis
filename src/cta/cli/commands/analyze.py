"""The ``analyze`` command: analyses one trace, glob or run bundle."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from cta.cli import output


def analyze_command(
    trace: Annotated[
        str,
        typer.Argument(help="Single trace file, glob, or RunBundle directory."),
    ],
    aspects: Annotated[
        str | None,
        typer.Option("--aspects", "-a", help="Comma-separated aspect names."),
    ] = None,
    warmup_steps: Annotated[
        int,
        typer.Option(
            "--warmup-steps", help="Skip initial warmup steps/iterations."
        ),
    ] = 0,
    max_steps: Annotated[
        int | None,
        typer.Option("--max-steps", help="Max steps/iterations to analyze."),
    ] = None,
    collective_algorithm: Annotated[
        str,
        typer.Option(
            "--collective-algorithm",
            help="Collective decomposition model (ring, tree, direct).",
        ),
    ] = "ring",
    alpha: Annotated[
        float,
        typer.Option(
            "--alpha", help="Significance level for statistical tests."
        ),
    ] = 0.05,
    scope_depth: Annotated[
        int | None,
        typer.Option(
            "--scope-depth",
            min=1,
            help=(
                "Group kernels by the N-th enclosing op/range from the "
                "outermost (default: innermost)."
            ),
        ),
    ] = None,
    ranks: Annotated[
        str | None,
        typer.Option(
            "--ranks", help="Analyse only these ranks, e.g. 0,1 (default: all)."
        ),
    ] = None,
    devices: Annotated[
        str | None,
        typer.Option(
            "--devices",
            help="Analyse only these GPU devices, e.g. 0 (default: all).",
        ),
    ] = None,
    label: Annotated[
        str | None,
        typer.Option(
            "--label", "-l", help="Display label for the trace/bundle."
        ),
    ] = None,
    out_dir: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--out-dir", "-o", help="Directory to export reports/plots."
        ),
    ] = None,
    export: Annotated[
        str,
        typer.Option(
            "--export",
            help=(
                "Comma-separated export formats "
                "(markdown,latex,csv,json,pdf,svg,png,html)."
            ),
        ),
    ] = "markdown,csv,json",
    output_format: Annotated[
        str, typer.Option("--format", help=output.FORMAT_HELP)
    ] = "rich",
    cache_dir: Annotated[
        pathlib.Path, typer.Option("--cache-dir", help="Cache directory.")
    ] = pathlib.Path(".cta_cache"),
) -> None:
    """Analyses one trace, a glob, or a directory holding a run bundle."""
    from cta import api

    output.check_export_formats(out_dir, export)
    try:
        report = api.analyze(
            trace,
            aspects=output.split_comma_separated(aspects),
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            collective_algorithm=collective_algorithm,
            alpha=alpha,
            scope_depth=scope_depth,
            ranks=output.split_comma_separated_ints(ranks),
            devices=output.split_comma_separated_ints(devices),
            cache_dir=cache_dir,
            label=label,
        )
    except (ValueError, FileNotFoundError) as exc:
        output.print_error(exc)
        raise typer.Exit(code=2) from exc
    output.emit_report(report, output_format, out_dir, export)


def register(app: typer.Typer) -> None:
    """Adds ``cta analyze`` to the application."""
    app.command("analyze")(analyze_command)
