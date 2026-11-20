"""The ``compare`` command: compares a baseline with candidates, with gates."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from cta.cli import output


def compare_command(
    traces: Annotated[
        list[str] | None,
        typer.Argument(
            help="Two or more trace files, globs, or RunBundle directories."
        ),
    ] = None,
    manifest: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--manifest", "-m", help="Declarative YAML/TOML sweep manifest."
        ),
    ] = None,
    aspects: Annotated[
        str | None,
        typer.Option("--aspects", "-a", help="Comma-separated aspect names."),
    ] = None,
    warmup_steps: Annotated[
        int | None,
        typer.Option(
            "--warmup-steps",
            help="Skip initial warmup steps/iterations (overrides manifest).",
        ),
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option(
            "--max-steps",
            help="Max steps/iterations to analyze (overrides manifest).",
        ),
    ] = None,
    gate: Annotated[
        str | None,
        typer.Option(
            "--gate", "-g", help="Regression gates, e.g. 'p99_latency_ms:+5%'"
        ),
    ] = None,
    allow_missing_gates: Annotated[
        bool,
        typer.Option(
            "--allow-missing-gates",
            help="Do not fail when a gate metric is absent from the report.",
        ),
    ] = False,
    collective_algorithm: Annotated[
        str | None,
        typer.Option(
            "--collective-algorithm",
            help=(
                "Collective decomposition model: ring, tree, direct (default "
                "ring; overrides manifest)."
            ),
        ),
    ] = None,
    alpha: Annotated[
        float | None,
        typer.Option(
            "--alpha",
            help=(
                "Significance level for Welch/Mann-Whitney tests (default "
                "0.05; overrides manifest)."
            ),
        ),
    ] = None,
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
    predict: Annotated[
        str | None,
        typer.Option(
            "--predict",
            help=(
                "What-if scenario for the 'whatif' aspect, e.g. 'comm:1.6x' "
                "(rules joined by ';'); gate it with 'whatif_residual:5%'."
            ),
        ),
    ] = None,
    labels: Annotated[
        str | None,
        typer.Option("--labels", "-l", help="Comma-separated group labels."),
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
        pathlib.Path | None,
        typer.Option(
            "--cache-dir",
            help="Cache directory (default .cta_cache; overrides manifest).",
        ),
    ] = None,
) -> None:
    """Compares two traces or bundles (baseline and candidate), or N of them."""
    from cta import api
    from cta import engine
    from cta import gating
    from cta.ingest import manifest as ingest_manifest

    output.check_export_formats(out_dir, export)
    try:
        gates = gating.parse_gate_spec(gate)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--gate") from exc
    try:
        if manifest is not None:
            if traces:
                output.err_console.print(
                    "[yellow]warning: positional traces are ignored when "
                    "--manifest is given.[/yellow]"
                )
            groups, config = ingest_manifest.load_manifest(manifest)
            # Options given on the command line override the manifest config.
            overrides = {
                "aspects": tuple(output.split_comma_separated(aspects) or ())
                or None,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "collective_algorithm": collective_algorithm,
                "alpha": alpha,
                "scope_depth": scope_depth,
                "ranks": tuple(output.split_comma_separated_ints(ranks))
                or None,
                "devices": tuple(output.split_comma_separated_ints(devices))
                or None,
                "predict": predict,
                "cache_dir": cache_dir,
            }
            for field_name, value in overrides.items():
                if value is not None:
                    setattr(config, field_name, value)
            config.gates.update(gates)
            config.allow_missing_gates = (
                config.allow_missing_gates or allow_missing_gates
            )
            report = engine.run_comparison(groups, config)
        else:
            if not traces or len(traces) < 2:
                raise typer.BadParameter(
                    "Provide at least two trace paths/directories or "
                    "--manifest."
                )
            report = api.compare(
                traces[0],
                traces[1],
                *traces[2:],
                aspects=output.split_comma_separated(aspects),
                warmup_steps=warmup_steps or 0,
                max_steps=max_steps,
                gates=gates,
                allow_missing_gates=allow_missing_gates,
                collective_algorithm=collective_algorithm or "ring",
                alpha=alpha if alpha is not None else 0.05,
                scope_depth=scope_depth,
                ranks=output.split_comma_separated_ints(ranks),
                devices=output.split_comma_separated_ints(devices),
                predict=predict,
                cache_dir=cache_dir or pathlib.Path(".cta_cache"),
                labels=output.split_comma_separated(labels),
            )
    except (ValueError, FileNotFoundError) as exc:
        # For example an unknown gate metric, no common formats or
        # inapplicable aspects.
        output.print_error(exc)
        raise typer.Exit(code=2) from exc

    output.emit_report(report, output_format, out_dir, export)
    if report.violations:
        raise typer.Exit(code=2)


def register(app: typer.Typer) -> None:
    """Adds ``cta compare`` to the application."""
    app.command("compare")(compare_command)
