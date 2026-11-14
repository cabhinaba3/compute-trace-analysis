"""The ``compare``, ``verify-equivalence`` and ``analyze`` commands."""

from __future__ import annotations

import pathlib
from typing import Annotated

from rich import table
import typer

from cta.cli import output


def _split_comma_separated_ints(value: str | None) -> list[int]:
    """Parses a comma-separated option value such as ``"0,1"`` into ints."""
    return [int(item) for item in output.split_comma_separated(value) or []]


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
    """Compare two (Baseline vs Candidate) or N traces/bundles across performance aspects."""  # noqa: E501
    from cta.core import diagnosis
    from cta.core import engine
    from cta.core import manifest as core_manifest

    try:
        gates = diagnosis.parse_gate_spec(gate)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--gate") from exc
    try:
        if manifest is not None:
            if traces:
                output.err_console.print(
                    "[yellow]warning: positional traces are ignored when "
                    "--manifest is given.[/yellow]"
                )
            groups, config = core_manifest.load_manifest(manifest)
            # Options given on the command line override the manifest config.
            overrides = {
                "aspects": tuple(output.split_comma_separated(aspects) or ())
                or None,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "collective_algorithm": collective_algorithm,
                "alpha": alpha,
                "scope_depth": scope_depth,
                "ranks": tuple(_split_comma_separated_ints(ranks)) or None,
                "devices": tuple(_split_comma_separated_ints(devices)) or None,
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
            report = engine.compare(
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
                ranks=_split_comma_separated_ints(ranks),
                devices=_split_comma_separated_ints(devices),
                cache_dir=cache_dir or pathlib.Path(".cta_cache"),
                labels=output.split_comma_separated(labels),
            )
    except (ValueError, FileNotFoundError) as exc:
        # For example an unknown gate metric, no common formats or
        # inapplicable aspects.
        output.err_console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc

    output.emit_report(report, output_format, out_dir, export)
    if report.violations:
        raise typer.Exit(code=2)


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
    """Verify compute duration and decomposed collective byte equivalence between Chakra-ET and GOAL."""  # noqa: E501
    from cta.aspects import distributed

    verdict = distributed.verify_chakra_goal_equivalence(
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
    """Analyze and characterize a single trace, glob, or multi-artifact RunBundle directory."""  # noqa: E501
    from cta.core import engine

    try:
        report = engine.analyze(
            trace,
            aspects=output.split_comma_separated(aspects),
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            collective_algorithm=collective_algorithm,
            alpha=alpha,
            scope_depth=scope_depth,
            ranks=_split_comma_separated_ints(ranks),
            devices=_split_comma_separated_ints(devices),
            cache_dir=cache_dir,
            label=label,
        )
    except (ValueError, FileNotFoundError) as exc:
        output.err_console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc
    output.emit_report(report, output_format, out_dir, export)
