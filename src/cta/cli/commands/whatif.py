"""The ``whatif`` command."""

from __future__ import annotations

import json
import pathlib
from typing import Annotated

from rich import table as rich_table
import typer

from cta.cli import output


def _format_ns(value: float) -> str:
    """Formats nanoseconds with a readable unit."""
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.3f} {unit}"
    return f"{value:.0f} ns"


def whatif_command(
    trace: Annotated[
        str,
        typer.Argument(help="Chakra .et trace or directory, or GOAL schedule."),
    ],
    speed: Annotated[
        list[str] | None,
        typer.Option(
            "--speed",
            "-s",
            help=(
                "A what-if scenario, '<selector>:<factor>' with rules joined "
                "by ';', e.g. 'comm:2x', 'name~attention:-30%', "
                "'rank=7:median'. Repeat for several scenarios."
            ),
        ),
    ] = None,
    sweep: Annotated[
        bool,
        typer.Option(
            "--sweep",
            help=(
                "Speed up every op group in turn. On by default when no "
                "--speed is given."
            ),
        ),
    ] = False,
    by: Annotated[
        str,
        typer.Option("--by", help="Sweep groups: family, class, rank or name."),
    ] = "family",
    factors: Annotated[
        str,
        typer.Option(
            "--factors",
            help="Sweep speedups, comma separated; 'inf' is always added.",
        ),
    ] = "1.1,1.25,1.5,2,4",
    top: Annotated[
        int,
        typer.Option("--top", min=1, help="Groups to keep in the sweep."),
    ] = 24,
    threads: Annotated[
        int,
        typer.Option(
            "--threads", min=0, help="Worker threads (0 = all usable CPUs)."
        ),
    ] = 0,
    memory_gb: Annotated[
        float,
        typer.Option(
            "--memory-gb",
            min=0.0,
            help="Memory the solve may use (0 = 60% of what is free).",
        ),
    ] = 0.0,
    output_format: Annotated[
        str, typer.Option("--format", help=output.FORMAT_HELP)
    ] = "rich",
    cache_dir: Annotated[
        pathlib.Path, typer.Option("--cache-dir", help="Cache directory.")
    ] = pathlib.Path(".cta_cache"),
) -> None:
    """Predicts how much faster a run gets if some of its ops get faster.

    Solves the run's execution graph again with the changed op times, so a
    run's critical path answers where speeding something up pays off.
    """
    from cta.analysis import whatif

    if output_format not in ("rich", "json"):
        raise typer.BadParameter(
            f"Unknown --format '{output_format}' (expected 'rich' or 'json')."
        )
    budget = int(memory_gb * 1024**3)
    try:
        scenarios = [whatif.parse_scenario(text) for text in speed or []]
        sweep_factors = whatif.parse_factors(factors)
        model = whatif.load_model(trace, cache_dir=cache_dir)
        run = whatif.run_scenarios(
            model, scenarios, threads=threads, memory_budget_bytes=budget
        )
        result = (
            whatif.sweep(
                model,
                by=by,
                factors=sweep_factors,
                max_groups=top,
                threads=threads,
                memory_budget_bytes=budget,
            )
            if sweep or not scenarios
            else None
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        output.print_error(exc)
        raise typer.Exit(code=2) from exc

    if output_format == "json":
        document: dict[str, object] = {
            "vertices": model.graph.num_nodes,
            "edges": model.plan.num_edges,
            "prepare_seconds": model.prepare_seconds,
            "scenarios": run.table.to_dicts(),
        }
        if result is not None:
            document["sweep"] = result.table.to_dicts()
            document["ceiling"] = result.ceiling.to_dicts()
            document["sweep_seconds"] = result.solve_seconds
        typer.echo(json.dumps(document, default=str, allow_nan=True))
        return

    console = output.console
    console.print(
        f"{model.graph.num_nodes:,} ops, {model.plan.num_edges:,} edges; "
        f"graph prepared in {model.prepare_seconds:.2f} s"
    )
    if scenarios:
        scenario_table = rich_table.Table(title="What-if scenarios")
        for column in (
            "scenario",
            "makespan",
            "speedup",
            "compute on path",
            "comm on path",
        ):
            scenario_table.add_column(column)
        for row in run.table.iter_rows(named=True):
            scenario_table.add_row(
                str(row["scenario"]),
                _format_ns(row["makespan_ns"]),
                f"{row['speedup']:.3f}x",
                _format_ns(row["cp_compute_ns"]),
                _format_ns(row["cp_exposed_comm_ns"]),
            )
        console.print(scenario_table)
    if result is not None:
        ceiling_table = rich_table.Table(
            title=(
                f"Where speeding up pays off (baseline "
                f"{_format_ns(result.baseline_makespan_ns)})"
            )
        )
        for column in (
            "group",
            "time share",
            "on critical path",
            "ceiling",
            "90% of it at",
        ):
            ceiling_table.add_column(column)
        for row in result.ceiling.iter_rows(named=True):
            knee = row["knee_factor"]
            ceiling_table.add_row(
                str(row["group"]),
                f"{row['time_share_pct']:.1f}%",
                f"{row['critical_share_pct']:.1f}%",
                f"{row['ceiling_speedup']:.3f}x",
                "removal only" if knee == float("inf") else f"{knee:g}x",
            )
        console.print(ceiling_table)
        console.print(
            f"{result.table.height + 1} scenarios solved in "
            f"{result.solve_seconds:.2f} s ({result.lanes} per pass, "
            f"{result.workers} threads)"
        )


def register(app: typer.Typer) -> None:
    """Adds ``cta whatif`` to the application."""
    app.command("whatif")(whatif_command)
