"""The ``analyze`` command: analyses one trace, glob or run bundle."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from typing import Annotated

import typer

from cta.cli import output


def analyze_command(
    trace: Annotated[
        str,
        typer.Argument(
            help="Single trace file, glob, RunBundle directory, or '-' stdin."
        ),
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
    streaming: Annotated[
        str,
        typer.Option(
            "--streaming",
            help=(
                "Execution-graph streaming mode: auto, exact, frontier, "
                "windowed, aggregate, or off."
            ),
        ),
    ] = "auto",
    engine_name: Annotated[
        str,
        typer.Option(
            "--engine",
            help="DAG engine: gridgraph, graphchi, xstream, or frontier.",
        ),
    ] = "gridgraph",
    quantile_sketch: Annotated[
        str,
        typer.Option(
            "--quantile-sketch",
            help="Streaming quantile sketch: ddsketch, kll, gk, or exact.",
        ),
    ] = "ddsketch",
    memory_gb: Annotated[
        float,
        typer.Option(
            "--memory-gb",
            min=0.0,
            help="RAM budget in GiB for streaming passes (0 = auto).",
        ),
    ] = 0.0,
    collective_sync: Annotated[
        str,
        typer.Option(
            "--collective-sync",
            help="Cross-rank collective synchronization: none or barrier.",
        ),
    ] = "none",
    goal_semantics: Annotated[
        str,
        typer.Option(
            "--goal-semantics",
            help="GOAL interpretation: cta_legacy or strict.",
        ),
    ] = "cta_legacy",
    window_size: Annotated[
        int,
        typer.Option(
            "--window-size",
            min=1024,
            help="Frontier/windowed cache capacity in nodes.",
        ),
    ] = 262144,
    fail_on_cycle: Annotated[
        bool,
        typer.Option(
            "--fail-on-cycle",
            help="Fail immediately when a dependency cycle is detected.",
        ),
    ] = False,
    critical_path_ops: Annotated[
        bool,
        typer.Option(
            "--critical-path-ops",
            help="Extract the exact critical-path op chain table.",
        ),
    ] = False,
    delay_attribution: Annotated[
        bool,
        typer.Option(
            "--delay-attribution",
            help="Run Scalasca backward wait-state & delay attribution.",
        ),
    ] = False,
    scratch_dir: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--scratch-dir",
            help="Directory for temporary out-of-core scratch files.",
        ),
    ] = None,
    accelerator: Annotated[
        str,
        typer.Option(
            "--accelerator",
            help=(
                "Where to run: auto, cpu, cuda or metal. A GPU that is "
                "missing falls back to the CPU."
            ),
        ),
    ] = "auto",
) -> None:
    """Analyses one trace, a glob, or a directory holding a run bundle."""
    from cta import api
    from cta.analysis import streaming as streaming_mod

    if trace == "-":
        if streaming.lower().strip() != "aggregate":
            output.print_error(
                ValueError(
                    "Streaming from stdin ('-') requires --streaming aggregate."
                )
            )
            raise typer.Exit(code=2)
        if sys.stdin is sys.__stdin__:
            stream, inc_coll = streaming_mod.open_stream_for_path(
                "-", fmt="goal", goal_semantics=goal_semantics
            )
            agg = streaming_mod.run_streaming_aggregates(
                stream,
                include_collectives_in_traffic=inc_coll,
                quantile_sketch=quantile_sketch,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="cta_stdin_") as tmp_dir:
                tmp_goal = pathlib.Path(tmp_dir) / "stdin.goal"
                with tmp_goal.open("w", encoding="utf-8") as fh:
                    while chunk := sys.stdin.read(65536):
                        fh.write(chunk)
                stream, inc_coll = streaming_mod.open_stream_for_path(
                    tmp_goal, fmt="goal", goal_semantics=goal_semantics
                )
                agg = streaming_mod.run_streaming_aggregates(
                    stream,
                    include_collectives_in_traffic=inc_coll,
                    quantile_sketch=quantile_sketch,
                )
        payload = {
            "num_nodes": agg.num_nodes,
            "world_size": agg.world_size,
            "max_rank_compute_us": agg.max_rank_compute_us,
            "mean_rank_compute_us": agg.mean_rank_compute_us,
            "compute_skew_ratio": agg.compute_skew_ratio,
            "total_comm_mb": agg.total_comm_mb,
            "comm_op_count": agg.comm_op_count,
            "median_msg_kb": agg.median_msg_kb,
            "p95_msg_kb": agg.p95_msg_kb,
            "max_msg_kb": agg.max_msg_kb,
            "total_p2p_mb": agg.total_p2p_mb,
            "provenance": streaming_mod.provenance_row(
                label or "stdin", 0, agg.provenance
            ),
        }
        typer.echo(json.dumps(payload, indent=2))
        return

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
            predict=predict,
            cache_dir=cache_dir,
            label=label,
            streaming=streaming,
            graph_engine=engine_name,
            quantile_sketch=quantile_sketch,
            memory_budget_bytes=int(memory_gb * 1024**3),
            collective_sync=collective_sync,
            goal_semantics=goal_semantics,
            window_size=window_size,
            fail_on_cycle=fail_on_cycle,
            extract_critical_path_ops=critical_path_ops,
            delay_attribution=delay_attribution,
            scratch_dir=scratch_dir,
            accelerator=accelerator,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        output.print_error(exc)
        raise typer.Exit(code=2) from exc
    output.emit_report(report, output_format, out_dir, export)


def register(app: typer.Typer) -> None:
    """Adds ``cta analyze`` to the application."""
    app.command("analyze")(analyze_command)
