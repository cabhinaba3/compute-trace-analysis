"""Plot kinds of simulated runs (the ``simulation`` aspect's tables).

``cta simulate compare --out-dir D --plot ...`` and ``cta compare --aspects
simulation`` both produce these tables.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry

# Critical-path segments in drawing order (bottom of the stack first).
_SEGMENTS = (
    "compute",
    "noise",
    "overhead",
    "latency",
    "transfer",
    "contention",
    "collective",
    "idle",
)


def _table(ctx: context.DataContext, name: str) -> pl.DataFrame:
    """Returns a details table of the simulation aspect, or explains why not."""
    if not ctx.has_aspect("simulation"):
        raise registry.NotApplicableError("needs the 'simulation' aspect")
    table = ctx.report_table("simulation", name)
    if table.is_empty():
        raise registry.NotApplicableError(f"the simulation has no {name} table")
    return table


@registry.register_plot(
    "sim_breakdown",
    group="distributed",
    title="Simulated critical path by segment",
    needs="aspect simulation",
    exported=True,
)
def sim_breakdown(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Stacks each run's critical path by segment (compute, latency, ...)."""
    del options
    table = _table(ctx, "breakdown")
    runs = table["run"].unique(maintain_order=True).to_list()
    traces = []
    for index, segment in enumerate(_SEGMENTS):
        rows = table.filter(pl.col("segment") == segment)
        values = dict(
            zip(
                rows["run"].to_list(), (rows["ns"] / 1e6).to_list(), strict=True
            )
        )
        if not any(values.values()):
            continue
        traces.append(
            ir.Trace(
                "bar",
                x=runs,
                y=[values.get(run, 0.0) for run in runs],
                name=segment,
                style={"ci": index},
            )
        )
    return ir.Chart(
        title="Simulated critical path",
        panels=[
            ir.Panel(
                traces=traces,
                ylabel="time on the critical path (ms)",
                stacked=True,
            )
        ],
    )


@registry.register_plot(
    "sim_attribution",
    group="distributed",
    title="Makespan gap to the baseline by critical-path segment",
    needs="aspect simulation with two or more runs",
    exported=True,
)
def sim_attribution(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots, per run, how much each segment adds to or removes from the gap."""
    del options
    table = _table(ctx, "attribution")
    runs = table["run"].unique(maintain_order=True).to_list()
    panels = []
    for run in runs:
        rows = table.filter((pl.col("run") == run) & (pl.col("delta_ns") != 0))
        panels.append(
            ir.Panel(
                traces=[
                    ir.Trace(
                        "bar",
                        x=rows["segment"].to_list(),
                        y=(rows["delta_ns"] / 1e6).to_list(),
                        name=run,
                    )
                ],
                title=run,
                ylabel="delta vs baseline (ms)",
                legend=False,
            )
        )
    return ir.Chart(
        title="Where the makespan gap comes from",
        panels=panels,
        cols=min(len(panels), 2),
    )


@registry.register_plot(
    "sim_ranks",
    group="distributed",
    title="Simulated finish time per rank",
    needs="aspect simulation",
    exported=True,
)
def sim_ranks(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each rank's simulated finish time, one line per run."""
    del options
    table = _table(ctx, "ranks")
    traces = [
        ir.Trace(
            "line",
            x=rows["rank"].to_list(),
            y=(rows["finish_ns"] / 1e6).to_list(),
            name=str(run),
        )
        for (run,), rows in table.sort("rank").group_by(
            "run", maintain_order=True
        )
    ]
    return ir.Chart(
        title="Simulated finish time per rank",
        panels=[ir.Panel(traces=traces, xlabel="rank", ylabel="finish (ms)")],
    )


@registry.register_plot(
    "sim_phases",
    group="distributed",
    title="Phase span differences from the baseline",
    needs="aspect simulation with two or more runs",
    options={
        "rank": registry.Opt("list", None, "Only these ranks."),
    },
    exported=True,
)
def sim_phases(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each phase's span minus the baseline's, per run and rank."""
    table = _table(ctx, "phases")
    if options.get("rank"):
        wanted = [int(r) for r in options["rank"]]
        table = table.filter(pl.col("rank").is_in(wanted))
    traces = [
        ir.Trace(
            "scatter",
            x=rows["phase"].to_list(),
            y=(rows["span_delta_ns"] / 1e3).to_list(),
            name=f"{run} r{rank}",
            style={"alpha": 0.7},
        )
        for (run, rank), rows in table.sort("phase").group_by(
            "run", "rank", maintain_order=True
        )
    ]
    return ir.Chart(
        title="Phase span: run minus baseline",
        panels=[
            ir.Panel(
                traces=traces,
                xlabel="phase (compute anchor index)",
                ylabel="span delta (us)",
            )
        ],
    )
