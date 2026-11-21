"""Plot kinds of the ``simulation`` aspect's tables."""

from __future__ import annotations

from typing import Any

import polars as pl

from cta.sim import result
from cta.viz import context
from cta.viz import ir
from cta.viz import registry


def _table(ctx: context.DataContext, name: str) -> pl.DataFrame:
    if not ctx.has_aspect("simulation"):
        raise registry.NotApplicableError("needs the 'simulation' aspect")
    table = ctx.report_table("simulation", name)
    if table.is_empty():
        raise registry.NotApplicableError(f"the simulation has no {name} table")
    return table


def _kind(name: str, title: str) -> Any:
    return registry.register_plot(
        name,
        group="distributed",
        title=title,
        needs="aspect simulation",
        exported=True,
    )


@_kind("sim_breakdown", "Simulated critical path by segment")
def sim_breakdown(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Each run's critical path, stacked by segment."""
    del options
    table = _table(ctx, "breakdown")
    traces = [
        ir.Trace(
            "bar",
            x=rows["run"].to_list(),
            y=(rows["ns"] / 1e6).to_list(),
            name=segment,
            style={"ci": i},
        )
        for i, segment in enumerate(result.SEGMENTS)
        if (rows := table.filter(pl.col("segment") == segment))["ns"].sum() > 0
    ]
    return ir.Chart(
        title="Simulated critical path",
        panels=[
            ir.Panel(
                traces=traces, ylabel="ms on the critical path", stacked=True
            )
        ],
    )


@_kind(
    "sim_attribution", "Makespan gap to the baseline by critical-path segment"
)
def sim_attribution(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Per run, what each segment adds to the gap."""
    del options
    panels = [
        ir.Panel(
            traces=[
                ir.Trace(
                    "bar",
                    x=rows["segment"].to_list(),
                    y=(rows["delta_ns"] / 1e6).to_list(),
                )
            ],
            title=str(run),
            ylabel="delta (ms)",
            legend=False,
        )
        for (run,), rows in _table(ctx, "attribution")
        .filter(pl.col("delta_ns") != 0)
        .group_by("run", maintain_order=True)
    ]
    return ir.Chart(
        title="Where the makespan gap comes from",
        panels=panels,
        cols=min(len(panels), 2),
    )


@_kind("sim_ranks", "Simulated finish time per rank")
def sim_ranks(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Each rank's finish time, one line per run."""
    del options
    traces = [
        ir.Trace(
            "line",
            x=rows["rank"].to_list(),
            y=(rows["finish_ns"] / 1e6).to_list(),
            name=str(run),
        )
        for (run,), rows in _table(ctx, "ranks")
        .sort("rank")
        .group_by("run", maintain_order=True)
    ]
    return ir.Chart(
        title="Simulated finish time per rank",
        panels=[ir.Panel(traces=traces, xlabel="rank", ylabel="finish (ms)")],
    )


@_kind("sim_phases", "Phase span differences from the baseline")
def sim_phases(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Each phase's span minus the baseline's, per run and rank."""
    del options
    traces = [
        ir.Trace(
            "scatter",
            x=rows["phase"].to_list(),
            y=(rows["span_delta_ns"] / 1e3).to_list(),
            name=f"{run} r{rank}",
        )
        for (run, rank), rows in _table(ctx, "phases")
        .sort("phase")
        .group_by("run", "rank", maintain_order=True)
    ]
    return ir.Chart(
        title="Phase span: run minus baseline",
        panels=[ir.Panel(traces=traces, xlabel="phase", ylabel="delta (us)")],
    )
