"""Plot kinds of distributed runs: traffic, rank skew and the critical path."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz.kinds import chart_helpers


def _traffic_matrix(volumes: pl.DataFrame, n_ranks: int) -> np.ndarray:
    """Fills an n_ranks x n_ranks source-by-destination matrix of MB sent."""
    matrix = np.zeros((n_ranks, n_ranks))
    for row in volumes.iter_rows(named=True):
        matrix[int(row["src_rank"]), int(row["dst_rank"])] = float(
            row["volume_mb"]
        )
    return matrix


def _highest_rank(volumes: pl.DataFrame) -> int:
    """Returns the highest source or destination rank in a traffic table."""
    return max(
        int(cast(Any, volumes["src_rank"].max())),
        int(cast(Any, volumes["dst_rank"].max())),
    )


@registry.register_plot(
    "traffic_delta",
    group="distributed",
    title="Collective traffic matrix",
    aliases=("heatmap", "traffic"),
    exported=True,
    needs="aspect traffic_matrix",
)
def traffic(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots rank-to-rank collective traffic volume.

    Draws one heatmap per group, or for a pair the candidate minus the
    baseline.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel delta chart for two groups, else a grid of heatmaps.

    Raises:
        NotApplicableError: The report lacks the traffic_matrix aspect.
    """
    labels = ctx.labels
    traffic_aspect = ctx.require_report().aspects.get("traffic_matrix")
    if traffic_aspect is None:
        raise registry.NotApplicableError(
            "Traffic heatmap requires 'traffic_matrix' aspect"
        )
    volumes_per_label = [
        traffic_aspect.details.get(f"matrix_{label}", pl.DataFrame())
        for label in labels
    ]
    n_ranks = max(
        (
            _highest_rank(volumes) + 1
            for volumes in volumes_per_label
            if volumes.height
        ),
        default=2,
    )
    ranks = list(range(n_ranks))
    if len(labels) == 2:
        baseline_matrix = _traffic_matrix(volumes_per_label[0], n_ranks)
        candidate_matrix = _traffic_matrix(volumes_per_label[1], n_ranks)
        delta = candidate_matrix - baseline_matrix
        color_limit = max(1e-6, float(np.abs(delta).max()))
        heatmap = ir.Trace(
            "heatmap",
            x=ranks,
            y=ranks,
            z=delta.tolist(),
            style={"cmap": "RdBu_r", "vmin": -color_limit, "vmax": color_limit},
        )
        return chart_helpers.single(
            f"Traffic delta ({labels[1]} - {labels[0]})",
            ir.Panel(
                traces=[heatmap],
                xlabel="Destination rank",
                ylabel="Source rank",
                colorbar="Delta volume (MB)",
                legend=False,
            ),
        )
    panels = []
    for label, volumes in zip(labels, volumes_per_label, strict=True):
        heatmap = ir.Trace(
            "heatmap",
            x=ranks,
            y=ranks,
            z=_traffic_matrix(volumes, n_ranks).tolist(),
            style={"cmap": "Blues"},
        )
        panels.append(
            ir.Panel(
                traces=[heatmap],
                title=label,
                xlabel="Destination rank",
                ylabel="Source rank",
                colorbar="Volume (MB)",
                legend=False,
            )
        )
    return chart_helpers.grid("Traffic matrix", panels)


@registry.register_plot(
    "skew",
    group="distributed",
    title="Per-rank load distribution",
    exported=True,
    needs="aspect rank_skew",
)
def skew(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots a box of per-rank compute time for each group.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one box per group that has per-rank data.

    Raises:
        NotApplicableError: The report lacks the rank_skew aspect or its
            per-rank data.
    """
    if not ctx.has_aspect("rank_skew"):
        raise registry.NotApplicableError(
            "Skew plot requires 'rank_skew' aspect"
        )
    per_rank_by_label = ctx.require_report().aspects["rank_skew"].details
    traces = []
    for i, label in enumerate(ctx.labels):
        if (
            label not in per_rank_by_label
            or not per_rank_by_label[label].height
        ):
            continue
        traces.append(
            ir.Trace(
                "box",
                y=per_rank_by_label[label]["compute_us"].to_list(),
                name=label,
                style={"ci": i},
                meta={"label": label},
            )
        )
    if not traces:
        raise registry.NotApplicableError(
            "No per-rank distribution data available for skew plot"
        )
    return chart_helpers.single(
        "Multi-rank load distribution",
        ir.Panel(
            traces=traces, ylabel="Per-rank compute time (us)", legend=False
        ),
    )


@registry.register_plot(
    "critical_path",
    group="distributed",
    title="Critical path: compute, exposed communication and slack",
    needs="aspect dag_critical_path (Chakra or GOAL traces)",
    aliases=("cp_summary",),
    options={},
)
def critical_path(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots what the execution graph's critical path is made of.

    The solver reports totals only, so this shows how long the path is, how
    much of it is compute and how much is communication nobody hides, and
    how much slack the other ops have. It does not draw the path itself.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind (none).

    Returns:
        Three panels: path length split in two, mean slack, and the share of
        ops with no slack.

    Raises:
        NotApplicableError: The report lacks the aspect.
    """
    del options
    values = chart_helpers.summary_values(
        ctx.report_table("dag_critical_path"), ctx.labels
    )
    labels = [
        label
        for label in ctx.labels
        if label in next(iter(values.values()), {})
    ]
    if not labels:
        raise registry.NotApplicableError(
            "the aspect summary has no group values"
        )

    def bars(
        metric: str, name: str | None, index: int | None = None
    ) -> ir.Trace:
        return ir.Trace(
            "bar",
            x=labels,
            y=[
                values.get(metric, {}).get(label, float("nan"))
                for label in labels
            ],
            name=name,
            style={"ci": index} if index is not None else {},
        )

    return chart_helpers.grid(
        "Critical path",
        [
            ir.Panel(
                traces=[
                    bars("cp_compute_us", "compute", 0),
                    bars("cp_exposed_comm_us", "exposed communication", 1),
                ],
                title="Critical path length",
                ylabel="us",
                stacked=True,
            ),
            ir.Panel(
                traces=[bars("mean_slack_us", None, 2)],
                title="Mean slack of an op",
                ylabel="us",
                legend=False,
            ),
            ir.Panel(
                traces=[bars("zero_slack_pct", None, 3)],
                title="Ops with no slack",
                ylabel="% of ops",
                legend=False,
            ),
        ],
        max_cols=3,
    )
