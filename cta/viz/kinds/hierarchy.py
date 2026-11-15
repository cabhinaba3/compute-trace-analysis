"""Plots of how time nests in scopes, and where the GPU sits idle."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz import data
from cta.viz import ir
from cta.viz import spec
from cta.viz.kinds import chart_helpers

_SEPARATOR = " > "
_GAP_EDGES_US = (10.0, 100.0, 1_000.0, 10_000.0)
_GAP_CLASSES = ("<10 us", "10-100 us", "0.1-1 ms", "1-10 ms", ">=10 ms")


class _Node:
    """One scope in the tree: its inclusive time and its children."""

    def __init__(self) -> None:
        self.value = 0.0
        self.children: dict[str, _Node] = {}


def _scope_paths(group: pl.DataFrame) -> tuple[pl.Series, str]:
    """Gives each kernel the path it hangs under in the tree.

    Kernels carry the NVTX or CPU-operator scopes that enclose their launch.
    A trace without any falls back to category, operator family and name.

    Returns:
        The path of every row, and a note saying what it is made of.
    """
    paths = group["scope_path"].fill_null("")
    if (paths != "").any():
        return (
            paths.str.replace("^$", "(no scope)"),
            "Levels are the scopes that enclose each kernel's launch.",
        )
    fallback = pl.concat_str(
        [
            pl.col("category"),
            pl.col("op_family").fill_null("other"),
            pl.col("name").str.slice(0, 40),
        ],
        separator=_SEPARATOR,
    )
    return (
        group.select(fallback.alias("path"))["path"],
        "The trace has no scopes: levels are category, operator family, name.",
    )


def _build_tree(paths: pl.Series, values: pl.Series, depth: int) -> _Node:
    """Adds up the values of every path and of each prefix of it."""
    root = _Node()
    totals = (
        pl.DataFrame({"path": paths, "value": values}).group_by("path").sum()
    )
    for path, value in totals.iter_rows():
        node = root
        node.value += value
        for part in path.split(_SEPARATOR)[:depth]:
            node = node.children.setdefault(part, _Node())
            node.value += value
    return root


def _place(
    node: _Node,
    left: float,
    level: int,
    top: int | None,
    out: list[tuple[int, float, float, str, int]],
) -> None:
    """Lays out the children of a node left to right, largest first."""
    for index, (name, child) in enumerate(
        sorted(node.children.items(), key=lambda item: -item[1].value)
    ):
        branch = index if top is None else top
        out.append((level, left, left + child.value, name, branch))
        _place(child, left, level + 1, branch, out)
        left += child.value


@spec.register_plot(
    "scope_icicle",
    group="timeline",
    title="Time by nested scope",
    needs="kernels table with scopes (NVTX ranges or CPU operators)",
    aliases=("icicle", "scope_tree"),
    options={
        "metric": spec.Opt(
            "str",
            "duration_ms",
            "Column summed over the kernels of each scope.",
        ),
        "depth": spec.Opt("int", 6, "Levels of nesting to draw."),
        "min_share": spec.Opt(
            "float",
            0.5,
            "Leave out scopes below this percent of the group's total.",
        ),
    },
)
def scope_icicle(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots GPU time as an icicle: each scope sits under its parent.

    A scope's width is the time of every kernel launched inside it, so the
    row below a scope shows what it spent its time on. Each group gets a
    panel with the same width scale.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A grid with one icicle per group.

    Raises:
        spec.NotApplicableError: The kernels table is empty or lacks the
            metric column.
    """
    kernels = ctx.frame("kernels", options)
    metric = options["metric"]
    if metric not in kernels.columns:
        raise spec.NotApplicableError(f"the kernels table has no '{metric}'")
    depth = max(1, int(options["depth"]))
    lanes = [f"level {level}" for level in range(depth)]
    panels, notes = [], ""
    trees = []
    for label in chart_helpers.unique_in_order(kernels, "label"):
        group = kernels.filter(pl.col("label") == label)
        paths, notes = _scope_paths(group)
        trees.append((str(label), _build_tree(paths, group[metric], depth)))
    widest = max(tree.value for _, tree in trees)
    for label, tree in trees:
        cells: list[tuple[int, float, float, str, int]] = []
        _place(tree, 0.0, 0, None, cells)
        cutoff = tree.value * float(options["min_share"]) / 100.0
        cells = [cell for cell in cells if cell[2] - cell[1] >= cutoff]
        traces: list[ir.Trace] = []
        for branch in sorted({cell[4] for cell in cells}):
            mine = [cell for cell in cells if cell[4] == branch]
            traces.append(
                ir.Trace(
                    "segment",
                    x=[cell[1] for cell in mine],
                    x1=[cell[2] for cell in mine],
                    y=[lanes[cell[0]] for cell in mine],
                    text=[
                        f"{cell[3]}: {cell[2] - cell[1]:.4g}" for cell in mine
                    ],
                    style={"lanes": lanes, "ci": branch, "width": 13},
                    meta={"label": label},
                )
            )
        wide = [c for c in cells if c[2] - c[1] >= 0.04 * widest]
        traces.append(
            ir.Trace(
                "text",
                x=[(c[1] + c[2]) / 2 for c in wide],
                y=[lanes[c[0]] for c in wide],
                text=[c[3][:18] for c in wide],
                style={"lanes": lanes},
                meta={"label": label},
            )
        )
        panels.append(
            ir.Panel(
                traces=traces,
                title=label,
                xlabel=metric,
                xrange=(0.0, widest),
                invert_y=True,
                legend=False,
            )
        )
    return chart_helpers.grid(
        "Time by nested scope", panels, max_cols=1, notes=notes
    )


def _gaps_ms(group: pl.DataFrame) -> tuple[np.ndarray, float]:
    """Finds the idle gaps of a group's devices.

    A gap is a stretch inside a device's active span where none of its
    kernels runs. Each device is taken on its own, in its rank's time.

    Returns:
        The gap lengths in milliseconds, and the total active span.
    """
    gaps, span = [], 0.0
    for _, device in group.group_by(["rank", "device"]):
        starts, ends = chart_helpers.merge_intervals(
            device["t_ms"].to_numpy().astype(float),
            device["end_ms"].to_numpy().astype(float),
        )
        if not len(starts):
            continue
        span += float(ends[-1] - starts[0])
        between = starts[1:] - ends[:-1]
        gaps.append(between[between > 0])
    return (np.concatenate(gaps) if gaps else np.zeros(0)), span


@spec.register_plot(
    "idle_gaps",
    group="timeline",
    title="GPU idle gaps between kernels",
    needs="kernels table",
    aliases=("gaps", "gpu_idle"),
    options={
        "min_gap_us": spec.Opt(
            "float", 0.0, "Ignore gaps shorter than this many microseconds."
        ),
        "bins": spec.Opt("int", 30, "Bins of the gap-length histogram."),
    },
)
def idle_gaps(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots how long the GPU waits between kernels, and which waits add up.

    Gaps come from the union of each device's kernels, so overlapping
    kernels on several streams never make a negative gap. The left panel
    counts gaps by length; the right one shows how the idle time splits into
    short launch gaps and long stalls.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        Two panels: gap lengths, and idle time by gap length.

    Raises:
        spec.NotApplicableError: No device has a gap.
    """
    kernels = ctx.frame("kernels", options)
    floor_ms = float(options["min_gap_us"]) / 1e3
    found = []
    for label in chart_helpers.unique_in_order(kernels, "label"):
        gaps, span = _gaps_ms(kernels.filter(pl.col("label") == label))
        found.append((str(label), gaps[gaps >= floor_ms], span))
    if not any(len(gaps) for _, gaps, _ in found):
        raise spec.NotApplicableError("the kernels leave no gaps to measure")
    everything = np.concatenate([gaps for _, gaps, _ in found if len(gaps)])
    bins = max(2, int(options["bins"]))
    edges = np.geomspace(
        everything.min(), max(everything.max(), everything.min() * 2), bins + 1
    )
    class_edges = np.array(_GAP_EDGES_US) / 1e3
    counts, shares, notes = [], [], []
    for index, (label, gaps, span) in enumerate(found):
        style = {"ci": index}
        histogram, _ = np.histogram(gaps, bins=edges)
        counts.append(
            ir.Trace(
                "step",
                x=(edges[:-1] * 1e3).tolist(),
                y=histogram.tolist(),
                name=label,
                style=style,
                meta={"label": label},
            )
        )
        by_class = np.bincount(
            np.searchsorted(class_edges, gaps, side="right"),
            weights=gaps,
            minlength=len(_GAP_CLASSES),
        )
        total = by_class.sum()
        shares.append(
            ir.Trace(
                "bar",
                x=list(_GAP_CLASSES),
                y=(100.0 * by_class / total).tolist()
                if total > 0
                else [0.0] * len(_GAP_CLASSES),
                name=label,
                style=style,
                meta={"label": label},
            )
        )
        if span > 0:
            notes.append(f"{label}: idle {100.0 * gaps.sum() / span:.1f}%")
    return chart_helpers.grid(
        "GPU idle gaps",
        [
            ir.Panel(
                traces=counts,
                title="Gaps by length",
                xlabel="gap length (us)",
                ylabel="gaps",
                xscale="log",
            ),
            ir.Panel(
                traces=shares,
                title="Idle time by gap length",
                xlabel="gap length",
                ylabel="% of idle time",
                legend=False,
            ),
        ],
        max_cols=2,
        notes="Share of the active span spent idle: " + "; ".join(notes),
        meta={"skip_suitability": True},
    )


@spec.register_plot(
    "critical_path",
    group="distributed",
    title="Critical path: compute, exposed communication and slack",
    needs="aspect dag_critical_path (Chakra or GOAL traces)",
    aliases=("cp_summary",),
    options={},
)
def critical_path(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
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
        spec.NotApplicableError: The report lacks the aspect.
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
        raise spec.NotApplicableError("the aspect summary has no group values")

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
