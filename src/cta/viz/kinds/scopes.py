"""Plot kind scope_icicle: how kernel time nests in scopes."""

from __future__ import annotations

from typing import Any

import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz.kinds import chart_helpers

_SEPARATOR = " > "


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


@registry.register_plot(
    "scope_icicle",
    group="timeline",
    title="Time by nested scope",
    needs="kernels table with scopes (NVTX ranges or CPU operators)",
    aliases=("icicle", "scope_tree"),
    options={
        "metric": registry.Opt(
            "str",
            "duration_ms",
            "Column summed over the kernels of each scope.",
        ),
        "depth": registry.Opt("int", 6, "Levels of nesting to draw."),
        "min_share": registry.Opt(
            "float",
            0.5,
            "Leave out scopes below this percent of the group's total.",
        ),
    },
)
def scope_icicle(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
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
        NotApplicableError: The kernels table is empty or lacks the
            metric column.
    """
    kernels = ctx.frame("kernels", options)
    metric = options["metric"]
    if metric not in kernels.columns:
        raise registry.NotApplicableError(
            f"the kernels table has no '{metric}'"
        )
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
