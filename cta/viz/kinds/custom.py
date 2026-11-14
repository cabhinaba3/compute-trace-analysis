"""Plot kind ``custom``: any raw table, report table or SQL result.

A small grammar of graphics: the user picks a raw table, a report table or a
DuckDB SQL query, then maps its columns to x, y, colour, facet and size.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz import data
from cta.viz import ir
from cta.viz import spec
from cta.viz.kinds import chart_helpers

_AGGREGATIONS = (
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "p50",
    "p90",
    "p99",
)
_MARKS = (
    "line",
    "step",
    "bar",
    "barh",
    "scatter",
    "box",
    "hist",
    "heatmap",
    "area",
)
# Marks that plot y against x, one trace per colour series.
_XY_MARKS = ("line", "step", "area", "scatter", "bar", "barh")
# Line series shorter than this get a marker on every point.
_MAX_POINTS_WITH_MARKERS = 60


def _aggregation_expr(column: str, aggregation: str) -> pl.Expr:
    """Returns the expression that aggregates a column, keeping its name."""
    values = pl.col(column)
    expr_by_aggregation = {
        "sum": values.sum(),
        "mean": values.mean(),
        "median": values.median(),
        "min": values.min(),
        "max": values.max(),
        "count": values.count(),
        "p50": values.quantile(0.5),
        "p90": values.quantile(0.9),
        "p99": values.quantile(0.99),
    }
    return expr_by_aggregation[aggregation].alias(column)


@spec.register_plot(
    "custom",
    group="custom",
    title="Any table or SQL query",
    needs="a source (raw table, report:<aspect>[.<table>]) or sql",
    options={
        "source": spec.Opt(
            "str",
            None,
            "kernels, steps, gpu_metrics, runtime, ncu_kernels, ncu_findings,"
            " jtop, osrt, or report:<aspect>[.<table>].",
        ),
        "sql": spec.Opt(
            "str",
            None,
            "DuckDB SQL over the named tables (cta plot --list-sources);"
            " overrides source.",
        ),
        "mark": spec.Opt("str", "line", "How to draw it.", _MARKS),
        "x": spec.Opt("str", None, "Column on x."),
        "y": spec.Opt("str", None, "Column on y (values)."),
        "color": spec.Opt(
            "str", None, "Column whose values become separate coloured series."
        ),
        "facet": spec.Opt(
            "str", None, "Column whose values become separate panels."
        ),
        "size": spec.Opt("str", None, "Column that scales scatter markers."),
        "agg": spec.Opt(
            "str", None, "Aggregate y per (x, color, facet).", _AGGREGATIONS
        ),
        "sort": spec.Opt(
            "str", None, "Sort bars by y: asc or desc.", ("asc", "desc")
        ),
        "top_k": spec.Opt(
            "int", None, "Keep the first K x values after sorting."
        ),
        "xscale": spec.Opt(
            "str", "linear", "linear or log.", ("linear", "log")
        ),
        "yscale": spec.Opt(
            "str", "linear", "linear or log.", ("linear", "log")
        ),
        "stack": spec.Opt("bool", False, "Stack bars."),
        "bins": spec.Opt("int", 30, "Histogram bins."),
        "xlabel": spec.Opt("str", None, "x axis label."),
        "ylabel": spec.Opt("str", None, "y axis label."),
    },
)
def custom(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Draws a user-defined plot of a chosen source or SQL result.

    The user picks a source or writes SQL, then maps columns to x, y, colour,
    facet and size.

    Args:
        ctx: Data access for the report or history being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart with one panel per facet value (one panel without a facet).

    Raises:
        spec.NotApplicableError: Neither `source` nor `sql` is given.
        ValueError: A mapped column is missing from the table, or the mark
            lacks a column it needs.
    """
    table = _read_source(ctx, options)
    mark, x, y = options["mark"], options.get("x"), options.get("y")
    _check_mapping(table, options, mark, x, y)
    if options.get("agg") and y:
        table = _aggregate(table, options, x, y)
    facet_column = options.get("facet")
    facet_values = (
        table[facet_column].unique(maintain_order=True).to_list()
        if facet_column
        else [None]
    )
    # Only a heatmap has a colour bar; it is labelled with the size column.
    colorbar = (options.get("size") or "") if mark == "heatmap" else ""
    panels = []
    for facet_value in facet_values:
        rows = (
            table
            if facet_value is None
            else table.filter(pl.col(options["facet"]) == facet_value)
        )
        if options.get("sort") and y and mark in ("bar", "barh"):
            rows = rows.sort(y, descending=options["sort"] == "desc")
        if options.get("top_k"):
            rows = rows.head(int(options["top_k"]))
        panels.append(
            ir.Panel(
                traces=_traces(rows, options, mark, x, y),
                title="" if facet_value is None else str(facet_value),
                xlabel=options.get("xlabel") or (x or ""),
                ylabel=options.get("ylabel") or (y or ""),
                xscale=options["xscale"],
                yscale=options["yscale"],
                stacked=bool(options["stack"]),
                colorbar=colorbar,
            )
        )
    default_title = f"{y} by {x}" if x and y else "custom plot"
    title = options.get("title") or default_title
    return chart_helpers.grid(title, panels, max_cols=3)


def _read_source(
    ctx: data.DataContext, options: dict[str, Any]
) -> pl.DataFrame:
    """Loads the table to plot from `sql`, a report table or a raw table."""
    if options.get("sql"):
        return ctx.sql(str(options["sql"]), options)
    if options.get("source"):
        source = str(options["source"])
        if source.startswith("report:"):
            aspect, _, table_name = source[len("report:") :].partition(".")
            return ctx.report_table(aspect, table_name or "summary")
        return ctx.frame(source, options)
    raise spec.NotApplicableError("custom needs `source` or `sql`")


def _check_mapping(
    table: pl.DataFrame,
    options: dict[str, Any],
    mark: str,
    x: str | None,
    y: str | None,
) -> None:
    """Raises ValueError for a missing column or a mark without its columns."""
    role_columns = (
        ("x", x),
        ("y", y),
        ("color", options.get("color")),
        ("facet", options.get("facet")),
        ("size", options.get("size")),
    )
    for role, column in role_columns:
        if column and column not in table.columns:
            raise ValueError(
                f"{role}='{column}' is not a column. "
                f"Columns: {', '.join(table.columns)}"
            )
    if mark not in ("hist", "box", "heatmap") and not (x and y):
        raise ValueError(f"mark '{mark}' needs both x and y")
    if mark == "hist" and not (y or x):
        raise ValueError("hist needs y (or x) for the values")


def _aggregate(
    table: pl.DataFrame, options: dict[str, Any], x: str | None, y: str
) -> pl.DataFrame:
    """Aggregates y per distinct (x, color, facet) key, or over all rows."""
    keys = [
        column
        for column in (x, options.get("color"), options.get("facet"))
        if column
    ]
    expr = _aggregation_expr(y, options["agg"])
    if keys:
        return table.group_by(list(dict.fromkeys(keys))).agg(expr)
    return table.select(expr)


def _traces(
    table: pl.DataFrame,
    options: dict[str, Any],
    mark: str,
    x: str | None,
    y: str | None,
) -> list[ir.Trace]:
    """Builds the traces of one panel, one series per value of `color`."""
    color_column = options.get("color")
    if mark == "heatmap":
        return [_heatmap_trace(table, x, y, color_column)]
    group_frames = (
        table.partition_by(color_column, maintain_order=True, as_dict=True)
        if color_column
        else {(None,): table}
    )
    traces: list[ir.Trace] = []
    for i, (key, group_frame) in enumerate(group_frames.items()):
        name = None if key[0] is None else str(key[0])
        has_markers = (
            mark == "line" and group_frame.height < _MAX_POINTS_WITH_MARKERS
        )
        style = {"ci": i, "marker": "o" if has_markers else None}
        if mark in _XY_MARKS:
            if x is None or y is None:
                raise ValueError(f"mark '{mark}' needs both x and y")
            traces.append(
                _xy_trace(
                    mark, group_frame, x, y, options.get("size"), name, style
                )
            )
        elif mark in ("box", "hist"):
            value_column = y or x
            if value_column is None:
                raise ValueError(f"mark '{mark}' needs y (or x) for the values")
            traces.extend(
                _distribution_traces(
                    mark,
                    group_frame,
                    x,
                    value_column,
                    options,
                    name,
                    style,
                    len(traces),
                )
            )
    return traces


def _heatmap_trace(
    table: pl.DataFrame, x: str | None, y: str | None, value_column: str | None
) -> ir.Trace:
    """Builds a heatmap of the mean `value_column` per (y, x) cell."""
    if not (x and y and value_column):
        raise ValueError("heatmap needs x, y and color (the cell value)")
    cells = table.pivot(
        on=x, index=y, values=value_column, aggregate_function="mean"
    ).sort(y)
    x_values = [column for column in cells.columns if column != y]
    return ir.Trace(
        "heatmap",
        x=x_values,
        y=cells[y].to_list(),
        z=cells.select(x_values).to_numpy().tolist(),
        style={"cmap": "viridis"},
    )


def _xy_trace(
    mark: str,
    group_frame: pl.DataFrame,
    x: str,
    y: str,
    size_column: str | None,
    name: str | None,
    style: dict[str, Any],
) -> ir.Trace:
    """Builds one trace of a mark that plots y against x.

    Args:
        mark: One of `_XY_MARKS`.
        group_frame: Rows of one colour series.
        x: Column on x (categories for bars).
        y: Column on y.
        size_column: Column that scales scatter markers, if any.
        name: Series name shown in the legend.
        style: Mark options shared by the series.

    Returns:
        The trace; line-like marks are sorted by x.
    """
    if mark in ("line", "step", "area"):
        sorted_rows = group_frame.sort(x)
        return ir.Trace(
            "step" if mark == "step" else "line",
            x=sorted_rows[x].to_list(),
            y=sorted_rows[y].to_list(),
            name=name,
            style=style,
        )
    if mark == "scatter":
        sizes = None
        if size_column:
            magnitudes = (
                group_frame[size_column].fill_null(0).to_numpy().astype(float)
            )
            peak = max(float(np.max(magnitudes)), 1e-9)
            sizes = (6 + 30 * magnitudes / peak).tolist()
        return ir.Trace(
            "scatter",
            x=group_frame[x].to_list(),
            y=group_frame[y].to_list(),
            name=name,
            size=sizes,
            style=style,
        )
    categories = [str(v) for v in group_frame[x].to_list()]
    values = group_frame[y].to_list()
    return ir.Trace(
        mark,
        x=categories if mark == "bar" else values,
        y=values if mark == "bar" else categories,
        name=name,
        style=style,
    )


def _distribution_traces(
    mark: str,
    group_frame: pl.DataFrame,
    category_column: str | None,
    value_column: str,
    options: dict[str, Any],
    name: str | None,
    style: dict[str, Any],
    first_color_index: int,
) -> list[ir.Trace]:
    """Builds box or histogram traces of one column, a box per category.

    Args:
        mark: `box` or `hist`.
        group_frame: Rows of one colour series.
        category_column: Column whose values each get a box, if any.
        value_column: Column holding the values.
        options: Resolved options of the kind (reads `bins`).
        name: Series name shown in the legend.
        style: Mark options shared by the series.
        first_color_index: Colour index of the first box per category.

    Returns:
        One histogram, one box, or one box per category.
    """
    if mark == "hist":
        return [
            ir.Trace(
                "hist",
                y=group_frame[value_column].to_list(),
                name=name,
                style={**style, "bins": options["bins"]},
            )
        ]
    if category_column is None:
        return [
            ir.Trace(
                "box",
                y=group_frame[value_column].to_list(),
                name=name or value_column,
                style=style,
            )
        ]
    traces = []
    categories = (
        group_frame[category_column].unique(maintain_order=True).to_list()
    )
    for offset, category in enumerate(categories):
        in_category = group_frame.filter(pl.col(category_column) == category)
        traces.append(
            ir.Trace(
                "box",
                y=in_category[value_column].to_list(),
                name=f"{category}" if name is None else f"{name}/{category}",
                style={"ci": first_color_index + offset},
            )
        )
    return traces
