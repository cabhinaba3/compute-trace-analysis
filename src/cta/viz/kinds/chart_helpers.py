"""Small chart-building and statistics helpers shared by the plot kinds."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from cta.viz import ir
from cta.viz import theme

# Scales the median absolute deviation to the standard deviation of a normal
# distribution.
_MAD_TO_SIGMA = 1.4826


def single(title: str, panel: ir.Panel, **chart_fields: Any) -> ir.Chart:
    """Wraps one panel in a chart; the panel inherits the title if it has none.

    Args:
        title: Chart title, also used as the panel title when that is empty.
        panel: The only panel of the chart. Its title is set in place.
        **chart_fields: Further `ir.Chart` fields, such as `notes`.

    Returns:
        A one-panel chart.
    """
    panel.title = panel.title or title
    return ir.Chart(title=title, panels=[panel], **chart_fields)


def grid(
    title: str,
    panels: list[ir.Panel],
    max_cols: int = 3,
    **chart_fields: Any,
) -> ir.Chart:
    """Lays panels out in a grid of at most `max_cols` columns.

    Args:
        title: Chart title.
        panels: Panels in reading order.
        max_cols: Upper bound on the number of columns.
        **chart_fields: Further `ir.Chart` fields, such as `notes`.

    Returns:
        A chart with as many columns as panels, capped at `max_cols`.
    """
    return ir.Chart(
        title=title,
        panels=panels,
        cols=min(max_cols, max(1, len(panels))),
        **chart_fields,
    )


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Computes a centred rolling median, shrinking the window at both ends.

    Args:
        values: The series to smooth.
        window: Window length in samples; 1 or less turns smoothing off. Each
            median spans ``window // 2`` samples on either side, so an even
            window covers ``window + 1`` samples.

    Returns:
        The smoothed series, or `values` itself when smoothing is off or the
        series has fewer than two points.
    """
    if window <= 1 or len(values) < 2:
        return values
    half_window = window // 2
    return np.array(
        [
            np.median(values[max(0, i - half_window) : i + half_window + 1])
            for i in range(len(values))
        ]
    )


def mad_sigma(values: np.ndarray) -> float:
    """Computes a robust sigma: 1.4826 x the median absolute deviation.

    Args:
        values: The sample.

    Returns:
        The robust sigma, or 0.0 for an empty sample.
    """
    if not len(values):
        return 0.0
    deviations = np.abs(values - np.median(values))
    return float(_MAD_TO_SIGMA * np.median(deviations))


def trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Fits a Theil-Sen line to y against x and tests it with Kendall's tau.

    Args:
        x: Positions, such as step indices.
        y: Values at those positions.

    Returns:
        The slope, intercept and Kendall-tau p-value, or None when there are
        fewer than four points or y is constant.
    """
    if len(x) < 4 or np.ptp(y) == 0:
        return None
    # Lazy: scipy.stats costs more than a second to import.

    slope, intercept, _, _ = sp_stats.theilslopes(y, x)
    p_value = float(sp_stats.kendalltau(x, y).pvalue)
    return float(slope), float(intercept), p_value


def delta_color(delta: float) -> str:
    """Returns the theme colour for a change: bad when positive, else good.

    Args:
        delta: A change where positive means worse (e.g. more time).

    Returns:
        `theme.BAD` or `theme.GOOD`.
    """
    return theme.BAD if delta > 0 else theme.GOOD


def percentile(values: np.ndarray, q: float) -> float:
    """Computes the q-th percentile, or NaN for an empty array.

    Args:
        values: The sample.
        q: Percentile in [0, 100].

    Returns:
        The percentile as a float.
    """
    return float(np.percentile(values, q)) if len(values) else float("nan")


def unique_in_order(df: pl.DataFrame, column: str) -> list[Any]:
    """Returns the distinct values of a column in first-seen order.

    Args:
        df: Any frame.
        column: Name of the column to read.

    Returns:
        The distinct values, in the order they first appear.
    """
    return df[column].unique(maintain_order=True).to_list()


def merge_intervals(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merges intervals into the disjoint ranges that cover them.

    Intervals are half-open ``[start, end)``, so ones that only touch stay
    separate.

    Args:
        starts: Interval starts, in any order.
        ends: Interval ends, in the same order.

    Returns:
        The starts and ends of the merged ranges, in increasing order.
    """
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    if not len(starts):
        return starts, ends
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    reach = np.maximum.accumulate(ends)
    # A new range begins where an interval starts at or after everything
    # before it has ended.
    begins = np.concatenate([[True], starts[1:] >= reach[:-1]])
    first = np.flatnonzero(begins)
    last = np.append(first[1:] - 1, len(starts) - 1)
    return starts[first], reach[last]


def covered_length(
    merged_starts: np.ndarray,
    merged_ends: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> np.ndarray:
    """Measures how much of each ``[low, high)`` the merged ranges cover.

    Args:
        merged_starts: Starts from ``merge_intervals``.
        merged_ends: Ends from ``merge_intervals``.
        lows: Starts of the intervals to measure.
        highs: Ends of the intervals to measure.

    Returns:
        The covered length of each interval.
    """
    if not len(merged_starts):
        return np.zeros(len(lows))
    lengths = np.concatenate([[0.0], np.cumsum(merged_ends - merged_starts)])

    def covered_before(point: np.ndarray) -> np.ndarray:
        """Length of the merged ranges that lie before each point."""
        index = np.searchsorted(merged_starts, point, side="right")
        inside = np.where(
            index > 0,
            np.clip(
                point - merged_starts[np.maximum(index - 1, 0)],
                0.0,
                merged_ends[np.maximum(index - 1, 0)]
                - merged_starts[np.maximum(index - 1, 0)],
            ),
            0.0,
        )
        return lengths[np.maximum(index - 1, 0)] + inside

    return covered_before(np.asarray(highs, float)) - covered_before(
        np.asarray(lows, float)
    )


def summary_values(
    table: pl.DataFrame, labels: list[str]
) -> dict[str, dict[str, float]]:
    """Reads an aspect summary as ``{metric: {group label: mean}}``.

    Args:
        table: A summary from ``compare_scalar_metrics``: ``baseline`` and
            ``candidate`` columns for two groups, one column per label
            otherwise.
        labels: The group labels, baseline first.

    Returns:
        The mean of every metric in every group the table has a column for.
    """
    if {"baseline", "candidate"} <= set(table.columns):
        columns = dict(zip(labels, ("baseline", "candidate"), strict=False))
    else:
        columns = {label: label for label in labels if label in table.columns}
    return {
        row["metric"]: {
            label: float(row[column]) for label, column in columns.items()
        }
        for row in table.iter_rows(named=True)
        if "metric" in row
    }


# Raw tables whose rows carry a time, for the kinds with a time axis.
TIMED_SOURCES = (
    "kernels",
    "gpu_metrics",
    "steps",
    "network_ops",
    "jtop",
)


# Chart ``meta`` that turns off the suitability checks for the chart.
SKIP_SUITABILITY = {"skip_suitability": True}
