"""Small chart-building and statistics helpers shared by the plot kinds."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

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


def label_index(labels: list[str]) -> dict[str, int]:
    """Maps each label to its position in `labels`.

    Args:
        labels: Group labels.

    Returns:
        The index of every label; a repeated label keeps its last position.
    """
    return {label: i for i, label in enumerate(labels)}


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
    from scipy import stats as sp_stats

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
