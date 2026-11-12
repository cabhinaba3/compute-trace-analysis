"""Small helpers shared by plot kinds."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz.ir import Chart, Panel
from cta.viz.theme import BAD, GOOD


def single(title: str, panel: Panel, **kw: Any) -> Chart:
    panel.title = panel.title or title
    return Chart(title=title, panels=[panel], **kw)


def grid(title: str, panels: list[Panel], max_cols: int = 3, **kw: Any) -> Chart:
    return Chart(title=title, panels=panels, cols=min(max_cols, max(1, len(panels))), **kw)


def label_index(labels: list[str]) -> dict[str, int]:
    return {lb: i for i, lb in enumerate(labels)}


def rolling_median(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) < 2:
        return y
    h = k // 2
    return np.array([np.median(y[max(0, i - h) : i + h + 1]) for i in range(len(y))])


def mad_sigma(y: np.ndarray) -> float:
    """Robust sigma: 1.4826 x the median absolute deviation."""
    return float(1.4826 * np.median(np.abs(y - np.median(y)))) if len(y) else 0.0


def trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Theil-Sen slope, intercept and Kendall-tau p-value of y against x (needs >= 4 points)."""
    if len(x) < 4 or np.ptp(y) == 0:
        return None
    from scipy import stats as sp_stats  # lazy: scipy.stats costs >1 s to import

    slope, intercept, _, _ = sp_stats.theilslopes(y, x)
    p = float(sp_stats.kendalltau(x, y).pvalue)
    return float(slope), float(intercept), p


def delta_color(delta: float) -> str:
    return BAD if delta > 0 else GOOD


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def ordered(df: pl.DataFrame, col: str) -> list[Any]:
    """Distinct values of ``col`` in first-seen order."""
    return df[col].unique(maintain_order=True).to_list()
