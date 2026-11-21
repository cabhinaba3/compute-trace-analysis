"""Links between charts: one selection that every chart of a run follows.

A :class:`Selection` picks up to two things, and :func:`apply` applies it to a
chart after the plot kind has built it, so it works the same in the
Matplotlib and Plotly backends, the dashboard, the live server and
``cta plot``. Nothing here runs in a browser.

* **A step** (index in ``step_trace``). Every chart with a time axis
  (``Panel.xkind == "time"``: the GPU timeline, kernels running at once, API
  calls and kernels) zooms to that step. The charts share one time base, ms
  since the rank's origin, so a step's window is its ``t_ms`` and ``end_ms`` in
  the steps table. A chart that did not draw at least half of the step is left
  alone and says so in its notes.
* **A kernel family.** Every series that is one op family (``meta["family"]``)
  or bar of a category axis of families (``meta["family_axis"]``) other than
  the picked one is dimmed, in all charts that carry the tags.

The plot kinds tag what a selection can pick (``Trace.meta``); plugin kinds
can use the same tags.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np
import polars as pl

from cta.viz import registry

# Windows of at most this many (label, trial, rank, step) rows are used; a run
# of hundreds of ranks with more steps than this does not pick a step.
MAX_WINDOW_ROWS = 200_000

# Opacity factor of the families that are not picked.
DIM = 0.25

# Legend entries and category labels are cut to this many characters.
_NAME_LENGTH = 36

# A step is zoomed to when at least this share of it was drawn.
_MIN_DRAWN = 0.5


@dataclasses.dataclass(frozen=True, slots=True)
class StepPick:
    """One step of one rank.

    Attributes:
        idx: The step's index among the rank's kept steps, as ``step_trace``
            plots it.
        rank: The rank; None takes the first rank that has the step.
        trial: The trial; None takes the first one that has the step.
        label: The group label; None takes each chart's own label.
    """

    idx: int
    rank: int | None = None
    trial: int | None = None
    label: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Selection:
    """What the charts are linked on.

    Attributes:
        step: The picked step, if any.
        family: The picked kernel family, if any.
    """

    step: StepPick | None = None
    family: str | None = None

    def __bool__(self) -> bool:
        """Returns whether anything is picked."""
        return self.step is not None or self.family is not None

    def describe(self) -> str:
        """Returns the selection as text, for the charts' notes."""
        parts: list[str] = []
        if self.step is not None:
            where = (
                f" (rank {self.step.rank})"
                if self.step.rank is not None
                else ""
            )
            parts.append(f"step {self.step.idx}{where}")
        if self.family is not None:
            parts.append(f"family {self.family}")
        return " · ".join(parts)


def parse_step(text: str) -> StepPick:
    """Reads a step pick from its command-line form.

    Args:
        text: ``IDX`` or ``IDX@RANK`` (for example ``4`` or ``4@1``).

    Returns:
        The pick.

    Raises:
        ValueError: If the text is not of that form or a number is negative.
    """
    idx_text, at, rank_text = text.strip().partition("@")
    try:
        idx = int(idx_text)
        rank = int(rank_text) if at else None
    except ValueError:
        raise ValueError(
            f"A step is IDX or IDX@RANK, for example 4 or 4@1; got {text!r}"
        ) from None
    if idx < 0 or (rank is not None and rank < 0):
        raise ValueError(f"A step and its rank are not negative; got {text!r}")
    return StepPick(idx=idx, rank=rank)


def step_windows(ctx: Any) -> list[dict[str, Any]]:
    """Returns every step's window in the time base of the time-axis charts.

    Args:
        ctx: The ``DataContext`` the charts draw from.

    Returns:
        One entry per (label, trial, rank): ``label``, ``trial``, ``rank`` and
        the columns ``idx`` (the step's index among the rank's kept steps, as
        ``step_trace`` plots it), ``t0`` and ``t1`` (ms since the rank's
        origin). Lowest ranks first; empty when the report has no steps or
        has more than ``MAX_WINDOW_ROWS`` of them.
    """
    try:
        steps = ctx.frame("steps")
    except registry.NotApplicableError:
        return []
    if steps.height == 0 or steps.height > MAX_WINDOW_ROWS:
        return []
    out: list[dict[str, Any]] = []
    ordered = steps.sort("label", "trial", "rank", "idx", maintain_order=True)
    for (label, trial, rank), part in ordered.group_by(
        ["label", "trial", "rank"], maintain_order=True
    ):
        out.append(
            {
                "label": str(label),
                "trial": int(trial),
                "rank": int(rank),
                "idx": part["idx"].cast(pl.Int64).to_list(),
                "t0": [round(v, 4) for v in part["t_ms"].to_list()],
                "t1": [round(v, 4) for v in part["end_ms"].to_list()],
            }
        )
    return out


def find_window(
    windows: Sequence[dict[str, Any]], pick: StepPick, label: str | None = None
) -> tuple[float, float] | None:
    """Finds the window of a picked step.

    The most exact match wins: the rank and trial of the pick, then its rank,
    then any rank that has the step. A pick with no rank or trial matches any.

    Args:
        windows: The result of :func:`step_windows`.
        pick: The step.
        label: The group label of the chart; None uses the pick's own, and
            with neither any label matches.

    Returns:
        ``(t0, t1)`` in ms, or None if no series has that step.
    """
    want = label if label is not None else pick.label
    scoped = [s for s in windows if want is None or s["label"] == want]
    tiers = (
        lambda s: (
            (pick.rank is None or s["rank"] == pick.rank)
            and (pick.trial is None or s["trial"] == pick.trial)
        ),
        lambda s: pick.rank is None or s["rank"] == pick.rank,
        lambda s: True,
    )
    for accept in tiers:
        for series in scoped:
            if not accept(series):
                continue
            idx = series["idx"]
            # idx is sorted, but a plain scan is as fast at this size.
            try:
                k = idx.index(pick.idx)
            except ValueError:
                continue
            return float(series["t0"][k]), float(series["t1"][k])
    return None


def same_family(a: Any, b: Any) -> bool:
    """Returns whether two family names are the same once cut for display."""
    return str(a)[:_NAME_LENGTH] == str(b)[:_NAME_LENGTH]


def _extent(traces: Sequence[Any]) -> tuple[float, float] | None:
    """Returns the lowest and highest x a panel drew (segments to their end)."""
    low, high = math.inf, -math.inf
    for trace in traces:
        for values in (trace.x, trace.x1):
            if values is None:
                continue
            array = np.asarray(values, dtype=float)
            array = array[np.isfinite(array)]
            if array.size:
                low = min(low, float(array.min()))
                high = max(high, float(array.max()))
    return (low, high) if low <= high else None


def _drawn(
    extent: tuple[float, float] | None, window: tuple[float, float]
) -> bool:
    """Returns whether a panel drew enough of a window to zoom to it."""
    if extent is None:
        return False
    length = window[1] - window[0]
    if length <= 0:
        return extent[0] <= window[0] <= extent[1]
    overlap = min(extent[1], window[1]) - max(extent[0], window[0])
    return overlap / length >= _MIN_DRAWN


def _dim_family(chart: Any, family: str) -> None:
    """Dims the traces of every family but ``family``, in place."""
    for panel in chart.panels:
        for trace in panel.traces:
            meta = trace.meta or {}
            base = trace.style.get("alpha", 1.0)
            if "family" in meta:
                if not same_family(meta["family"], family):
                    # A new dict: kinds may share one style between traces.
                    trace.style = {**trace.style, "alpha": base * DIM}
            elif meta.get("family_axis") in ("x", "y"):
                categories = trace.x if meta["family_axis"] == "x" else trace.y
                if categories is not None:
                    trace.style = {
                        **trace.style,
                        "alpha": [
                            base if same_family(c, family) else base * DIM
                            for c in categories
                        ],
                    }


def _zoom_step(
    chart: Any, pick: StepPick, windows: Sequence[dict[str, Any]]
) -> list[str]:
    """Zooms the time-axis panels to a step, in place.

    Returns:
        A note for each panel that did not draw the step.
    """
    notes: list[str] = []
    for panel in chart.panels:
        if panel.xkind != "time":
            continue
        labels = {t.meta.get("label") for t in panel.traces if t.meta}
        labels.discard(None)
        window = find_window(
            windows, pick, str(next(iter(labels))) if len(labels) == 1 else None
        )
        if window is not None and _drawn(_extent(panel.traces), window):
            panel.xrange = window
        else:
            where = f" ({panel.title})" if panel.title else ""
            notes.append(
                f"Step {pick.idx} is not in what this chart drew{where}."
            )
    return notes


def apply(chart: Any, selection: Selection, ctx: Any) -> Any:
    """Applies a selection to a chart, in place.

    Args:
        chart: The ``ir.Chart`` a plot kind built.
        selection: What to pick.
        ctx: The ``DataContext`` the chart was drawn from (its steps give the
            step windows).

    Returns:
        ``chart``, with the picked step's window as the x range of its time
        axes, the other families dimmed, and the selection and any panel that
        could not follow it named in the notes.
    """
    if not selection:
        return chart
    notes = [f"Selection: {selection.describe()}."]
    if selection.family is not None:
        _dim_family(chart, selection.family)
    if selection.step is not None and any(
        panel.xkind == "time" for panel in chart.panels
    ):
        notes.extend(_zoom_step(chart, selection.step, step_windows(ctx)))
    chart.notes = f"{chart.notes} {' '.join(notes)}".strip()
    return chart


def check(ctx: Any, selection: Selection) -> None:
    """Checks that a selection names a step the report has.

    Args:
        ctx: The ``DataContext`` the charts draw from.
        selection: The selection to check.

    Raises:
        ValueError: If the picked step is in no (label, trial, rank) series,
            or the report has no steps.
    """
    if selection.step is None:
        return
    windows = step_windows(ctx)
    if not windows:
        raise ValueError(
            "A step can be picked only on a report with step windows "
            f"(at most {MAX_WINDOW_ROWS} steps)."
        )
    pick = selection.step
    # Charts fall back to another rank for a group that lacks the picked one,
    # but a rank named on the command line has to exist.
    exact = [
        s
        for s in windows
        if (pick.rank is None or s["rank"] == pick.rank)
        and (pick.trial is None or s["trial"] == pick.trial)
    ]
    if find_window(exact, pick) is None:
        raise ValueError(
            f"Step {selection.step.idx} is not in the report"
            + (
                f" for rank {selection.step.rank}"
                if selection.step.rank is not None
                else ""
            )
            + "."
        )
