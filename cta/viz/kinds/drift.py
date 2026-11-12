"""Across-run history: one metric over many saved ``report.json`` files with robust control limits."""

from __future__ import annotations

import glob
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cta.viz.data import DataContext
from cta.viz.ir import Chart, Panel, Trace
from cta.viz.kinds._util import mad_sigma, single, trend
from cta.viz.spec import NotApplicable, Opt, register_plot
from cta.viz.theme import BAD, NEUTRAL


def load_history(pattern: str) -> list[dict[str, Any]]:
    """Saved reports matching a glob, oldest first. ``created_at`` falls back to the file's mtime for old reports."""
    out = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        p = Path(f)
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or "aspects" not in doc:
            continue
        stamp = doc.get("created_at")
        when = datetime.fromisoformat(stamp) if stamp else datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
        out.append({"path": p, "when": when, "doc": doc, "name": p.parent.name or p.stem})
    return sorted(out, key=lambda r: r["when"])


def _value(doc: dict[str, Any], metric: str, aspect: str | None, series: str | None) -> float | None:
    labels: list[str] = doc.get("groups", [])
    for name, rows in doc["aspects"].items():
        if aspect and name != aspect:
            continue
        for r in rows:
            if r.get("metric") != metric:
                continue
            col: str | None
            if len(labels) == 2 and "candidate" in r:
                col = "baseline" if series == labels[0] else "candidate"
            else:
                col = series if series in r else (labels[0] if labels and labels[0] in r else None)
            if col and r.get(col) is not None:
                return float(r[col])
    return None


@register_plot(
    "history", group="drift", title="Metric across saved runs (control chart)", needs="--history GLOB of report.json files (or option history)",
    options={
        "history": Opt("str", None, "Glob of report.json files, e.g. 'reports/*/report.json'."),
        "metric": Opt("str", "mean_step_ms", "Summary metric to track."),
        "aspect": Opt("str", None, "Restrict to one aspect."),
        "series": Opt("str", None, "Group label to track (default: the candidate of a pair, else the first group)."),
        "k": Opt("float", 3.5, "Control limits at median +/- k robust sigmas (MAD)."),
        "worse": Opt("str", "higher", "Which direction is a regression.", ("higher", "lower")),
        "order_by": Opt("str", "created_at", "created_at (file mtime for old reports) or name.", ("created_at", "name")),
    },
)  # fmt: skip
def history(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Tracks a metric over saved reports; points outside the robust control band are flagged as regressions or improvements."""
    pattern = o.get("history") or ctx.history
    if not pattern:
        raise NotApplicable("history needs --history 'reports/*/report.json'")
    runs = load_history(pattern)
    if o["order_by"] == "name":
        runs.sort(key=lambda r: str(r["path"]))
    pts = [(r, _value(r["doc"], o["metric"], o.get("aspect"), o.get("series"))) for r in runs]
    pts = [(r, v) for r, v in pts if v is not None]
    if len(pts) < 3:
        raise NotApplicable(f"found {len(pts)} report(s) with metric '{o['metric']}' under {pattern}; need at least 3")
    x = [r["when"] for r, _ in pts] if o["order_by"] == "created_at" else [r["name"] for r, _ in pts]
    y = np.array([v for _, v in pts])
    center, sig = float(np.median(y)), mad_sigma(y)
    lo, hi = center - o["k"] * sig, center + o["k"] * sig
    sign = 1.0 if o["worse"] == "higher" else -1.0
    bad = sign * (y - center) > o["k"] * sig if sig > 0 else np.zeros(len(y), bool)
    good = sign * (center - y) > o["k"] * sig if sig > 0 else np.zeros(len(y), bool)
    traces = [
        Trace("band", x=x, y=[lo] * len(y), y1=[hi] * len(y), name=f"median +/- {o['k']:g} sigma", color=NEUTRAL),
        Trace("line", x=x, y=y.tolist(), name=o["metric"], style={"ci": 0, "marker": "o"}),
        Trace("hline", y=[center], color=NEUTRAL, style={"nolegend": True}),
    ]
    if bad.any():
        traces.append(Trace("scatter", x=[xi for xi, b in zip(x, bad, strict=True) if b], y=y[bad].tolist(), name="regression", color=BAD, style={"s": 60}))
    if good.any():
        traces.append(Trace("scatter", x=[xi for xi, b in zip(x, good, strict=True) if b], y=y[good].tolist(), name="improvement", color="#009E73", style={"s": 60}))
    notes = [f"{len(pts)} runs; latest {y[-1]:.4g} is {100 * (y[-1] - center) / center:+.1f}% vs median {center:.4g}" if center else ""]
    tr = trend(np.arange(len(y), dtype=float), y)
    if tr and tr[2] < 0.05:
        notes.append(f"trend {tr[0]:+.3g} per run (Kendall p={tr[2]:.2g})")
    return single(f"{o['metric']} across {len(pts)} runs", Panel(traces=traces, xlabel="run" if o["order_by"] == "created_at" else "report", ylabel=o["metric"]), notes="; ".join(n for n in notes if n))
