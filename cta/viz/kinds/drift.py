"""Plot kind ``history``: one metric across many saved ``report.json`` files.

Robust control limits (median +/- k MAD sigmas) flag the runs that regressed
or improved.
"""

from __future__ import annotations

import datetime as datetime_module
import glob
import json
import pathlib
from typing import Any

import numpy as np

from cta.viz import data
from cta.viz import ir
from cta.viz import spec
from cta.viz import theme
from cta.viz.kinds import chart_helpers


def load_history(pattern: str) -> list[dict[str, Any]]:
    """Loads the saved reports that match a glob, oldest first.

    ``created_at`` falls back to the file's mtime for old reports. Files that
    cannot be read or parsed, and JSON documents without ``aspects``, are
    skipped.

    Args:
        pattern: Glob of ``report.json`` files; ``**`` matches recursively.

    Returns:
        One dict per report, sorted by ``when``, with keys ``path`` (the
        file), ``when`` (creation time), ``doc`` (the parsed JSON) and
        ``name`` (the parent directory name, else the file stem).
    """
    runs = []
    for path_str in sorted(glob.glob(pattern, recursive=True)):
        report_path = pathlib.Path(path_str)
        try:
            report_doc = json.loads(report_path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(report_doc, dict) or "aspects" not in report_doc:
            continue
        created_at = report_doc.get("created_at")
        if created_at:
            when = datetime_module.datetime.fromisoformat(created_at)
        else:
            when = datetime_module.datetime.fromtimestamp(
                report_path.stat().st_mtime, tz=datetime_module.UTC
            )
        runs.append(
            {
                "path": report_path,
                "when": when,
                "doc": report_doc,
                "name": report_path.parent.name or report_path.stem,
            }
        )
    return sorted(runs, key=lambda run: run["when"])


def _metric_value(
    report_doc: dict[str, Any],
    metric: str,
    aspect: str | None,
    series: str | None,
) -> float | None:
    """Reads one summary metric of one group from a saved report.

    Args:
        report_doc: Parsed ``report.json``.
        metric: Summary metric name.
        aspect: Only look in this aspect, if given.
        series: Group label to read; by default the candidate of a pair,
            else the first group.

    Returns:
        The value from the first matching row that has one, else None.
    """
    labels: list[str] = report_doc.get("groups", [])
    for aspect_name, metric_rows in report_doc["aspects"].items():
        if aspect and aspect_name != aspect:
            continue
        for row in metric_rows:
            if row.get("metric") != metric:
                continue
            column: str | None
            if len(labels) == 2 and "candidate" in row:
                column = "baseline" if series == labels[0] else "candidate"
            elif series in row:
                column = series
            elif labels and labels[0] in row:
                column = labels[0]
            else:
                column = None
            if column and row.get(column) is not None:
                return float(row[column])
    return None


@spec.register_plot(
    "history",
    group="drift",
    title="Metric across saved runs (control chart)",
    needs="--history GLOB of report.json files (or option history)",
    options={
        "history": spec.Opt(
            "str",
            None,
            "Glob of report.json files, e.g. 'reports/*/report.json'.",
        ),
        "metric": spec.Opt("str", "mean_step_ms", "Summary metric to track."),
        "aspect": spec.Opt("str", None, "Restrict to one aspect."),
        "series": spec.Opt(
            "str",
            None,
            "Group label to track (default: the candidate of a pair, else the"
            " first group).",
        ),
        "k": spec.Opt(
            "float", 3.5, "Control limits at median +/- k robust sigmas (MAD)."
        ),
        "worse": spec.Opt(
            "str",
            "higher",
            "Which direction is a regression.",
            ("higher", "lower"),
        ),
        "order_by": spec.Opt(
            "str",
            "created_at",
            "created_at (file mtime for old reports) or name.",
            ("created_at", "name"),
        ),
    },
)
def history(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Tracks a metric over saved reports as a control chart.

    Points outside the robust control band are flagged as regressions or
    improvements.

    Args:
        ctx: Data access; its ``history`` glob is used unless the `history`
            option is set.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart; its notes give the latest change against the
        median and any significant trend.

    Raises:
        spec.NotApplicableError: No history glob is given, or fewer than three
            reports carry the metric.
    """
    pattern = options.get("history") or ctx.history
    if not pattern:
        raise spec.NotApplicableError(
            "history needs --history 'reports/*/report.json'"
        )
    runs = load_history(pattern)
    if options["order_by"] == "name":
        runs.sort(key=lambda run: str(run["path"]))
    metric = options["metric"]
    run_values = [
        (
            run,
            _metric_value(
                run["doc"], metric, options.get("aspect"), options.get("series")
            ),
        )
        for run in runs
    ]
    points = [(run, value) for run, value in run_values if value is not None]
    if len(points) < 3:
        raise spec.NotApplicableError(
            f"found {len(points)} report(s) with metric '{metric}' under"
            f" {pattern}; need at least 3"
        )
    is_time_ordered = options["order_by"] == "created_at"
    if is_time_ordered:
        x = [run["when"] for run, _ in points]
    else:
        x = [run["name"] for run, _ in points]
    y = np.array([value for _, value in points])

    limit_sigmas = options["k"]
    center, sigma = float(np.median(y)), chart_helpers.mad_sigma(y)
    lower, upper = center - limit_sigmas * sigma, center + limit_sigmas * sigma
    sign = 1.0 if options["worse"] == "higher" else -1.0
    if sigma > 0:
        is_regression = sign * (y - center) > limit_sigmas * sigma
        is_improvement = sign * (center - y) > limit_sigmas * sigma
    else:
        is_regression = np.zeros(len(y), bool)
        is_improvement = np.zeros(len(y), bool)

    traces = [
        ir.Trace(
            "band",
            x=x,
            y=[lower] * len(y),
            y1=[upper] * len(y),
            name=f"median +/- {limit_sigmas:g} sigma",
            color=theme.NEUTRAL,
        ),
        ir.Trace(
            "line",
            x=x,
            y=y.tolist(),
            name=metric,
            style={"ci": 0, "marker": "o"},
        ),
        ir.Trace(
            "hline", y=[center], color=theme.NEUTRAL, style={"nolegend": True}
        ),
    ]
    if is_regression.any():
        traces.append(
            _flagged_points(x, y, is_regression, "regression", theme.BAD)
        )
    if is_improvement.any():
        traces.append(
            _flagged_points(x, y, is_improvement, "improvement", "#009E73")
        )

    notes = []
    if center:
        change_pct = 100 * (y[-1] - center) / center
        notes.append(
            f"{len(points)} runs; latest {y[-1]:.4g} is {change_pct:+.1f}%"
            f" vs median {center:.4g}"
        )
    fit = chart_helpers.trend(np.arange(len(y), dtype=float), y)
    if fit and fit[2] < 0.05:
        slope, _, p_value = fit
        notes.append(f"trend {slope:+.3g} per run (Kendall p={p_value:.2g})")
    return chart_helpers.single(
        f"{metric} across {len(points)} runs",
        ir.Panel(
            traces=traces,
            xlabel="run" if is_time_ordered else "report",
            ylabel=metric,
        ),
        notes="; ".join(notes),
    )


def _flagged_points(
    x: list[Any], y: np.ndarray, is_flagged: np.ndarray, name: str, color: str
) -> ir.Trace:
    """Builds a scatter trace of the runs selected by a boolean mask."""
    return ir.Trace(
        "scatter",
        x=[
            run_x
            for run_x, flagged in zip(x, is_flagged, strict=True)
            if flagged
        ],
        y=y[is_flagged].tolist(),
        name=name,
        color=color,
        style={"s": 60},
    )
