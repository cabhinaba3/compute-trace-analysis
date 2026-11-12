from __future__ import annotations

import contextlib
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cta.core.models import ComparisonReport
from cta.exporters.latex import export_latex
from cta.exporters.markdown import export_markdown

# "tornado" is an alias of "topk", so it is not exported separately.
PLOT_KINDS = ("breakdown", "topk", "traffic_delta", "skew", "roofline", "scaling")


def _version() -> str:
    from cta import __version__

    return __version__


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def report_to_dict(report: ComparisonReport) -> dict[str, Any]:
    """JSON-serialisable view of a report. `aspects` keeps the legacy summary-only shape."""
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),  # lets `cta plot --kind history` order saved runs
        "cta_version": _version(),
        "git_commit": _git_commit(),
        "mode": report.mode,
        "groups": [g.label for g in report.groups],
        "diagnoses": [d.summary for d in report.diagnoses],
        "violations": [v.message for v in report.violations],
        "aspects": {name: asp.summary.to_dicts() for name, asp in report.aspects.items()},
        "details": {
            name: {t: df.to_dicts() for t, df in asp.details.items()}
            for name, asp in report.aspects.items()
            if asp.details
        },
        "scalars": {name: asp.scalars for name, asp in report.aspects.items() if asp.scalars},
    }


def report_to_json(report: ComparisonReport) -> str:
    return json.dumps(report_to_dict(report), indent=2, default=str)


def export_bundle(
    report: ComparisonReport,
    out_dir: Path,
    *,
    formats: Iterable[str] = ("markdown", "csv", "json"),
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    fmt_set = {f.lower() for f in formats}

    if "markdown" in fmt_set or "md" in fmt_set:
        p = out_dir / "report.md"
        export_markdown(report, p)
        written["markdown"] = p

    if "latex" in fmt_set or "tex" in fmt_set:
        p = out_dir / "report.tex"
        export_latex(report, p)
        written["latex"] = p

    if "csv" in fmt_set:
        for name, asp in report.aspects.items():
            if not asp.summary.is_empty():
                cp = out_dir / f"{name}_summary.csv"
                asp.summary.write_csv(cp)
                written[f"csv:{name}"] = cp

    if "json" in fmt_set:
        jp = out_dir / "report.json"
        jp.write_text(report_to_json(report))
        written["json"] = jp

    if "html" in fmt_set:
        with contextlib.suppress(ImportError, ValueError):  # plotly not installed, or nothing drawable: the dashboard is optional
            written["html"] = report.dashboard(out_dir / "dashboard.html")

    for ext in ("pdf", "svg", "png"):
        if ext in fmt_set:
            for kind in PLOT_KINDS:
                fig_path = out_dir / f"{kind}.{ext}"
                try:
                    report.plot(kind, path=fig_path)
                except ValueError:  # plot not applicable to this report's aspects
                    continue
                written[f"{ext}:{kind}"] = fig_path

    return written
