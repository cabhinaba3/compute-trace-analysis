from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from cta.core.models import ComparisonReport
from cta.exporters.latex import export_latex
from cta.exporters.markdown import export_markdown


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
        payload = {
            "mode": report.mode,
            "groups": [g.label for g in report.groups],
            "diagnoses": [d.summary for d in report.diagnoses],
            "violations": [v.message for v in report.violations],
            "aspects": {
                name: asp.summary.to_dicts()
                for name, asp in report.aspects.items()
            },
        }
        jp.write_text(json.dumps(payload, indent=2))
        written["json"] = jp

    if "pdf" in fmt_set or "png" in fmt_set or "svg" in fmt_set:
        ext = "pdf" if "pdf" in fmt_set else ("svg" if "svg" in fmt_set else "png")
        try:
            fig_path = out_dir / f"breakdown.{ext}"
            report.plot("breakdown", path=fig_path)
            written[ext] = fig_path
        except (NotImplementedError, ValueError):
            pass

    return written
