from __future__ import annotations

from pathlib import Path

from cta.core.models import ComparisonReport


def export_markdown(report: ComparisonReport, path: Path | None = None) -> str:
    heading = "CTA Trace Analysis Report" if report.mode == "single" else "CTA Comparison Report"
    lines = [f"# {heading} ({' vs '.join(g.label for g in report.groups)})", ""]
    if report.diagnoses:
        lines.append("## Bottleneck Diagnostics")
        for d in report.diagnoses:
            lines.append(f"- {d.summary}")
        lines.append("")
    if report.violations:
        lines.append("## Regression Gate Violations")
        for v in report.violations:
            lines.append(f"- **{v.metric}** ({v.aspect}): {v.message}")
        lines.append("")

    for aspect in report.aspects.values():
        df = aspect.summary
        if df.is_empty():
            continue
        lines.append(f"## {aspect.title} (`{aspect.name}`)")
        cols = [c for c in df.columns if not c.startswith("ci95_")]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in df.select(cols).iter_rows():
            formatted = [
                "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
                for v in row
            ]
            lines.append("| " + " | ".join(formatted) + " |")
        lines.append("")

    content = "\n".join(lines)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return content
