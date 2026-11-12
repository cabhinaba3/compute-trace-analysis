from __future__ import annotations

from pathlib import Path
from typing import Any

from cta.core.models import ComparisonReport


def _tex_escape(val: str) -> str:
    return (
        val.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def export_latex(
    report: ComparisonReport,
    path: Path | None = None,
    *,
    aspect: str | None = None,
    caption: str = "CTA Trace Performance Comparison",
    label: str = "tab:cta_comparison",
    **_: Any,
) -> str:
    targets = [report.aspect(aspect)] if aspect else list(report.aspects.values())
    blocks: list[str] = []
    for asp in targets:
        df = asp.summary
        if df.is_empty():
            continue
        cols = [
            c
            for c in df.columns
            if not c.startswith("ci95_") and c not in ("p_mannwhitney", "cohens_d", "baseline_median", "candidate_median")
        ]
        spec = "ll" + "r" * max(0, len(cols) - 2)
        header = " & ".join(f"\\textbf{{{_tex_escape(c)}}}" for c in cols) + " \\\\"
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{_tex_escape(caption)} --- {_tex_escape(asp.title)}}}",
            f"\\label{{{label}_{asp.name}}}",
            f"\\begin{{tabular}}{{{spec}}}",
            "\\toprule",
            header,
            "\\midrule",
        ]
        for row in df.select(cols).iter_rows(named=True):
            cells: list[str] = []
            for k, v in row.items():
                if v is None:
                    cells.append("--")
                elif isinstance(v, bool):
                    cells.append("Yes" if v else "No")
                elif isinstance(v, float):
                    if k.endswith("delta_pct"):
                        s = f"{v:+.2f}\\%"
                        cells.append(f"\\textbf{{{s}}}" if abs(v) >= 5.0 else s)
                    elif k == "speedup":
                        cells.append(f"{v:.2f}$\\times$")
                    else:
                        cells.append(f"{v:.3g}")
                else:
                    cells.append(_tex_escape(str(v)))
            lines.append(" & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        blocks.append("\n".join(lines))

    content = "\n\n".join(blocks)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return content
