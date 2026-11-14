"""Exports aspect summaries as LaTeX booktabs tables."""

from __future__ import annotations

import pathlib
from typing import Any

from cta.core import models

_TEX_SPECIAL = {
    "\\": "\\textbackslash{}",
    "{": "\\{",
    "}": "\\}",
    "$": "\\$",
    "&": "\\&",
    "#": "\\#",
    "^": "\\textasciicircum{}",
    "_": "\\_",
    "~": "\\textasciitilde{}",
    "%": "\\%",
}
# Summary columns left out of the LaTeX tables (besides every ci95_* one).
_OMITTED_COLUMNS = (
    "p_mannwhitney",
    "cohens_d",
    "baseline_median",
    "candidate_median",
)


def _tex_escape(text: str) -> str:
    """Escapes the characters that are special in LaTeX text."""
    return "".join(_TEX_SPECIAL.get(char, char) for char in text)


def _format_cell(column: str, value: Any) -> str:
    """Formats one summary value as LaTeX table cell content.

    Changes of 5% or more in a ``*delta_pct`` column are set in bold.
    """
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if column.endswith("delta_pct"):
            pct_text = f"{value:+.2f}\\%"
            return f"\\textbf{{{pct_text}}}" if abs(value) >= 5.0 else pct_text
        if column == "speedup":
            return f"{value:.2f}$\\times$"
        return f"{value:.3g}"
    return _tex_escape(str(value))


def export_latex(
    report: models.ComparisonReport,
    path: pathlib.Path | None = None,
    *,
    aspect: str | None = None,
    caption: str = "CTA Trace Performance Comparison",
    label: str = "tab:cta_comparison",
    **_: Any,
) -> str:
    """Renders aspect summaries as LaTeX booktabs tables.

    One ``table`` environment per aspect with a non-empty summary, without
    the CI, Mann-Whitney, Cohen's d and median columns.

    Args:
        report: The report to export.
        path: File to write the LaTeX to; its parent directories are
            created. None only returns the text.
        aspect: Name of the only aspect to export; None exports all.
        caption: Caption prefix; the aspect title is appended.
        label: Label prefix; ``_<aspect name>`` is appended.
        **_: Ignored.

    Returns:
        The LaTeX source, tables separated by a blank line.

    Raises:
        KeyError: ``aspect`` is not in the report.
    """
    aspect_results = (
        [report.aspect(aspect)] if aspect else list(report.aspects.values())
    )
    blocks: list[str] = []
    for aspect_result in aspect_results:
        summary = aspect_result.summary
        if summary.is_empty():
            continue
        columns = [
            column
            for column in summary.columns
            if not column.startswith("ci95_") and column not in _OMITTED_COLUMNS
        ]
        column_spec = "ll" + "r" * max(0, len(columns) - 2)
        header = (
            " & ".join(
                f"\\textbf{{{_tex_escape(column)}}}" for column in columns
            )
            + " \\\\"
        )
        title = _tex_escape(aspect_result.title)
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{_tex_escape(caption)} --- {title}}}",
            f"\\label{{{label}_{aspect_result.name}}}",
            f"\\begin{{tabular}}{{{column_spec}}}",
            "\\toprule",
            header,
            "\\midrule",
        ]
        for row in summary.select(columns).iter_rows(named=True):
            cells = [
                _format_cell(column, value) for column, value in row.items()
            ]
            lines.append(" & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        blocks.append("\n".join(lines))

    content = "\n\n".join(blocks)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return content
