from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cta.core.models import ComparisonReport


def render_terminal_report(report: ComparisonReport, console: Console | None = None) -> None:
    con = console or Console()
    lbls = " vs ".join(f"[bold cyan]{g.label}[/]" for g in report.groups)
    title_kind = "Analysis" if report.mode == "single" else f"Comparison ({report.mode.upper()})"
    con.print(Panel(f"CTA {title_kind}: {lbls}", expand=False))

    for shift in report.diagnoses:
        color = "yellow" if shift.baseline_regime != shift.candidate_regime else "green"
        con.print(f"[{color}]• {shift.summary}[/{color}]")

    for aspect in report.aspects.values():
        df = aspect.summary
        if df.is_empty():
            continue
        tbl = Table(title=f"{aspect.title} [{aspect.name}]", show_lines=False)
        for col in df.columns:
            if col.startswith("ci95_") or col in ("p_mannwhitney", "cohens_d"):
                continue
            tbl.add_column(col, justify="left" if col in ("metric", "unit") else "right")

        vis_cols = [c for c in df.columns if not c.startswith("ci95_") and c not in ("p_mannwhitney", "cohens_d")]
        for row in df.select(vis_cols).iter_rows(named=True):
            cells: list[str] = []
            for k, v in row.items():
                if v is None:
                    cells.append("-")
                elif isinstance(v, bool):
                    cells.append("yes" if v else "no")
                elif isinstance(v, float):
                    if k.endswith("delta_pct"):
                        color = "red" if v > 2.0 else ("green" if v < -2.0 else "white")
                        cells.append(f"[{color}]{v:+.2f}%[/{color}]")
                    elif k == "speedup":
                        cells.append(f"{v:.2f}x")
                    else:
                        cells.append(f"{v:.4g}")
                else:
                    cells.append(str(v))
            tbl.add_row(*cells)
        con.print(tbl)

    for v in report.violations:
        con.print(f"[bold red]✗ {v.message}[/bold red]")
