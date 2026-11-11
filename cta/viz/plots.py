"""Publication-grade Matplotlib visualizations (vector Type-42 PDF/SVG/PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from cta.core.models import ComparisonReport  # noqa: E402

# Okabe-Ito colorblind-safe academic palette
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#333333")

STYLE_SIZES: dict[str, tuple[float, float]] = {
    "paper_single_col": (3.5, 2.4),
    "paper_double_col": (7.16, 3.0),
    "presentation": (6.4, 4.0),
}

PAPER_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
}


def _extract_group_metric(df: pl.DataFrame, labels: list[str], metric: str) -> list[float]:
    sub = df.filter(pl.col("metric") == metric)
    if sub.is_empty():
        return [0.0] * len(labels)
    row = sub.row(0, named=True)
    if len(labels) == 2 and "baseline" in row and "candidate" in row:
        return [float(row["baseline"] or 0.0), float(row["candidate"] or 0.0)]
    return [float(row.get(lbl) or 0.0) for lbl in labels]


def _plot_breakdown(report: ComparisonReport, fig: Figure) -> None:
    ax = fig.add_subplot(111)
    labels = [g.label for g in report.groups]
    x = np.arange(len(labels))

    if "timeline_overlap" in report.aspects:
        df = report.aspects["timeline_overlap"].summary
        series = [
            ("Compute Only", _extract_group_metric(df, labels, "compute_only_ms")),
            ("Overlapped Comm", _extract_group_metric(df, labels, "overlapped_comm_ms")),
            ("Exposed Comm", _extract_group_metric(df, labels, "exposed_comm_ms")),
            ("GPU Idle", _extract_group_metric(df, labels, "gpu_idle_ms")),
        ]
        ylabel = "Time (ms)"
        title = "Timeline Execution Breakdown"
    elif "dag_critical_path" in report.aspects:
        df = report.aspects["dag_critical_path"].summary
        series = [
            ("CP Compute", _extract_group_metric(df, labels, "cp_compute_us")),
            ("CP Exposed Comm", _extract_group_metric(df, labels, "cp_exposed_comm_us")),
        ]
        ylabel = "Critical-Path Time (us)"
        title = "Critical-Path Composition"
    else:
        raise ValueError("Breakdown plot requires 'timeline_overlap' or 'dag_critical_path' aspect")

    bottom = np.zeros(len(labels), dtype=np.float64)
    for idx, (sname, vals) in enumerate(series):
        arr = np.asarray(vals, dtype=np.float64)
        ax.bar(x, arr, width=0.52, bottom=bottom, label=sname, color=OKABE_ITO[idx % len(OKABE_ITO)])
        bottom += arr

    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")


def _plot_tornado(report: ComparisonReport, fig: Figure, *, top_k: int = 10, table: str | None = None) -> None:
    if len(report.groups) not in (1, 2):
        raise ValueError("Tornado / Top-K plot requires 1 trace (hotspots) or 2 traces (Baseline vs Candidate)")
    is_single = len(report.groups) == 1
    val_col = report.groups[0].label if is_single else "delta"
    ax = fig.add_subplot(111)

    df: pl.DataFrame | None = None
    key_col = ""
    unit = ""
    candidates = [
        ("kernel_diff", table or "by_family", "op_family" if (table or "by_family") == "by_family" else "canonical_name", "ms"),
        ("layer_fusion", "by_layer", "name", "ms"),
        ("comm_collectives", "by_comm_type", "comm_type", "MB"),
        ("os_runtime", "by_syscall", "name", "ms"),
        ("roofline_diff", "by_kernel", "canonical_name", "us"),
    ]
    for asp_name, tbl_name, kcol, u in candidates:
        if asp_name in report.aspects and tbl_name in report.aspects[asp_name].details:
            cand_df = report.aspects[asp_name].details[tbl_name]
            if cand_df.height > 0 and val_col in cand_df.columns:
                df = cand_df.head(top_k)
                key_col = kcol if kcol in df.columns else df.columns[0]
                unit = u
                break

    if df is None:
        raise ValueError(f"No entity breakdown table with '{val_col}' available for Top-K / tornado plot")

    sub = df.reverse()
    names = [str(s)[:32] for s in sub[key_col].to_list()]
    vals = sub[val_col].to_numpy().astype(np.float64)
    colors = [OKABE_ITO[0]] * len(vals) if is_single else [OKABE_ITO[1] if d > 0 else OKABE_ITO[2] for d in vals]

    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.6)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, names)
    if is_single:
        ax.set_xlabel(f"{report.groups[0].label} [{unit}]")
        ax.set_title(f"Top Hotspots ({report.groups[0].label})")
    else:
        ax.set_xlabel(f"Delta ({report.groups[1].label} - {report.groups[0].label}) [{unit}]")
        ax.set_title("Top Contributors to Net Change")


def _plot_traffic_heatmap(report: ComparisonReport, fig: Figure) -> None:
    if "traffic_matrix" not in report.aspects:
        raise ValueError("Traffic heatmap requires 'traffic_matrix' aspect")
    asp = report.aspects["traffic_matrix"]
    labels = [g.label for g in report.groups]
    ax = fig.add_subplot(111)

    mats = [asp.details.get(f"matrix_{lbl}", pl.DataFrame()) for lbl in labels]
    max_r = max(
        (
            max(int(m["src_rank"].to_numpy().max()), int(m["dst_rank"].to_numpy().max())) + 1
            for m in mats
            if m.height
        ),
        default=2,
    )

    def _to_dense(mdf: pl.DataFrame) -> np.ndarray:
        grid = np.zeros((max_r, max_r), dtype=np.float64)
        if mdf.height:
            for r in mdf.iter_rows(named=True):
                grid[int(r["src_rank"]), int(r["dst_rank"])] = float(r["volume_mb"])
        return grid

    if len(labels) == 2:
        mat = _to_dense(mats[1]) - _to_dense(mats[0])
        vmax = max(1e-6, float(np.max(np.abs(mat))))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
        fig.colorbar(im, ax=ax, label="Delta Volume (MB)")
        ax.set_title(f"Traffic Delta ({labels[1]} - {labels[0]})")
    else:
        mat = _to_dense(mats[0])
        im = ax.imshow(mat, cmap="Blues", origin="upper")
        fig.colorbar(im, ax=ax, label="Volume (MB)")
        ax.set_title(f"Traffic Matrix ({labels[0]})")

    ax.set_xlabel("Destination Rank")
    ax.set_ylabel("Source Rank")


def _plot_skew(report: ComparisonReport, fig: Figure) -> None:
    if "rank_skew" not in report.aspects:
        raise ValueError("Skew plot requires 'rank_skew' aspect")
    asp = report.aspects["rank_skew"]
    ax = fig.add_subplot(111)
    labels = [g.label for g in report.groups]
    data = [
        asp.details[lbl]["compute_us"].to_numpy()
        for lbl in labels
        if lbl in asp.details and asp.details[lbl].height
    ]
    valid_labels = [lbl for lbl in labels if lbl in asp.details and asp.details[lbl].height]
    if not data:
        raise ValueError("No per-rank distribution data available for skew plot")

    bp = ax.boxplot(data, tick_labels=valid_labels, patch_artist=True)
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(OKABE_ITO[idx % len(OKABE_ITO)])
        patch.set_alpha(0.75)
    ax.set_ylabel("Per-Rank Compute Time (us)")
    ax.set_title("Multi-Rank Load Distribution")


def _plot_roofline(report: ComparisonReport, fig: Figure) -> None:
    if "roofline_diff" not in report.aspects:
        raise ValueError("Roofline plot requires 'roofline_diff' aspect")
    asp = report.aspects["roofline_diff"]
    labels = [g.label for g in report.groups]
    ax = fig.add_subplot(111)

    for idx, lbl in enumerate(labels):
        df = asp.details.get(f"roofline_{lbl}", pl.DataFrame())
        if df.is_empty():
            continue
        ai = np.maximum(1e-3, df["arith_intensity"].to_numpy())
        tf = np.maximum(1e-3, df["achieved_tflops"].to_numpy())
        ax.scatter(ai, tf, label=lbl, color=OKABE_ITO[idx % len(OKABE_ITO)], s=28, zorder=3)

    if len(labels) == 2:
        d0 = asp.details.get(f"roofline_{labels[0]}", pl.DataFrame())
        d1 = asp.details.get(f"roofline_{labels[1]}", pl.DataFrame())
        if d0.height and d1.height:
            joined = d0.join(d1, on="canonical_name", suffix="_cand")
            for row in joined.iter_rows(named=True):
                x0, y0 = max(1e-3, float(row["arith_intensity"])), max(1e-3, float(row["achieved_tflops"]))
                x1, y1 = max(1e-3, float(row["arith_intensity_cand"])), max(1e-3, float(row["achieved_tflops_cand"]))
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#444444"})

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / Byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title("Kernel Roofline Shift")
    ax.legend(frameon=False, loc="best")


def _plot_scaling(report: ComparisonReport, fig: Figure, *, metric: str | None = None) -> None:
    ax = fig.add_subplot(111)
    labels = [g.label for g in report.groups]

    target_asp = next(iter(report.aspects.values()), None)
    if target_asp is None or target_asp.summary.is_empty():
        raise ValueError("No summary metrics available for scaling plot")

    chosen_metric = metric
    for asp in report.aspects.values():
        if asp.summary.is_empty():
            continue
        m_list = asp.summary["metric"].to_list()
        if chosen_metric and chosen_metric in m_list:
            target_asp = asp
            break
        for pref in ("mean_step_ms", "critical_path_us", "throughput_per_s", "energy_per_query_mj", "total_energy_j"):
            if pref in m_list:
                target_asp = asp
                chosen_metric = pref
                break
        if chosen_metric:
            break

    if not chosen_metric:
        chosen_metric = str(target_asp.summary["metric"][0])

    vals = _extract_group_metric(target_asp.summary, labels, chosen_metric)
    unit_row = target_asp.summary.filter(pl.col("metric") == chosen_metric)
    unit = str(unit_row["unit"][0]) if unit_row.height else ""

    x = np.arange(len(labels))
    if len(labels) == 1:
        ax.bar(x, vals, width=0.45, color=OKABE_ITO[0])
    else:
        ax.plot(x, vals, marker="o", linewidth=1.6, color=OKABE_ITO[0])
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"{chosen_metric} ({unit})" if unit else chosen_metric)
    ax.set_title(f"Metric Summary: {chosen_metric}" if len(labels) == 1 else f"Scaling Comparison: {chosen_metric}")


def render_plot(
    report: ComparisonReport,
    *,
    kind: str = "breakdown",
    path: Path | None = None,
    style: str = "paper_single_col",
    dpi: int = 300,
    **kwargs: Any,
) -> Figure:
    figsize = STYLE_SIZES.get(style, STYLE_SIZES["paper_single_col"])
    with plt.rc_context(cast(Any, PAPER_RC)):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        k = kind.lower()
        if k == "breakdown":
            _plot_breakdown(report, fig)
        elif k in ("tornado", "topk", "top_k", "hotspots"):
            _plot_tornado(report, fig, **kwargs)
        elif k in ("traffic_delta", "heatmap", "traffic"):
            _plot_traffic_heatmap(report, fig)
        elif k == "skew":
            _plot_skew(report, fig)
        elif k == "roofline":
            _plot_roofline(report, fig)
        elif k == "scaling":
            _plot_scaling(report, fig, **kwargs)
        else:
            plt.close(fig)
            raise ValueError(
                f"Unknown plot kind '{kind}'. Supported: breakdown, tornado, topk, traffic_delta, skew, roofline, scaling"
            )

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        return fig
