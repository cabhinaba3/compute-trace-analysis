"""Tests for violin plots, plot kinds catalog, and multi-backend rendering."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import polars as pl
import pytest

import cta
from cta import models
from cta import viz
from cta.viz import context
from cta.viz import ir
from cta.viz import plots
from cta.viz import spec as viz_spec
from cta.viz import suitability
from cta.viz.backends import matplotlib as mpl_backend
from cta.viz.backends import plotly as plotly_backend


def _trace(mark: str, **fields: Any) -> ir.Trace:
    """Builds a trace from fields of any type, such as arrays and Series."""
    return ir.Trace(mark, **fields)


def _kineto_trace(
    path: pathlib.Path, *, steps: int = 6, slowdown_us: float = 0.0
) -> pathlib.Path:
    """Writes a one-rank Kineto trace with a GEMM and NCCL kernel per step."""
    events = []
    for i in range(steps):
        step_start_us = i * 1000.0
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": f"ProfilerStep#{i}",
                "ts": step_start_us,
                "dur": 500.0 + slowdown_us * i,
                "pid": 1,
                "tid": 1,
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": step_start_us + 10.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm_kernel",
                "ts": step_start_us + 40.0,
                "dur": 200.0 + slowdown_us * i,
                "pid": 1,
                "tid": 1,
                "args": {"stream": 7, "device": 0},
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "ncclKernel_AllReduce",
                "ts": step_start_us + 260.0,
                "dur": 80.0,
                "pid": 1,
                "tid": 1,
                "args": {"stream": 7, "device": 0},
            }
        )
    import json

    path.write_text(json.dumps({"traceEvents": events}))
    return path


def _report_fixture(tmp_path: pathlib.Path) -> models.Report:
    """Returns a test comparison report with 6 steps and 2 kernels per step."""
    base_path = _kineto_trace(tmp_path / "a.json")
    cand_path = _kineto_trace(tmp_path / "b.json", slowdown_us=20.0)
    return cta.compare(
        base_path, cand_path, labels=["base", "cand"], cache_dir=tmp_path / "c"
    )


def test_ir_mark_violin() -> None:
    assert "violin" in ir.MARKS
    trace = ir.Trace("violin", y=[1.0, 2.0, 3.0], name="test")
    assert trace.mark == "violin"
    with pytest.raises(ValueError, match="Unknown mark 'not_a_mark'"):
        ir.Trace("not_a_mark", y=[1.0])


def test_catalog_registration_and_aliases() -> None:
    catalog = {kind.name: kind for kind in viz_spec.kind_catalog()}
    assert "violin" in catalog
    assert "boxplot" in catalog
    assert "line_plot" in catalog

    violin_kind = viz_spec.get_kind("violin")
    assert violin_kind.name == "violin"
    assert violin_kind.group == "overview"
    assert set(violin_kind.aliases) == {"violinplot", "violin_plot"}
    assert viz_spec.get_kind("violinplot").name == "violin"
    assert viz_spec.get_kind("violin_plot").name == "violin"

    boxplot_kind = viz_spec.get_kind("boxplot")
    assert boxplot_kind.name == "boxplot"
    assert boxplot_kind.group == "overview"
    assert set(boxplot_kind.aliases) == {"box", "box_plot"}
    assert viz_spec.get_kind("box").name == "boxplot"
    assert viz_spec.get_kind("box_plot").name == "boxplot"

    line_kind = viz_spec.get_kind("line_plot")
    assert line_kind.name == "line_plot"
    assert line_kind.group == "timeline"
    assert set(line_kind.aliases) == {"line", "lineplot"}
    assert viz_spec.get_kind("line").name == "line_plot"
    assert viz_spec.get_kind("lineplot").name == "line_plot"


def test_matplotlib_violin_rendering_formats(tmp_path: pathlib.Path) -> None:
    panel = ir.Panel(
        traces=[
            ir.Trace("violin", y=[1.0, 2.0, 2.5, 3.0, 3.5, 4.0], name="A"),
            ir.Trace("violin", y=[2.0, 3.0, 3.5, 4.0, 4.5, 6.0], name="B"),
        ]
    )
    chart = ir.Chart(title="Violin Test", panels=[panel])
    fig = mpl_backend.render(chart)

    # Test PDF
    pdf_path = tmp_path / "violin.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    assert pdf_path.stat().st_size > 1000

    # Test PNG
    png_path = tmp_path / "violin.png"
    fig.savefig(png_path, bbox_inches="tight")
    assert png_path.stat().st_size > 1000

    # Test SVG
    svg_path = tmp_path / "violin.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    assert svg_path.stat().st_size > 500
    assert "<svg" in svg_path.read_text()


def test_plotly_violin_rendering(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("plotly")
    panel = ir.Panel(
        traces=[
            ir.Trace("violin", y=[1.0, 2.0, 2.5, 3.0, 3.5, 4.0], name="A"),
            ir.Trace("violin", y=[2.0, 3.0, 3.5, 4.0, 4.5, 6.0], name="B"),
        ]
    )
    chart = ir.Chart(title="Violin Test", panels=[panel])
    plotly_fig = plotly_backend.render(chart)
    html_path = tmp_path / "violin.html"
    plots.write_html(plotly_fig, html_path)
    assert html_path.stat().st_size > 1000
    content = html_path.read_text().lower()
    assert "plotly" in content


def test_custom_mark_violin(tmp_path: pathlib.Path) -> None:
    report = _report_fixture(tmp_path)
    ctx = context.DataContext(report)
    chart, _ = viz.build_chart(
        ctx,
        "custom",
        {"source": "kernels", "mark": "violin", "y": "duration_ms"},
    )
    assert any(t.mark == "violin" for p in chart.panels for t in p.traces)


def test_report_plot_api(tmp_path: pathlib.Path) -> None:
    report = _report_fixture(tmp_path)

    # Violin plot static
    v_png = tmp_path / "v.png"
    report.plot(kind="violin", path=v_png)
    assert v_png.stat().st_size > 1000

    # Violin plot interactive HTML
    pytest.importorskip("plotly")
    v_html = tmp_path / "v.html"
    report.plot(kind="violin", path=v_html)
    assert v_html.stat().st_size > 1000

    # Boxplot static
    b_png = tmp_path / "b.png"
    report.plot(kind="boxplot", path=b_png)
    assert b_png.stat().st_size > 1000

    # Boxplot alias "box"
    b_alias_png = tmp_path / "box_alias.png"
    report.plot(kind="box", path=b_alias_png)
    assert b_alias_png.stat().st_size > 1000

    # Line plot static
    l_png = tmp_path / "l.png"
    report.plot(kind="line_plot", path=l_png)
    assert l_png.stat().st_size > 1000

    # Line plot alias "line"
    l_alias_png = tmp_path / "line_alias.png"
    report.plot(kind="line", path=l_alias_png)
    assert l_alias_png.stat().st_size > 1000


def test_violin_with_numpy_array(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("plotly")
    y_arr = np.array([1.0, 2.0, 3.0])
    trace = _trace("violin", y=y_arr, name="np_trace")
    panel = ir.Panel(traces=[trace])
    chart = ir.Chart(title="Violin NumPy", panels=[panel])

    # Check suitability without raising ValueError
    report = _report_fixture(tmp_path)
    ctx = context.DataContext(report)
    msgs = suitability.check_suitability(ctx, "violin", {}, chart=chart)
    assert any("Sample size (N=3) is too small" in m for m in msgs)

    # Render in Matplotlib (.pdf, .png, .svg)
    fig = mpl_backend.render(chart)
    pdf_path = tmp_path / "np_violin.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    assert pdf_path.stat().st_size > 1000

    png_path = tmp_path / "np_violin.png"
    fig.savefig(png_path, bbox_inches="tight")
    assert png_path.stat().st_size > 1000

    svg_path = tmp_path / "np_violin.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    assert svg_path.stat().st_size > 500

    # Render in Plotly (.html)
    plotly_fig = plotly_backend.render(chart)
    html_path = tmp_path / "np_violin.html"
    plots.write_html(plotly_fig, html_path)
    assert html_path.stat().st_size > 1000


def test_violin_with_polars_series(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("plotly")
    y_series = pl.Series("y", [1.0, 2.0, 3.0])
    trace = _trace("violin", y=y_series, name="pl_trace")
    panel = ir.Panel(traces=[trace])
    chart = ir.Chart(title="Violin Polars", panels=[panel])

    # Check suitability without raising TypeError
    report = _report_fixture(tmp_path)
    ctx = context.DataContext(report)
    msgs = suitability.check_suitability(ctx, "violin", {}, chart=chart)
    assert any("Sample size (N=3) is too small" in m for m in msgs)

    # Render in Matplotlib (.pdf, .png, .svg)
    fig = mpl_backend.render(chart)
    png_path = tmp_path / "pl_violin.png"
    fig.savefig(png_path, bbox_inches="tight")
    assert png_path.stat().st_size > 1000

    # Render in Plotly (.html)
    plotly_fig = plotly_backend.render(chart)
    html_path = tmp_path / "pl_violin.html"
    plots.write_html(plotly_fig, html_path)
    assert html_path.stat().st_size > 1000


def test_violin_zero_array_no_silent_data_loss(tmp_path: pathlib.Path) -> None:
    trace = _trace("violin", y=np.array([0.0]), name="zero_arr")
    panel = ir.Panel(traces=[trace])
    chart = ir.Chart(title="Violin Zero Array", panels=[panel])
    report = _report_fixture(tmp_path)
    ctx = context.DataContext(report)
    msgs = suitability.check_suitability(ctx, "violin", {}, chart=chart)
    # Correctly evaluated as sample size 1 (not 0)
    assert any("Sample size (N=1) is too small" in m for m in msgs)


def test_violin_none_filtering_render_mpl(tmp_path: pathlib.Path) -> None:
    trace = ir.Trace("violin", y=[1.0, None, 2.0], name="none_val")
    panel = ir.Panel(traces=[trace])
    chart = ir.Chart(title="Violin None Filtering", panels=[panel])
    # Matplotlib rendering should filter cleanly without TypeError
    fig = mpl_backend.render(chart)
    png_path = tmp_path / "none_filter.png"
    fig.savefig(png_path, bbox_inches="tight")
    assert png_path.stat().st_size > 1000
