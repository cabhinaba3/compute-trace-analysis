"""Tests for the plot suitability checks."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import polars as pl
import pytest

from cta import models
from cta.viz import context
from cta.viz import ir
from cta.viz import suitability


def _trace(mark: str, **fields: Any) -> ir.Trace:
    """Builds a trace from fields of any type, such as arrays and Series."""
    return ir.Trace(mark, **fields)


def _empty_report(world_size: int = 1) -> models.ComparisonReport:
    """Creates a minimal synthetic ComparisonReport."""
    art = models.TraceArtifact(
        format="kineto",
        paths=(pathlib.Path("/fake/trace.json"),),
        world_size=world_size,
    )
    bundle = models.RunBundle(
        run_id="run0",
        label="base",
        artifacts={"trace": art},
    )
    group = models.TrialGroup(
        group_id="group0",
        label="base",
        trials=(bundle,),
    )
    return models.ComparisonReport(
        mode="single",
        groups=(group,),
        aspects={},
        sources={},
    )


def test_suitability_warning_class() -> None:
    assert issubclass(suitability.SuitabilityWarning, UserWarning)


def test_violin_sample_size_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # Trace with N = 4 (< 10)
    trace = ir.Trace("violin", y=[1.0, 2.0, 3.0, 4.0], name="test")
    chart = ir.Chart(title="Violin", panels=[ir.Panel(traces=[trace])])
    msgs = suitability.check_suitability(ctx, "violin", {}, chart=chart)
    assert any("Sample size (N=4) is too small" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'boxplot', 'kernel_scatter', "
        "'category_compare'" in m
        for m in msgs
    )


def test_boxplot_sample_size_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # Trace with N = 3 (< 5)
    trace = ir.Trace("box", y=[1.0, 2.0, 3.0], name="test")
    chart = ir.Chart(title="Box", panels=[ir.Panel(traces=[trace])])
    msgs = suitability.check_suitability(ctx, "boxplot", {}, chart=chart)
    assert any(
        "Sample size (N=3) is insufficient to calculate" in m for m in msgs
    )
    assert any(
        "Suggested alternatives: 'kernel_scatter', 'category_compare'" in m
        for m in msgs
    )


def test_line_plot_sample_size_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # Trace with N = 1 (< 2)
    trace = ir.Trace("line", x=[1.0], y=[2.0], name="test")
    chart = ir.Chart(title="Line", panels=[ir.Panel(traces=[trace])])
    msgs = suitability.check_suitability(ctx, "line_plot", {}, chart=chart)
    assert any(
        "Sample size (N=1) is insufficient for line plot" in m for m in msgs
    )
    assert any(
        "Suggested alternatives: 'kernel_scatter', 'category_compare'" in m
        for m in msgs
    )


def test_violin_distribution_continuity_and_zero_variance() -> None:
    ctx = context.DataContext(_empty_report())
    # N = 12, but only 2 distinct values (3 or fewer is too few)
    trace_discrete = ir.Trace("violin", y=[1.0, 2.0] * 6, name="discrete")
    chart_discrete = ir.Chart(
        title="Violin", panels=[ir.Panel(traces=[trace_discrete])]
    )
    msgs = suitability.check_suitability(
        ctx, "violin", {}, chart=chart_discrete
    )
    assert any("only 2 distinct values" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'boxplot', 'metric_dist', 'category_compare'"
        in m
        for m in msgs
    )

    # N = 12, zero variance
    trace_zero_var = ir.Trace("violin", y=[5.0] * 12, name="zero_var")
    chart_zero_var = ir.Chart(
        title="Violin", panels=[ir.Panel(traces=[trace_zero_var])]
    )
    msgs_var = suitability.check_suitability(
        ctx, "violin", {}, chart=chart_zero_var
    )
    assert any("no variance" in m for m in msgs_var)


def test_boxplot_multimodality_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # Bimodal distribution with two separate peaks
    peak1 = np.random.normal(loc=10.0, scale=0.5, size=25)
    peak2 = np.random.normal(loc=100.0, scale=0.5, size=25)
    bimodal = np.concatenate([peak1, peak2]).tolist()
    trace = ir.Trace("box", y=bimodal, name="bimodal")
    chart = ir.Chart(title="Box", panels=[ir.Panel(traces=[trace])])
    msgs = suitability.check_suitability(ctx, "boxplot", {}, chart=chart)
    assert any("several peaks" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'violin', 'metric_dist'" in m for m in msgs
    )


def test_line_plot_unordered_categorical_x_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # X has non-numeric category strings
    trace = ir.Trace(
        "line",
        x=["gemm", "conv", "bn", "relu"],
        y=[1.0, 2.0, 3.0, 4.0],
        name="kernels",
    )
    chart = ir.Chart(title="Line", panels=[ir.Panel(traces=[trace])])
    msgs = suitability.check_suitability(ctx, "line_plot", {}, chart=chart)
    assert any("holds categories" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'category_compare', 'kernel_scatter'" in m
        for m in msgs
    )


def test_line_plot_duplicate_x_with_varying_y_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # Duplicate X values with differing Y values
    trace = ir.Trace(
        "line",
        x=[1.0, 2.0, 2.0, 3.0],
        y=[10.0, 20.0, 30.0, 40.0],
        name="zigzag",
    )
    chart = ir.Chart(title="Line", panels=[ir.Panel(traces=[trace])])
    msgs = suitability.check_suitability(ctx, "line_plot", {}, chart=chart)
    assert any("duplicate X values with varying Y values" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'kernel_scatter', 'boxplot', 'violin'" in m
        for m in msgs
    )


def test_multi_rank_on_single_rank_trace_heuristic() -> None:
    # world_size = 1
    ctx = context.DataContext(_empty_report(world_size=1))
    msgs = suitability.check_suitability(ctx, "skew", {})
    assert any(
        "Multi-rank distribution plot 'skew' requested" in m for m in msgs
    )
    assert any("trace has only a single rank (world_size=1)" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'breakdown', 'timeline', 'step_dist'" in m
        for m in msgs
    )

    # By rank on single rank
    msgs_by = suitability.check_suitability(ctx, "metric_dist", {"by": "rank"})
    assert any(
        "Multi-rank distribution plot 'metric_dist' requested" in m
        for m in msgs_by
    )


def test_violin_on_two_ranks_heuristic() -> None:
    # world_size = 2
    ctx = context.DataContext(_empty_report(world_size=2))
    msgs = suitability.check_suitability(ctx, "violin", {"by": "rank"})
    assert any("contains only 2 ranks" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'category_compare', 'boxplot', 'step_compare'"
        in m
        for m in msgs
    )


def test_kernel_duration_dynamic_range_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # Kernel durations spanning ratio > 1000 without log scale
    trace = ir.Trace(
        "box",
        y=[0.01, 0.05, 0.1, 1.0, 10.0, 100.0, 500.0],
        name="durations",
    )
    chart = ir.Chart(title="Box", panels=[ir.Panel(traces=[trace])])
    # max / min = 500.0 / 0.01 = 50,000 > 1000
    msgs = suitability.check_suitability(
        ctx, "boxplot", {"metric": "duration_ms", "logy": False}, chart=chart
    )
    assert any(
        "Kernel durations span multiple orders of magnitude" in m for m in msgs
    )
    assert any("enable log scale (logy=True)" in m for m in msgs)

    # When logy is True, no warning
    msgs_log = suitability.check_suitability(
        ctx, "boxplot", {"metric": "duration_ms", "logy": True}, chart=chart
    )
    assert not any("span multiple orders of magnitude" in m for m in msgs_log)


def test_valid_data_emits_no_warnings() -> None:
    ctx = context.DataContext(_empty_report(world_size=4))
    # N = 20, reasonable variance, continuous
    trace = ir.Trace(
        "violin",
        y=np.linspace(10.0, 20.0, 20).tolist(),
        name="valid",
    )
    chart = ir.Chart(title="Violin", panels=[ir.Panel(traces=[trace])])
    msgs = suitability.check_suitability(
        ctx, "violin", {"metric": "duration_ms", "logy": False}, chart=chart
    )
    assert msgs == []


def test_validate_and_warn_emits_suitability_warning() -> None:
    ctx = context.DataContext(_empty_report())
    trace = ir.Trace("violin", y=[1.0, 2.0, 3.0], name="test")
    chart = ir.Chart(title="Violin", panels=[ir.Panel(traces=[trace])])
    with pytest.warns(suitability.SuitabilityWarning, match="Sample size"):
        suitability.validate_and_warn(ctx, "violin", {}, chart=chart)


def test_suitability_with_numpy_and_polars_sequences() -> None:
    ctx = context.DataContext(_empty_report())

    # Violin with ndarray (N=3)
    v_trace_np = _trace("violin", y=np.array([1.0, 2.0, 3.0]))
    v_chart_np = ir.Chart(title="V NP", panels=[ir.Panel(traces=[v_trace_np])])
    msgs_v_np = suitability.check_suitability(
        ctx, "violin", {}, chart=v_chart_np
    )
    assert any("Sample size (N=3) is too small" in m for m in msgs_v_np)

    # Violin with polars Series (N=3)
    v_trace_pl = _trace("violin", y=pl.Series("y", [1.0, 2.0, 3.0]))
    v_chart_pl = ir.Chart(title="V PL", panels=[ir.Panel(traces=[v_trace_pl])])
    msgs_v_pl = suitability.check_suitability(
        ctx, "violin", {}, chart=v_chart_pl
    )
    assert any("Sample size (N=3) is too small" in m for m in msgs_v_pl)

    # Boxplot with ndarray (N=3)
    b_trace_np = _trace("box", y=np.array([1.0, 2.0, 3.0]))
    b_chart_np = ir.Chart(title="B NP", panels=[ir.Panel(traces=[b_trace_np])])
    msgs_b_np = suitability.check_suitability(
        ctx, "boxplot", {}, chart=b_chart_np
    )
    assert any(
        "Sample size (N=3) is insufficient to calculate" in m for m in msgs_b_np
    )

    # Boxplot with polars Series (N=3)
    b_trace_pl = _trace("box", y=pl.Series("y", [1.0, 2.0, 3.0]))
    b_chart_pl = ir.Chart(title="B PL", panels=[ir.Panel(traces=[b_trace_pl])])
    msgs_b_pl = suitability.check_suitability(
        ctx, "boxplot", {}, chart=b_chart_pl
    )
    assert any(
        "Sample size (N=3) is insufficient to calculate" in m for m in msgs_b_pl
    )

    # Line plot with ndarray (N=1)
    l_trace_np = _trace(
        "line",
        x=np.array([1.0]),
        y=np.array([2.0]),
    )
    l_chart_np = ir.Chart(title="L NP", panels=[ir.Panel(traces=[l_trace_np])])
    msgs_l_np = suitability.check_suitability(
        ctx, "line_plot", {}, chart=l_chart_np
    )
    assert any(
        "Sample size (N=1) is insufficient for line plot" in m
        for m in msgs_l_np
    )

    # Line plot with polars Series (N=1)
    l_trace_pl = _trace(
        "line",
        x=pl.Series("x", [1.0]),
        y=pl.Series("y", [2.0]),
    )
    l_chart_pl = ir.Chart(title="L PL", panels=[ir.Panel(traces=[l_trace_pl])])
    msgs_l_pl = suitability.check_suitability(
        ctx, "line_plot", {}, chart=l_chart_pl
    )
    assert any(
        "Sample size (N=1) is insufficient for line plot" in m
        for m in msgs_l_pl
    )


def test_suitability_zero_array_no_silent_loss() -> None:
    ctx = context.DataContext(_empty_report())
    trace = _trace("violin", y=np.array([0.0]))
    chart = ir.Chart(title="Zero", panels=[ir.Panel(traces=[trace])])
    msgs = suitability.check_suitability(ctx, "violin", {}, chart=chart)
    assert any("Sample size (N=1) is too small" in m for m in msgs)


def test_suitability_none_in_sequence_filtered() -> None:
    ctx = context.DataContext(_empty_report())
    trace = ir.Trace("violin", y=[1.0, None, 2.0])
    chart = ir.Chart(title="None", panels=[ir.Panel(traces=[trace])])
    msgs = suitability.check_suitability(ctx, "violin", {}, chart=chart)
    # Only 2 finite elements
    assert any("Sample size (N=2) is too small" in m for m in msgs)


def test_extract_suggested_alternatives() -> None:
    msg1 = (
        "Sample size (N=4) is too small. "
        "Suggested alternatives: 'boxplot', 'kernel_scatter', "
        "'category_compare'."
    )
    assert suitability.extract_suggested_alternatives(msg1) == [
        "boxplot",
        "kernel_scatter",
        "category_compare",
    ]
    msg2 = "No alternatives here."
    assert suitability.extract_suggested_alternatives(msg2) == []


def test_queue_depth_sample_size_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # Chart with meta indicating N=3 (< 5)
    trace = ir.Trace("step", x=[0.0, 1.0], y=[1.0, 1.0], name="test")
    chart = ir.Chart(
        title="Queue Depth",
        panels=[ir.Panel(traces=[trace])],
        meta={"kernel_count": 3},
    )
    msgs = suitability.check_suitability(ctx, "queue_depth", {}, chart=chart)
    assert any("Sample size (N=3) is insufficient" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'timeline', 'metric_trace'" in m for m in msgs
    )


def test_queue_depth_serialized_execution_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # Chart with N=10 kernels, but max concurrency is 1 (never overlaps)
    trace = ir.Trace(
        "step",
        x=[0.0, 1.0, 2.0, 3.0],
        y=[1.0, 1.0, 0.0, 1.0],
        name="test",
    )
    chart = ir.Chart(
        title="Queue Depth",
        panels=[ir.Panel(traces=[trace])],
        meta={"kernel_count": 10},
    )
    msgs = suitability.check_suitability(ctx, "concurrency", {}, chart=chart)
    assert any("never exceeds 1" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'timeline', 'stream_util', 'breakdown'" in m
        for m in msgs
    )


def test_queue_depth_idle_starvation_heuristic() -> None:
    ctx = context.DataContext(_empty_report())
    # 9 out of 10 points are 0 concurrency (90% idle > 80%)
    trace = ir.Trace(
        "step",
        x=list(range(10)),
        y=[2.0] + [0.0] * 9,
        name="test",
    )
    chart = ir.Chart(
        title="Queue Depth",
        panels=[ir.Panel(traces=[trace])],
        meta={"kernel_count": 10},
    )
    msgs = suitability.check_suitability(ctx, "active_kernels", {}, chart=chart)
    assert any("over 80% of the window" in m for m in msgs)
    assert any(
        "Suggested alternatives: 'overlap_timeline', 'breakdown'" in m
        for m in msgs
    )


def test_queue_depth_suitable_data() -> None:
    ctx = context.DataContext(_empty_report())
    # N=10, max concurrency = 3, idle is only 1 out of 10 points (10% idle)
    trace = ir.Trace(
        "step",
        x=list(range(10)),
        y=[1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 0.0],
        name="test",
    )
    chart = ir.Chart(
        title="Queue Depth",
        panels=[ir.Panel(traces=[trace])],
        meta={"kernel_count": 10},
    )
    msgs = suitability.check_suitability(ctx, "queue_depth", {}, chart=chart)
    assert msgs == []
