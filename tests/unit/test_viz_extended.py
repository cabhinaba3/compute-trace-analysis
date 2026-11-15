"""Tests of the statistical, memory, hierarchy, runtime and general plot kinds.

The raw tables are built in memory, so the data of every case is known and the
numbers drawn can be checked against it.
"""

from __future__ import annotations

import pathlib
from typing import Any
import warnings

import numpy as np
import polars as pl
import pytest

from cta import viz
from cta.core import models
from cta.viz import data
from cta.viz import spec as viz_spec
from cta.viz import suitability
from cta.viz.kinds import chart_helpers
from cta.viz.kinds import memory
from cta.viz.kinds import runtime_api

LABELS = ("base", "cand")


def _report(
    aspects: dict[str, models.AspectResult] | None = None,
    labels: tuple[str, ...] = LABELS,
) -> models.ComparisonReport:
    """Returns a report of groups with these labels and these aspects."""
    groups = tuple(
        models.TrialGroup(
            group_id=f"g{i}",
            label=label,
            trials=(
                models.RunBundle(
                    run_id=f"r{i}",
                    label=label,
                    artifacts={
                        "trace": models.TraceArtifact(
                            format="kineto",
                            paths=(pathlib.Path("/none"),),
                            world_size=2,
                        )
                    },
                ),
            ),
        )
        for i, label in enumerate(labels)
    )
    return models.ComparisonReport(
        mode="pair" if len(labels) == 2 else "multi",
        groups=groups,
        aspects=aspects or {},
        sources={},
    )


def _context(
    raw: dict[str, list[dict[str, Any]]] | None = None,
    aspects: dict[str, models.AspectResult] | None = None,
    labels: tuple[str, ...] = LABELS,
) -> data.DataContext:
    """Returns a context over in-memory raw tables."""
    ctx = data.DataContext(_report(aspects, labels))
    ctx._raw_tables = {
        name: pl.DataFrame(rows) for name, rows in (raw or {}).items()
    }
    ctx._is_built = True
    return ctx


def _kernel(
    label: str,
    start_ms: float,
    duration_ms: float,
    *,
    rank: int = 0,
    stream: int = 7,
    category: str = "compute",
    name: str = "gemm",
    size: int = 0,
    scope_path: str = "",
) -> dict[str, Any]:
    start_ns = int(start_ms * 1e6)
    end_ns = int((start_ms + duration_ms) * 1e6)
    return {
        "label": label,
        "trial": 0,
        "rank": rank,
        "device": rank,
        "stream": stream,
        "step": 0,
        "idx": 0,
        "category": category,
        "name": name,
        "op_family": "matmul" if category == "compute" else "copy",
        "scope": scope_path.split(" > ")[-1],
        "scope_path": scope_path,
        "bytes": size,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        "t_ms": start_ms,
        "end_ms": start_ms + duration_ms,
        "duration_ms": duration_ms,
    }


def _call(
    label: str,
    start_ms: float,
    duration_ms: float,
    name: str,
    *,
    thread: int = 1,
    rank: int = 0,
) -> dict[str, Any]:
    start_ns = int(start_ms * 1e6)
    end_ns = int((start_ms + duration_ms) * 1e6)
    return {
        "label": label,
        "trial": 0,
        "rank": rank,
        "step": 0,
        "idx": 0,
        "name": name,
        "thread": thread,
        "correlation": -1,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        "t_ms": start_ms,
        "end_ms": start_ms + duration_ms,
        "duration_ms": duration_ms,
    }


def _summary(rows: list[dict[str, Any]]) -> models.AspectResult:
    return models.AspectResult(
        name="demo", title="demo", summary=pl.DataFrame(rows)
    )


def _pair_row(
    metric: str,
    baseline: float,
    candidate: float,
    *,
    ci_base: float = 0.0,
    ci_cand: float = 0.0,
    p: float | None = None,
    d: float = 0.0,
    significant: bool = False,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "unit": "ms",
        "baseline": baseline,
        "candidate": candidate,
        "delta": candidate - baseline,
        "delta_pct": (candidate - baseline) / baseline * 100.0,
        "ci95_baseline": ci_base,
        "ci95_candidate": ci_cand,
        "p_welch": p,
        "p_mannwhitney": p,
        "cohens_d": d,
        "significant": significant,
    }


def _raw() -> dict[str, list[dict[str, Any]]]:
    """Two groups on two ranks: scopes, copies, sync calls, API totals."""
    kernels, calls, runtime, osrt = [], [], [], []
    for scale, label in enumerate(LABELS, start=1):
        for rank in (0, 1):
            for step in range(6):
                base = step * 10.0
                kernels += [
                    _kernel(
                        label,
                        base,
                        2.0 * scale,
                        rank=rank,
                        scope_path="train > forward > attn",
                        name="gemm_a",
                    ),
                    _kernel(
                        label,
                        base + 3.0,
                        1.0 * scale,
                        rank=rank,
                        scope_path="train > forward > mlp",
                        name="gemm_b",
                    ),
                    _kernel(
                        label,
                        base + 5.0,
                        0.5,
                        rank=rank,
                        scope_path="train > backward",
                        name="gemm_c",
                    ),
                    _kernel(
                        label,
                        base + 1.0,
                        1.0,
                        rank=rank,
                        stream=9,
                        category="memcpy",
                        name="Memcpy_H2D",
                        size=4_000_000 * (step + 1),
                    ),
                    _kernel(
                        label,
                        base + 8.0,
                        0.5,
                        rank=rank,
                        stream=9,
                        category="memcpy",
                        name="Memcpy HtoD (Pinned -> Device)"
                        if step % 2
                        else "Memcpy_D2H",
                        size=1_000_000,
                    ),
                ]
                calls += [
                    _call(label, base, 0.02, "cudaLaunchKernel", rank=rank),
                    _call(
                        label,
                        base + 2.0,
                        1.5 * scale,
                        "cudaStreamSynchronize",
                        rank=rank,
                    ),
                    _call(label, base + 4.0, 0.3, "cudaMalloc", rank=rank),
                ]
        runtime += [
            {
                "label": label,
                "trial": 0,
                "name": name,
                "duration_ns": ns,
                "calls": n,
            }
            for name, ns, n in (
                ("cudaLaunchKernel", 4_000_000 * scale, 4000),
                ("cudaStreamSynchronize", 9_000_000 * scale, 60),
                ("cudaMalloc", 1_000_000, 12),
            )
        ]
        osrt += [
            {
                "label": label,
                "trial": 0,
                "name": "pthread_cond_wait",
                "duration_ns": 5_000_000 * scale,
                "calls": 100,
            },
            {
                "label": label,
                "trial": 0,
                "name": "read",
                "duration_ns": 1_000_000,
                "calls": 30,
            },
        ]
    return {
        "kernels": kernels,
        "runtime_calls": calls,
        "runtime": runtime,
        "osrt": osrt,
    }


def _aspects() -> dict[str, models.AspectResult]:
    return {
        "demo": _summary(
            [
                _pair_row(
                    "step_ms",
                    10.0,
                    14.0,
                    ci_base=0.5,
                    ci_cand=1.5,
                    p=0.001,
                    d=2.1,
                    significant=True,
                ),
                _pair_row(
                    "comm_ms", 4.0, 3.9, ci_base=0.4, ci_cand=0.4, p=0.6, d=-0.1
                ),
                _pair_row("idle_ms", 2.0, 2.6, p=0.04, d=0.8, significant=True),
                _pair_row("kernel_ms", 7.0, 9.0, p=0.2, d=0.5),
            ]
        )
    }


# Every kind builds, draws on both backends, and raises no suitability warning.

_RAW_KINDS: list[tuple[str, dict[str, Any]]] = [
    ("memcpy_bandwidth", {}),
    ("memcpy_sizes", {}),
    ("memcpy_overlap", {}),
    ("scope_icicle", {}),
    ("idle_gaps", {}),
    ("runtime_top", {}),
    ("runtime_top", {"source": "osrt", "metric": "calls"}),
    ("runtime_timeline", {"window_ms": 30}),
    ("sync_waits", {}),
    ("runtime_step_mix", {}),
    ("ridgeline", {}),
    ("ridgeline", {"by": "rank", "logx": True}),
    ("facet", {"kind": "metric_dist", "by": "rank"}),
    ("facet", {"kind": "memcpy_bandwidth", "by": "label"}),
]
_REPORT_KINDS: list[tuple[str, dict[str, Any]]] = [
    ("forest", {}),
    ("volcano", {}),
    ("metric_radar", {}),
    ("parallel_coords", {}),
]


@pytest.mark.parametrize(("kind", "options"), _RAW_KINDS + _REPORT_KINDS)
def test_kinds_build_and_draw_on_both_backends(
    kind: str, options: dict[str, Any], tmp_path: pathlib.Path
) -> None:
    ctx = _context(_raw(), _aspects())
    with warnings.catch_warnings():
        warnings.simplefilter("error", suitability.SuitabilityWarning)
        chart, resolved = viz.build_chart(ctx, kind, options)
    assert chart.panels
    assert all(panel.traces for panel in chart.panels)
    viz.render_chart(chart, resolved, tmp_path / "a.png")
    assert (tmp_path / "a.png").stat().st_size > 1000
    pytest.importorskip("plotly")
    viz.render_chart(chart, resolved, tmp_path / "a.html")
    assert "plotly" in (tmp_path / "a.html").read_text()[:200_000].lower()


def test_new_kinds_are_registered_with_options() -> None:
    viz_spec.load_kinds()
    names = {kind.name for kind in viz_spec.kind_catalog()}
    expected = {
        "forest",
        "volcano",
        "memcpy_bandwidth",
        "memcpy_sizes",
        "memcpy_overlap",
        "scope_icicle",
        "idle_gaps",
        "critical_path",
        "runtime_top",
        "runtime_timeline",
        "sync_waits",
        "runtime_step_mix",
        "metric_radar",
        "ridgeline",
        "parallel_coords",
        "facet",
    }
    assert expected <= names
    assert "runtime_calls" in data.SOURCES
    for name in sorted(expected):
        kind = viz_spec.get_kind(name)
        assert kind.group in viz_spec.GROUPS
        assert kind.title


# Interval arithmetic.


def test_merge_intervals_joins_overlaps_and_keeps_touching_apart() -> None:
    starts, ends = chart_helpers.merge_intervals(
        np.array([5.0, 0.0, 2.0, 10.0, 8.0]),
        np.array([6.0, 3.0, 4.0, 11.0, 10.0]),
    )
    # [8, 10) and [10, 11) only touch, so they stay two ranges.
    assert starts.tolist() == [0.0, 5.0, 8.0, 10.0]
    assert ends.tolist() == [4.0, 6.0, 10.0, 11.0]
    starts, ends = chart_helpers.merge_intervals(
        np.array([0.0, 1.0]), np.array([1.0, 2.0])
    )
    assert starts.tolist() == [0.0, 1.0]


def test_covered_length_counts_only_what_the_ranges_cover() -> None:
    merged = (np.array([0.0, 10.0]), np.array([4.0, 14.0]))
    covered = chart_helpers.covered_length(
        *merged,
        np.array([1.0, 3.0, 5.0, 12.0, -5.0]),
        np.array([2.0, 11.0, 9.0, 20.0, 20.0]),
    )
    assert covered.tolist() == [1.0, 2.0, 0.0, 2.0, 8.0]


# Memory.


@pytest.mark.parametrize(
    ("name", "direction"),
    [
        ("Memcpy_H2D", "H2D"),
        ("Memcpy HtoD (Pinned -> Device)", "H2D"),
        ("Memcpy DtoH (Device -> Pageable)", "D2H"),
        ("Memcpy_D2D", "D2D"),
        ("Memcpy PtoP (Device -> Device)", "P2P"),
        ("Memcpy_H2H", "H2H"),
        ("Memset (Device)", "memset"),
        ("Memcpy_Other", "other"),
    ],
)
def test_copy_direction_from_the_name(name: str, direction: str) -> None:
    assert memory.direction_of(name) == direction


def test_memcpy_bandwidth_is_bytes_over_time() -> None:
    rows = [
        _kernel(
            "base", 0, 2.0, category="memcpy", name="Memcpy_H2D", size=8_000_000
        ),
        _kernel(
            "base", 5, 2.0, category="memcpy", name="Memcpy_H2D", size=4_000_000
        ),
        _kernel("base", 9, 1.0, category="memcpy", name="Memcpy_D2H", size=0),
    ]
    chart, _ = viz.build_chart(
        _context({"kernels": rows}, labels=("base",)), "memcpy_bandwidth"
    )
    trace = chart.panels[0].traces[0]
    # 12 MB in 4 ms is 3 GB/s; the copy of unknown size is left out.
    assert dict(zip(trace.x or [], trace.y or [], strict=True)) == {"H2D": 3.0}


def test_memcpy_overlap_measures_the_share_under_compute() -> None:
    rows = [
        _kernel("base", 0, 4.0),
        _kernel("base", 10, 2.0),
        _kernel(
            "base", 3, 2.0, category="memcpy", name="Memcpy_H2D", size=1000
        ),
        _kernel(
            "base", 11, 4.0, category="memcpy", name="Memcpy_H2D", size=1000
        ),
        # Other device: its compute does not hide this copy.
        _kernel(
            "base",
            20,
            1.0,
            rank=1,
            category="memcpy",
            name="Memcpy_H2D",
            size=1000,
        ),
    ]
    chart, _ = viz.build_chart(
        _context({"kernels": rows}, labels=("base",)), "memcpy_overlap"
    )
    share, exposed = chart.panels
    # 1 of 2 ms, 1 of 4 ms and 0 of 1 ms hide behind compute: 2 of 7 ms.
    assert share.traces[0].y == [pytest.approx(100 * 2 / 7)]
    assert exposed.traces[0].y == [pytest.approx(5.0)]


def test_memcpy_kinds_say_when_there_are_no_copies() -> None:
    ctx = _context({"kernels": [_kernel("base", 0, 1.0)]}, labels=("base",))
    for kind in ("memcpy_bandwidth", "memcpy_sizes", "memcpy_overlap"):
        with pytest.raises(
            viz_spec.NotApplicableError, match="no memory copies"
        ):
            viz.build_chart(ctx, kind)


def test_memcpy_sizes_bins_calls_on_a_log_axis() -> None:
    sizes = [1_000, 1_000, 1_000_000, 100_000_000]
    rows = [
        _kernel("base", i, 1.0, category="memcpy", name="Memcpy_H2D", size=size)
        for i, size in enumerate(sizes)
    ]
    chart, _ = viz.build_chart(
        _context({"kernels": rows}, labels=("base",)),
        "memcpy_sizes",
        {"bins": 8},
    )
    trace = chart.panels[0].traces[0]
    assert chart.panels[0].xscale == "log"
    assert sum(trace.y or []) == 4
    assert (trace.y or [])[0] == 2


# Hierarchy.


def test_icicle_widths_add_up_and_nest() -> None:
    rows = [
        _kernel("base", 0, 3.0, scope_path="a > b"),
        _kernel("base", 4, 1.0, scope_path="a > c"),
        _kernel("base", 6, 2.0, scope_path="d"),
    ]
    chart, _ = viz.build_chart(
        _context({"kernels": rows}, labels=("base",)),
        "scope_icicle",
        {"min_share": 0},
    )
    segments = [t for t in chart.panels[0].traces if t.mark == "segment"]
    cells = {
        text.split(":")[0]: (x, x1, y)
        for trace in segments
        for text, x, x1, y in zip(
            trace.text or [],
            trace.x or [],
            trace.x1 or [],
            trace.y or [],
            strict=True,
        )
    }
    assert cells["a"] == (0.0, 4.0, "level 0")
    assert cells["d"] == (4.0, 6.0, "level 0")
    # Children sit inside their parent, largest first.
    assert cells["b"] == (0.0, 3.0, "level 1")
    assert cells["c"] == (3.0, 4.0, "level 1")


def test_icicle_falls_back_to_category_and_name_without_scopes() -> None:
    rows = [_kernel("base", i, 1.0, scope_path="") for i in range(4)]
    chart, _ = viz.build_chart(
        _context({"kernels": rows}, labels=("base",)), "scope_icicle"
    )
    assert "no scopes" in chart.notes


def test_idle_gaps_come_from_the_union_of_overlapping_kernels() -> None:
    rows = [
        _kernel("base", 0, 4.0, stream=1),
        _kernel("base", 1, 1.0, stream=2),
        _kernel("base", 5, 1.0, stream=1),
        _kernel("base", 6.5, 0.5, stream=2),
        # Another device has its own gaps; they are not mixed in.
        _kernel("base", 0, 1.0, rank=1),
        _kernel("base", 9, 1.0, rank=1),
    ]
    chart, _ = viz.build_chart(
        _context({"kernels": rows}, labels=("base",)),
        "idle_gaps",
        {"bins": 4},
    )
    counts, shares = chart.panels
    # Gaps: 1 ms and 0.5 ms on device 0, 8 ms on device 1. A 1 ms gap is
    # in the "1-10 ms" class, so that class holds 9 of the 9.5 idle ms.
    assert sum(counts.traces[0].y or []) == 3
    assert shares.traces[0].y == [
        pytest.approx(0.0),
        pytest.approx(0.0),
        pytest.approx(100 * 0.5 / 9.5),
        pytest.approx(100 * 9.0 / 9.5),
        pytest.approx(0.0),
    ]
    assert "idle" in chart.notes


def test_critical_path_reads_the_aspect_summary() -> None:
    rows = [
        _pair_row("cp_compute_us", 80.0, 90.0),
        _pair_row("cp_exposed_comm_us", 20.0, 40.0),
        _pair_row("mean_slack_us", 5.0, 4.0),
        _pair_row("zero_slack_pct", 30.0, 35.0),
    ]
    aspects = {"dag_critical_path": _summary(rows)}
    chart, _ = viz.build_chart(_context(aspects=aspects), "critical_path")
    length = chart.panels[0]
    assert length.stacked
    assert length.traces[0].y == [80.0, 90.0]
    assert length.traces[1].y == [20.0, 40.0]
    with pytest.raises(viz_spec.NotApplicableError, match="dag_critical_path"):
        viz.build_chart(_context(), "critical_path")


# Statistics.


def test_forest_draws_the_change_with_a_combined_interval() -> None:
    chart, _ = viz.build_chart(
        _context(aspects=_aspects()), "forest", {"top_k": 0}
    )
    traces = chart.panels[0].traces
    significant = next(
        t for t in traces if t.mark == "segment" and t.style["ci"] == 0
    )
    lane = list(significant.y or []).index("demo: step_ms")
    # step_ms rose 40%; the interval is sqrt(0.5^2 + 1.5^2) / 10 = 15.8%.
    half = np.hypot(0.5, 1.5) / 10.0 * 100.0
    assert list(significant.x or [])[lane] == pytest.approx(40.0 - half)
    assert list(significant.x1 or [])[lane] == pytest.approx(40.0 + half)
    only, _ = viz.build_chart(
        _context(aspects=_aspects()), "forest", {"only_significant": True}
    )
    points = [t for t in only.panels[0].traces if t.mark == "scatter"]
    assert [t.name for t in points] == ["significant"]
    assert len(points[0].x or []) == 2


def test_forest_needs_exactly_two_groups() -> None:
    ctx = _context(aspects=_aspects(), labels=("a", "b", "c"))
    with pytest.raises(viz_spec.NotApplicableError, match="two groups"):
        viz.build_chart(ctx, "forest")


def test_volcano_plots_minus_log_p() -> None:
    chart, _ = viz.build_chart(
        _context(aspects=_aspects()), "volcano", {"labels": 0}
    )
    points = {
        name: t
        for t in chart.panels[0].traces
        if t.mark == "scatter" and (name := t.name)
    }
    assert sorted(points["significant"].y or []) == [
        pytest.approx(-np.log10(0.04)),
        pytest.approx(3.0),
    ]
    marks = [t.mark for t in chart.panels[0].traces]
    assert "hline" in marks
    effect, _ = viz.build_chart(
        _context(aspects=_aspects()), "volcano", {"x": "cohens_d"}
    )
    assert effect.panels[0].xlabel.startswith("effect size")


def test_volcano_without_p_values_says_so() -> None:
    rows = [_pair_row("step_ms", 10.0, 12.0)]
    ctx = _context(aspects={"demo": _summary(rows)})
    with pytest.raises(viz_spec.NotApplicableError, match="two or more trials"):
        viz.build_chart(ctx, "volcano")


# Runtime.


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("cudaLaunchKernel", "launch"),
        ("cuLaunchKernelEx", "launch"),
        ("cudaStreamSynchronize", "sync"),
        ("cudaEventSynchronize", "sync"),
        ("cudaStreamWaitEvent", "sync"),
        ("cudaMalloc", "memory"),
        ("cudaMemcpyAsync", "memory"),
        ("cudaGetDevice", "other"),
    ],
)
def test_api_class(name: str, kind: str) -> None:
    assert runtime_api.api_class(name) == kind


def test_runtime_top_ranks_by_total_time() -> None:
    chart, _ = viz.build_chart(_context(_raw()), "runtime_top")
    trace = chart.panels[0].traces[0]
    # Largest total at the top, which is the end of the list.
    assert trace.y == [
        "cudaMalloc",
        "cudaLaunchKernel",
        "cudaStreamSynchronize",
    ]
    assert trace.x == [
        pytest.approx(1.0),
        pytest.approx(4.0),
        pytest.approx(9.0),
    ]


def test_runtime_timeline_puts_calls_and_gpu_in_lanes() -> None:
    chart, _ = viz.build_chart(
        _context(_raw()), "runtime_timeline", {"window_ms": 12}
    )
    panel = chart.panels[0]
    segments = [t for t in panel.traces if t.mark == "segment"]
    lanes = segments[0].style["lanes"]
    assert lanes[0] == "host thread 0"
    assert any(lane.startswith("GPU stream") for lane in lanes)
    assert {t.name for t in segments} >= {"launch", "sync", "GPU kernel"}
    # Segments are clipped to the window.
    assert max(max(t.x1 or []) for t in segments) <= 12.0


def test_runtime_timeline_without_a_call_in_the_window() -> None:
    with pytest.raises(viz_spec.NotApplicableError, match="window"):
        viz.build_chart(
            _context(_raw()), "runtime_timeline", {"start_ms": 5000}
        )


def test_sync_waits_count_only_synchronising_calls() -> None:
    chart, _ = viz.build_chart(_context(_raw()), "sync_waits", {"bins": 5})
    lengths, totals = chart.panels
    # 12 synchronising calls per group.
    assert [sum(t.y or []) for t in lengths.traces] == [12, 12]
    assert totals.traces[0].y == ["cudaStreamSynchronize"]
    no_sync = _raw()
    no_sync["runtime_calls"] = [
        row for row in no_sync["runtime_calls"] if "Sync" not in row["name"]
    ]
    with pytest.raises(viz_spec.NotApplicableError, match="synchronising"):
        viz.build_chart(_context(no_sync), "sync_waits")


# General charts.


def test_radar_puts_the_baseline_on_a_unit_circle() -> None:
    chart, _ = viz.build_chart(
        _context(aspects=_aspects()), "metric_radar", {"max_axes": 4}
    )
    assert chart.panels[0].polar
    base, cand = chart.panels[0].traces
    assert base.y == [1.0] * 4
    ratios = dict(zip(cand.x or [], cand.y or [], strict=True))
    assert ratios["demo: step_ms"] == pytest.approx(1.4)
    few = _context(aspects={"demo": _summary([_pair_row("a", 1.0, 2.0)])})
    with pytest.raises(viz_spec.NotApplicableError, match="three metrics"):
        viz.build_chart(few, "metric_radar")


def test_parallel_coordinates_scale_each_metric_to_its_range() -> None:
    chart, _ = viz.build_chart(_context(aspects=_aspects()), "parallel_coords")
    base, cand = chart.panels[0].traces
    # With two groups each metric runs from one to the other: 0 and 1, and
    # comm_ms, which fell, the other way round.
    lines = dict(
        zip(
            base.x or [],
            zip(base.y or [], cand.y or [], strict=True),
            strict=True,
        )
    )
    assert lines["demo: step_ms"] == (0.0, 1.0)
    assert lines["demo: comm_ms"] == (1.0, 0.0)
    assert len(lines) == 4


def test_ridgeline_orders_curves_and_draws_a_band_each() -> None:
    chart, _ = viz.build_chart(
        _context(_raw()), "ridgeline", {"by": "rank", "max_series": 2}
    )
    traces = chart.panels[0].traces
    assert [t.mark for t in traces].count("band") == 2
    first, second = (t for t in traces if t.mark == "band")
    # The first value is the top ridge.
    assert (first.y or [0])[0] > (second.y or [0])[0]
    assert all(
        top >= base
        for base, top in zip(first.y or [], first.y1 or [], strict=True)
    )


def test_facet_draws_one_panel_per_value_and_passes_options() -> None:
    chart, _ = viz.build_chart(
        _context(_raw()),
        "facet",
        {"kind": "metric_dist", "by": "rank", "inner": "mode=ecdf"},
    )
    assert [panel.title for panel in chart.panels] == ["rank 0", "rank 1"]
    with pytest.raises(ValueError, match="did you mean"):
        viz.build_chart(
            _context(_raw()),
            "facet",
            {"kind": "metric_dist", "inner": "mod=ecdf"},
        )
    with pytest.raises(ValueError, match="key=value"):
        viz.build_chart(
            _context(_raw()), "facet", {"kind": "metric_dist", "inner": "ecdf"}
        )
    with pytest.raises(ValueError, match="kind"):
        viz.build_chart(_context(_raw()), "facet", {"kind": "forest"})


def test_facet_keeps_the_slices_apart() -> None:
    rows = [
        _kernel(
            "base",
            i,
            1.0,
            rank=0,
            category="memcpy",
            name="Memcpy_H2D",
            size=1000,
        )
        for i in range(3)
    ]
    rows += [
        _kernel(
            "base",
            i,
            1.0,
            rank=1,
            category="memcpy",
            name="Memcpy_H2D",
            size=1000,
        )
        for i in range(5)
    ]
    chart, _ = viz.build_chart(
        _context({"kernels": rows}, labels=("base",)),
        "facet",
        {"kind": "memcpy_bandwidth", "by": "rank", "inner": "metric=calls"},
    )
    assert [panel.traces[0].y for panel in chart.panels] == [[3], [5]]


def test_facet_with_nothing_drawable_explains_each_slice() -> None:
    ctx = _context({"kernels": [_kernel("base", 0, 1.0)]}, labels=("base",))
    with pytest.raises(viz_spec.NotApplicableError, match="rank 0"):
        viz.build_chart(ctx, "facet", {"kind": "memcpy_bandwidth"})


def test_new_kinds_do_not_warn_on_a_clean_report() -> None:
    ctx = _context(_raw(), _aspects())
    with warnings.catch_warnings():
        warnings.simplefilter("error", suitability.SuitabilityWarning)
        for kind in (
            "metric_radar",
            "parallel_coords",
            "ridgeline",
            "idle_gaps",
        ):
            viz.build_chart(ctx, kind)
