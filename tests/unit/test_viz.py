"""Tests for `cta.viz`: plot kinds, rendering, dashboards and the live server.

Covers the plot-kind registry and spec files, the raw tables and SQL that feed
plots, both rendering backends, `cta plot` on the command line, the manifest
`plots:` block, and the filters of the static and the live dashboard.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
import warnings

import polars as pl
import pytest
from typer import testing

import cta
from cta import viz
from cta.cli import app as cli_app
from cta.core import models
from cta.viz import data
from cta.viz import spec as viz_spec

REAL_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data_real"


def _kineto_trace(
    path: pathlib.Path, *, steps: int = 6, slowdown_us: float = 0.0
) -> pathlib.Path:
    """Writes a one-rank Kineto trace with a GEMM and an NCCL kernel per step.

    Args:
        path: File to write.
        steps: Number of `ProfilerStep#i` ranges.
        slowdown_us: Step i and its GEMM kernel last `slowdown_us * i`
            microseconds longer.

    Returns:
        `path`.
    """
    events: list[dict[str, Any]] = []
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
                "ts": step_start_us + 10,
                "dur": 3.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlation": 2 * i},
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm",
                "ts": step_start_us + 20,
                "dur": 100.0 + slowdown_us * i,
                "pid": 0,
                "tid": 7,
                "args": {"correlation": 2 * i, "device": 0, "stream": 7},
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "ncclDevKernel_AllReduce",
                "ts": step_start_us + 200,
                "dur": 50.0,
                "pid": 0,
                "tid": 8,
                "args": {"correlation": 2 * i + 1, "device": 0, "stream": 8},
            }
        )
    path.write_text(
        json.dumps({"traceEvents": events, "distributedInfo": {"rank": 0}})
    )
    return path


@pytest.fixture()
def pair(tmp_path: pathlib.Path) -> models.ComparisonReport:
    """Returns a report comparing a baseline trace with a slower candidate.

    The groups are labelled "base" and "cand"; candidate step i is 40*i us
    slower.
    """
    base_path = _kineto_trace(tmp_path / "a.json")
    cand_path = _kineto_trace(tmp_path / "b.json", slowdown_us=40.0)
    return cta.compare(
        base_path, cand_path, labels=["base", "cand"], cache_dir=tmp_path / "c"
    )


def test_registry_documents_every_kind_and_rejects_unknown_options() -> None:
    kinds_by_name = {kind.name: kind for kind in viz.kind_catalog()}
    assert {
        "breakdown",
        "step_trace",
        "timeline",
        "waterfall",
        "history",
        "custom",
        "kernel_drift",
        "straggler",
        "power_trace",
    } <= set(kinds_by_name)
    assert all(
        kind.title and kind.group and kind.needs is not None
        for kind in kinds_by_name.values()
    )
    with pytest.raises(ValueError, match=r"did you mean 'rolling'"):
        viz_spec.resolve_options(kinds_by_name["step_trace"], {"rollin": 3})
    with pytest.raises(
        ValueError,
        match="Unknown plot kind 'step_tracee'.*Did you mean 'step_trace'",
    ):
        viz.PlotSpec.from_dict({"kind": "step_tracee"})


def test_spec_file_roundtrip_and_coercion(tmp_path: pathlib.Path) -> None:
    spec_path = tmp_path / "p.yaml"
    spec_path.write_text(
        "plots:\n"
        "  - kind: step_trace\n"
        "    rolling: '7'\n"
        "    steps: '1:4'\n"
        "  - {kind: custom, sql: 'select 1 as a', mark: bar, x: a, y: a}\n"
    )
    plot_specs = viz.load_specs(spec_path)
    first_kind = viz.kind_catalog()[0]
    assert [plot_spec.kind for plot_spec in plot_specs] == [
        "step_trace",
        "custom",
    ]
    assert first_kind
    step_trace_options = viz_spec.resolve_options(
        viz_spec.get_kind("step_trace"), plot_specs[0].options
    )
    assert step_trace_options["steps"] == (1, 4)


def test_raw_tables_are_per_rank_relative_and_honour_slicing(
    tmp_path: pathlib.Path,
) -> None:
    report = cta.analyze(
        _kineto_trace(tmp_path / "a.json"),
        label="a",
        warmup_steps=2,
        cache_dir=tmp_path / "c",
    )
    ctx = data.DataContext(report)
    steps = ctx.frame("steps")
    kernels = ctx.frame("kernels")
    # The warm-up steps are gone and time restarts at 0.
    assert steps["idx"].to_list() == [0, 1, 2, 3]
    assert steps["t_ms"].min() == 0.0
    expected_columns = {
        "label",
        "trial",
        "duration_ms",
        "t_ms",
        "end_ms",
        "idx",
    }
    assert expected_columns <= set(kernels.columns)
    sliced_kernels = ctx.frame("kernels", {"steps": (1, 3)})
    assert sliced_kernels["idx"].unique().sort().to_list() == [1, 2]
    comm_kernels = ctx.frame("kernels", {"where": "category = 'comm'"})
    assert comm_kernels["category"].unique().to_list() == ["comm"]
    with pytest.raises(viz_spec.NotApplicableError, match="not in this report"):
        ctx.frame("jtop")


def test_sql_over_named_tables(pair: models.ComparisonReport) -> None:
    kernel_counts = data.DataContext(pair).sql(
        "SELECT label, count(*) AS n FROM kernels GROUP BY label ORDER BY label"
    )
    assert kernel_counts["label"].to_list() == ["base", "cand"]
    assert kernel_counts["n"].to_list() == [12, 12]
    with pytest.raises(ValueError, match="SQL error"):
        data.DataContext(pair).sql("SELECT nope FROM kernels")


@pytest.mark.parametrize(
    "kind",
    [
        "step_trace",
        "step_dist",
        "step_compare",
        "timeline",
        "overlap_timeline",
        "stream_util",
        "pareto",
        "family_share",
        "kernel_scatter",
        "waterfall",
        "scorecard",
        "sweep",
        "kernel_drift",
        "breakdown",
        "topk",
    ],
)
def test_kinds_build_and_render_both_backends(
    pair: models.ComparisonReport, kind: str, tmp_path: pathlib.Path
) -> None:
    ctx = data.DataContext(pair)
    png_path = tmp_path / f"{kind}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chart, _ = viz.build_chart(ctx, kind)
        pair.plot(kind=kind, path=png_path)
    assert chart.panels
    assert png_path.stat().st_size > 1000
    pytest.importorskip("plotly")
    html_path = tmp_path / f"{kind}.html"
    pair.plot(kind=kind, path=html_path)
    assert "plotly" in html_path.read_text()[:200000].lower()


def test_step_trace_flags_a_drifting_run(tmp_path: pathlib.Path) -> None:
    report = cta.analyze(
        _kineto_trace(tmp_path / "a.json", steps=12, slowdown_us=60.0),
        label="drift",
        cache_dir=tmp_path / "c",
    )
    chart, _ = viz.build_chart(data.DataContext(report), "step_trace")
    assert "drifts" in chart.notes
    assert any(
        "trend" in (trace.name or "") for trace in chart.panels[0].traces
    )


def test_inapplicable_plots_raise_not_applicable_which_is_a_value_error(
    pair: models.ComparisonReport,
) -> None:
    inapplicable_kinds = (
        "power_trace",
        "rank_heatmap",
        "straggler",
        "roofline",
        "gates",
        "history",
    )
    for kind in inapplicable_kinds:
        with pytest.raises(ValueError):
            pair.plot(kind=kind)
    with pytest.raises(viz_spec.NotApplicableError):
        viz.build_chart(data.DataContext(pair), "power_trace")


def test_waterfall_adds_up(pair: models.ComparisonReport) -> None:
    chart, _ = viz.build_chart(data.DataContext(pair), "waterfall")
    totals = chart.panels[0].traces[0]
    total_values = totals.y or [0]
    assert len(totals.x or []) >= 3
    # The candidate is slower than the baseline.
    assert total_values[-1] > total_values[0]


def test_gates_plot_uses_configured_thresholds(tmp_path: pathlib.Path) -> None:
    base_path = _kineto_trace(tmp_path / "a.json")
    cand_path = _kineto_trace(tmp_path / "b.json", slowdown_us=60.0)
    report = cta.compare(
        base_path,
        cand_path,
        gates={"mean_step_ms": 5.0},
        cache_dir=tmp_path / "c",
        labels=["a", "b"],
    )
    chart, _ = viz.build_chart(data.DataContext(report), "gates")
    bar_names = [
        trace.name for trace in chart.panels[0].traces if trace.mark == "barh"
    ]
    assert bar_names == ["violated"]


def _write_history_reports(
    root: pathlib.Path, candidate_step_ms: list[float]
) -> str:
    """Writes one saved `report.json` per value, as a history of runs.

    Args:
        root: Directory that gets one `runNN` subdirectory per report.
        candidate_step_ms: The candidate's `mean_step_ms` in each report, one
            day apart.

    Returns:
        A glob pattern that matches every written report.
    """
    for i, step_ms in enumerate(candidate_step_ms):
        run_dir = root / f"run{i:02d}"
        run_dir.mkdir(parents=True)
        latency_row = {
            "metric": "mean_step_ms",
            "baseline": 1.0,
            "candidate": step_ms,
        }
        saved_report = {
            "created_at": f"2026-01-{i + 1:02d}T00:00:00+00:00",
            "groups": ["a", "b"],
            "aspects": {"latency_throughput": [latency_row]},
        }
        (run_dir / "report.json").write_text(json.dumps(saved_report))
    return str(root / "*" / "report.json")


def test_history_control_chart_flags_the_regression(
    tmp_path: pathlib.Path,
) -> None:
    pattern = _write_history_reports(
        tmp_path, [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 10.0, 15.0]
    )
    chart, _ = viz.build_chart(
        data.DataContext(None, history=pattern), "history"
    )
    flagged = [
        trace for trace in chart.panels[0].traces if trace.name == "regression"
    ]
    assert flagged
    assert len(flagged[0].y or []) == 1
    assert (flagged[0].y or [0])[0] == 15.0
    assert "latest 15" in chart.notes
    empty_history = str(tmp_path / "nothing" / "*.json")
    with pytest.raises(viz_spec.NotApplicableError, match="need at least 3"):
        viz.build_chart(
            data.DataContext(None, history=empty_history), "history"
        )


def test_report_json_carries_created_at_for_history(
    pair: models.ComparisonReport, tmp_path: pathlib.Path
) -> None:
    pair.export(tmp_path / "r", formats=["json"])
    saved_report = json.loads((tmp_path / "r" / "report.json").read_text())
    assert saved_report["created_at"] and saved_report["cta_version"]


def test_custom_plot_from_sql_and_report_table(
    pair: models.ComparisonReport, tmp_path: pathlib.Path
) -> None:
    ctx = data.DataContext(pair)
    chart, _ = viz.build_chart(
        ctx,
        "custom",
        {
            "sql": "SELECT idx, label, duration_ms FROM steps",
            "mark": "line",
            "x": "idx",
            "y": "duration_ms",
            "color": "label",
        },
    )
    assert [trace.name for trace in chart.panels[0].traces] == ["base", "cand"]
    chart, _ = viz.build_chart(
        ctx,
        "custom",
        {
            "source": "kernels",
            "mark": "bar",
            "x": "category",
            "y": "duration_ms",
            "agg": "sum",
            "facet": "label",
        },
    )
    assert len(chart.panels) == 2
    chart, _ = viz.build_chart(
        ctx,
        "custom",
        {
            "source": "report:kernel_diff.by_family",
            "mark": "barh",
            "x": "delta",
            "y": "op_family",
        },
    )
    assert chart.panels[0].traces
    with pytest.raises(ValueError, match="not a column"):
        viz.build_chart(
            ctx,
            "custom",
            {
                "source": "kernels",
                "mark": "line",
                "x": "nope",
                "y": "duration_ms",
            },
        )


def test_dashboard_has_tabs_filters_and_skips_what_does_not_apply(
    pair: models.ComparisonReport, tmp_path: pathlib.Path
) -> None:
    pytest.importorskip("plotly")
    dashboard_path = pair.dashboard(tmp_path / "d.html")
    html = dashboard_path.read_text()
    assert 'data-tab="steps"' in html
    assert 'class="f-label"' in html
    assert "plots not applicable" in html
    # plotly.js is embedded, so the page works offline.
    assert "Plotly.newPlot" in html
    assert dashboard_path.stat().st_size > 1_000_000
    cdn_path = pair.dashboard(
        tmp_path / "cdn.html", cdn=True, kinds=["step_trace"]
    )
    assert cdn_path.stat().st_size < 500_000
    assert "cdn.plot.ly" in cdn_path.read_text()


def test_cli_list_kinds_spec_html_and_errors(
    pair: models.ComparisonReport, tmp_path: pathlib.Path
) -> None:
    cli_runner = testing.CliRunner()
    figures_dir = tmp_path / "f"
    cache_dir = tmp_path / "c"
    result = cli_runner.invoke(cli_app.app, ["plot", "--list-kinds"])
    assert result.exit_code == 0
    assert "kernel_drift" in result.output
    assert "history" in result.output
    base_path = _kineto_trace(tmp_path / "a.json")
    cand_path = _kineto_trace(tmp_path / "b.json", slowdown_us=40.0)
    spec_path = tmp_path / "s.yaml"
    spec_path.write_text(
        "- {kind: step_dist, mode: hist}\n"
        "- {kind: pareto, top_k: 3, name: top3}\n"
    )
    result = cli_runner.invoke(
        cli_app.app,
        [
            "plot",
            str(base_path),
            str(cand_path),
            "--spec",
            str(spec_path),
            "-o",
            str(figures_dir),
            "-f",
            "png",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (figures_dir / "step_dist.png").exists()
    assert (figures_dir / "top3.png").exists()
    result = cli_runner.invoke(
        cli_app.app,
        [
            "plot",
            str(base_path),
            "--kind",
            "step_trace",
            "-O",
            "rollin=3",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert result.exit_code == 2
    assert "not accepted" in result.output
    result = cli_runner.invoke(
        cli_app.app,
        [
            "plot",
            str(base_path),
            "--sql",
            "SELECT idx, duration_ms FROM steps",
            "--mark",
            "line",
            "--x",
            "idx",
            "--y",
            "duration_ms",
            "-o",
            str(figures_dir),
            "-f",
            "svg",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert result.exit_code == 0
    assert (figures_dir / "custom.svg").exists()
    history_pattern = _write_history_reports(
        tmp_path / "h", [10, 10.1, 9.9, 10, 10.2]
    )
    result = cli_runner.invoke(
        cli_app.app,
        [
            "plot",
            "--kind",
            "history",
            "--history",
            history_pattern,
            "-o",
            str(figures_dir),
            "-f",
            "png",
        ],
    )
    assert result.exit_code == 0
    assert (figures_dir / "history.png").exists()


def test_manifest_plots_block(tmp_path: pathlib.Path) -> None:
    from cta.core import manifest

    trace_path = _kineto_trace(tmp_path / "a.json")
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(
        f"groups:\n  - {{id: a, label: A, path: {trace_path}}}\n"
        f"config: {{cache_dir: {tmp_path / 'c'}}}\n"
        "plots:\n  - {kind: step_trace, rolling: 3}\n"
    )
    _, config = manifest.load_manifest(manifest_path)
    assert config.plots == [{"kind": "step_trace", "rolling": 3}]
    figures_dir = tmp_path / "f"
    result = testing.CliRunner().invoke(
        cli_app.app,
        ["plot", "-m", str(manifest_path), "-o", str(figures_dir), "-f", "png"],
    )
    assert result.exit_code == 0
    assert (figures_dir / "step_trace.png").exists()


@pytest.mark.skipif(
    not (REAL_DATA_DIR / "e").exists(),
    reason="real DDP fixtures not generated",
)
def test_real_ddp_rank_plots(tmp_path: pathlib.Path) -> None:
    report = cta.analyze(
        REAL_DATA_DIR / "e", label="ddp", cache_dir=tmp_path / "c"
    )
    for kind in ("rank_heatmap", "straggler", "timeline", "step_trace"):
        chart, _ = viz.build_chart(data.DataContext(report), kind)
        assert chart.panels
    kernels = data.DataContext(report).frame("kernels")
    earliest_by_rank = kernels.group_by("rank").agg(pl.col("t_ms").min())
    assert earliest_by_rank.filter(pl.col("t_ms") < 0).is_empty()


def test_steps_only_bundle_gets_the_same_warmup_slicing_as_the_aspects(
    tmp_path: pathlib.Path,
) -> None:
    from cta.synth import tensorrt

    timing_path = tmp_path / "timing.json"
    tensorrt.write_trtexec_timing(timing_path, queries=12)
    report = cta.analyze(
        timing_path, label="trt", warmup_steps=2, cache_dir=tmp_path / "c"
    )
    latency = report.to_polars("latency_throughput")
    rows_by_metric = {
        row["metric"]: row for row in latency.iter_rows(named=True)
    }
    steps = data.DataContext(report).frame("steps")
    assert steps.height == int(rows_by_metric["total_steps"]["trt"])


def test_multi_trial_groups_keep_trials_apart(tmp_path: pathlib.Path) -> None:
    report = cta.analyze(
        [
            _kineto_trace(tmp_path / "a.json", steps=5),
            _kineto_trace(tmp_path / "b.json", steps=5, slowdown_us=30.0),
        ],
        label="g",
        cache_dir=tmp_path / "c",
    )
    ctx = data.DataContext(report)
    chart, _ = viz.build_chart(ctx, "step_trace")
    step_lines = [
        trace
        for trace in chart.panels[0].traces
        if trace.mark == "line"
        and "rolling" not in (trace.name or "")
        and "trend" not in (trace.name or "")
    ]
    # One line per trial: idx restarts per trial, so merging would zig-zag.
    line_names = sorted(trace.name or "" for trace in step_lines)
    assert line_names == ["g #0", "g #1"]
    assert all(
        list(trace.x or []) == sorted(trace.x or []) for trace in step_lines
    )
    timeline, _ = viz.build_chart(ctx, "timeline")
    lanes = {y for trace in timeline.panels[0].traces for y in trace.y or []}
    assert lanes == {
        "t0/r0/gpu0/s7",
        "t0/r0/gpu0/s8",
        "t1/r0/gpu0/s7",
        "t1/r0/gpu0/s8",
    }


def test_ncu_plots_do_not_pool_groups(tmp_path: pathlib.Path) -> None:
    from cta.synth import ncu

    base_path = tmp_path / "a.csv"
    cand_path = tmp_path / "b.csv"
    ncu.write_ncu_csv(base_path)
    ncu.write_ncu_csv(cand_path, gemm_dur_ns=250_000.0)
    report = cta.compare(
        base_path, cand_path, labels=["base", "cand"], cache_dir=tmp_path / "c"
    )
    ctx = data.DataContext(report)
    kernel_counts = ctx.frame("ncu_kernels").group_by("label").len()
    kernels_per_label = kernel_counts.sort("label")["len"].to_list()
    for kind in ("ncu_util", "stalls", "occupancy"):
        chart, _ = viz.build_chart(ctx, kind)
        assert [panel.title for panel in chart.panels] == ["base", "cand"]
    util_chart, _ = viz.build_chart(ctx, "ncu_util")
    # Kernel ids restart per report, so both groups keep all their kernels.
    plotted_per_label = [
        len(panel.traces[0].y or []) for panel in util_chart.panels
    ]
    assert plotted_per_label == kernels_per_label


def test_dashboard_filters_in_the_page(
    pair: models.ComparisonReport, tmp_path: pathlib.Path
) -> None:
    pytest.importorskip("plotly")
    page = pair.dashboard(tmp_path / "d.html").read_text()
    # The group filter also works on category axes.
    assert 'const LABELS=new Set(["base", "cand"])' in page
    assert "keep=cat.map" in page
    assert "meta.step_axes" in page


def _two_gpu_trace(path: pathlib.Path, steps: int = 8) -> pathlib.Path:
    """Writes a one-rank Kineto trace in which two GPUs run a kernel per step.

    In every step device 0 runs a 100 us kernel and device 1 a 300 us one.

    Args:
        path: File to write.
        steps: Number of `ProfilerStep#i` ranges.

    Returns:
        `path`.
    """
    events: list[dict[str, Any]] = []
    for i in range(steps):
        step_start_us = i * 1000.0
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": f"ProfilerStep#{i}",
                "ts": step_start_us,
                "dur": 600.0,
                "pid": 1,
                "tid": 1,
            }
        )
        for device, duration_us in ((0, 100.0), (1, 300.0)):
            correlation = 2 * i + device
            events.append(
                {
                    "ph": "X",
                    "cat": "cuda_runtime",
                    "name": "cudaLaunchKernel",
                    "ts": step_start_us + 10 + device,
                    "dur": 2.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlation": correlation},
                }
            )
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": f"gemm_d{device}",
                    "ts": step_start_us + 20,
                    "dur": duration_us,
                    "pid": device,
                    "tid": 7,
                    "args": {
                        "correlation": correlation,
                        "device": device,
                        "stream": 7,
                    },
                }
            )
    path.write_text(
        json.dumps({"traceEvents": events, "distributedInfo": {"rank": 0}})
    )
    return path


def _metric_value(report: models.ComparisonReport, metric: str) -> float:
    """Returns the first group's value of a metric from the aspect summaries.

    Args:
        report: The report to read.
        metric: Name of the metric row.

    Returns:
        The value in the first group's column of the first aspect summary
        that has a `metric` row.

    Raises:
        KeyError: No aspect summary has the metric.
    """
    for aspect in report.aspects.values():
        summary = aspect.summary
        if metric in summary["metric"].to_list():
            row = summary.filter(pl.col("metric") == metric).row(0, named=True)
            return float(row[report.groups[0].label])
    raise KeyError(metric)


def test_device_and_rank_filters_reach_metrics_and_plots(
    tmp_path: pathlib.Path,
) -> None:
    trace_path = _two_gpu_trace(tmp_path / "t.json")
    full_report = cta.analyze(trace_path, label="g", cache_dir=tmp_path / "c")
    device1_report = cta.analyze(
        trace_path, label="g", devices=[1], cache_dir=tmp_path / "c"
    )
    assert _metric_value(full_report, "kernel_launches") == 16.0
    assert _metric_value(device1_report, "kernel_launches") == 8.0
    assert _metric_value(device1_report, "total_kernel_ms") == pytest.approx(
        8 * 0.3
    )
    # Plots over the raw tables see the same slice.
    device1_kernels = data.DataContext(device1_report).frame("kernels")
    assert set(device1_kernels["device"].to_list()) == {1}
    # There is no rank 3: the slice is empty, which must not crash.
    missing_rank_report = cta.analyze(
        trace_path, label="g", ranks=[3], cache_dir=tmp_path / "c"
    )
    assert (
        "kernel_diff" not in missing_rank_report.aspects
        or _metric_value(missing_rank_report, "kernel_launches") == 0.0
    )


def test_live_session_recomputes_aggregates_for_device_and_step_window(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("plotly")
    from cta.viz import server

    report = cta.analyze(
        _two_gpu_trace(tmp_path / "t.json"), label="g", cache_dir=tmp_path / "c"
    )
    session = server.LiveSession(
        report,
        [
            viz.PlotSpec("family_share"),
            viz.PlotSpec("breakdown"),
            viz.PlotSpec("step_trace"),
        ],
    )
    assert session.choices["devices"] == [0, 1]
    assert session.choices["nsteps"] == 8
    # Without a filter nothing is recomputed.
    assert session.report_for(server.Filters()) is report
    device0_report = session.report_for(server.Filters(devices=(0,)))
    assert _metric_value(device0_report, "total_kernel_ms") == pytest.approx(
        8 * 0.1
    )
    window_report = session.report_for(server.Filters(steps=(2, 5)))
    assert _metric_value(window_report, "total_steps") == 3.0
    device1_window_report = session.report_for(
        server.Filters(devices=(1,), steps=(0, 4))
    )
    assert _metric_value(device1_window_report, "kernel_launches") == 4.0
    breakdown_card = next(
        card for card in session.cards if card.stem == "breakdown"
    )
    response = session.plot(breakdown_card, server.Filters(devices=(0,)))
    fig = json.loads(response["figure"])
    assert fig["data"] and "error" not in response
    # An empty group selection is reported as an error, not a crash.
    assert "error" in session.plot(breakdown_card, server.Filters(labels=()))


def test_live_server_answers_filter_queries_over_http(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("plotly")
    import threading
    import urllib.request

    from cta.viz import server

    report = cta.analyze(
        _two_gpu_trace(tmp_path / "t.json"), label="g", cache_dir=tmp_path / "c"
    )
    httpd, _ = server.make_server(
        report, [viz.PlotSpec("family_share"), viz.PlotSpec("step_trace")]
    )
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        page = urllib.request.urlopen(base_url + "/").read().decode()
        assert 'class="f-device"' in page
        assert "/api/plot" in page
        assert "live:" in page
        meta = json.loads(urllib.request.urlopen(base_url + "/api/meta").read())
        assert meta["devices"] == [0, 1]

        def plotted_values(query: str) -> list[float]:
            """Returns every y value of plot 0 under one filter query."""
            plot_url = f"{base_url}/api/plot?id=0&{query}"
            response = json.loads(urllib.request.urlopen(plot_url).read())
            figure = json.loads(response["figure"])
            return [y for trace in figure["data"] for y in trace["y"]]

        device0_total = sum(plotted_values("devices=0"))
        device1_total = sum(plotted_values("devices=1"))
        all_devices_total = sum(plotted_values(""))
        assert device0_total < device1_total < all_devices_total
        unknown_plot_url = base_url + "/api/plot?id=99"
        unknown_plot = json.loads(
            urllib.request.urlopen(unknown_plot_url).read()
        )
        assert "error" in unknown_plot
    finally:
        httpd.shutdown()
        httpd.server_close()
