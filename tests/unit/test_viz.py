from __future__ import annotations

import json
import warnings
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

import cta
from cta.cli.app import app
from cta.viz import PlotSpec, build_chart, kind_catalog, load_specs
from cta.viz.data import DataContext
from cta.viz.spec import NotApplicable, resolve_options

REAL = Path(__file__).resolve().parents[1] / "data_real"


def _trace(path: Path, *, steps: int = 6, slow: float = 0.0, ranks: int = 1) -> Path:
    """Kineto trace: `steps` steps, one compute and one NCCL kernel each; step i gets `slow*i` extra microseconds."""
    ev: list[dict] = []
    for i in range(steps):
        s = i * 1000.0
        ev.append({"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{i}", "ts": s, "dur": 500.0 + slow * i, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": s + 10, "dur": 3.0, "pid": 1, "tid": 1, "args": {"correlation": 2 * i}})
        ev.append({"ph": "X", "cat": "kernel", "name": "gemm", "ts": s + 20, "dur": 100.0 + slow * i, "pid": 0, "tid": 7, "args": {"correlation": 2 * i, "device": 0, "stream": 7}})
        ev.append({"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce", "ts": s + 200, "dur": 50.0, "pid": 0, "tid": 8, "args": {"correlation": 2 * i + 1, "device": 0, "stream": 8}})
    path.write_text(json.dumps({"traceEvents": ev, "distributedInfo": {"rank": 0}}))
    return path


@pytest.fixture()
def pair(tmp_path: Path):  # type: ignore[no-untyped-def]
    a, b = _trace(tmp_path / "a.json"), _trace(tmp_path / "b.json", slow=40.0)
    return cta.compare(a, b, labels=["base", "cand"], cache_dir=tmp_path / "c")


def test_registry_documents_every_kind_and_rejects_unknown_options() -> None:
    kinds = {k.name: k for k in kind_catalog()}
    assert {"breakdown", "step_trace", "timeline", "waterfall", "history", "custom", "kernel_drift", "straggler", "power_trace"} <= set(kinds)
    assert all(k.title and k.group and k.needs is not None for k in kinds.values())
    with pytest.raises(ValueError, match=r"did you mean 'rolling'"):
        resolve_options(kinds["step_trace"], {"rollin": 3})
    with pytest.raises(ValueError, match="Unknown plot kind 'step_tracee'.*Did you mean 'step_trace'"):
        PlotSpec.from_dict({"kind": "step_tracee"})


def test_spec_file_roundtrip_and_coercion(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text("plots:\n  - kind: step_trace\n    rolling: '7'\n    steps: '1:4'\n  - {kind: custom, sql: 'select 1 as a', mark: bar, x: a, y: a}\n")
    specs = load_specs(f)
    kind = kind_catalog()[0]
    assert [s.kind for s in specs] == ["step_trace", "custom"] and kind
    assert resolve_options(__import__("cta.viz.spec", fromlist=["get_kind"]).get_kind("step_trace"), specs[0].options)["steps"] == (1, 4)


def test_raw_tables_are_per_rank_relative_and_honour_slicing(tmp_path: Path) -> None:
    rep = cta.analyze(_trace(tmp_path / "a.json"), label="a", warmup_steps=2, cache_dir=tmp_path / "c")
    ctx = DataContext(rep)
    steps, kern = ctx.frame("steps"), ctx.frame("kernels")
    assert steps["idx"].to_list() == [0, 1, 2, 3] and steps["t_ms"].min() == 0.0  # warm-up steps are gone, time restarts at 0
    assert {"label", "trial", "duration_ms", "t_ms", "end_ms", "idx"} <= set(kern.columns)
    assert ctx.frame("kernels", {"steps": (1, 3)})["idx"].unique().sort().to_list() == [1, 2]
    assert ctx.frame("kernels", {"where": "category = 'comm'"})["category"].unique().to_list() == ["comm"]
    with pytest.raises(NotApplicable, match="not in this report"):
        ctx.frame("jtop")


def test_sql_over_named_tables(pair) -> None:  # type: ignore[no-untyped-def]
    out = DataContext(pair).sql("SELECT label, count(*) AS n FROM kernels GROUP BY label ORDER BY label")
    assert out["label"].to_list() == ["base", "cand"] and out["n"].to_list() == [12, 12]
    with pytest.raises(ValueError, match="SQL error"):
        DataContext(pair).sql("SELECT nope FROM kernels")


@pytest.mark.parametrize("kind", ["step_trace", "step_dist", "step_compare", "timeline", "overlap_timeline", "stream_util", "pareto",
                                  "family_share", "kernel_scatter", "waterfall", "scorecard", "sweep", "kernel_drift", "breakdown", "topk"])
def test_kinds_build_and_render_both_backends(pair, kind: str, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    ctx = DataContext(pair)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chart, opts = build_chart(ctx, kind)
        pair.plot(kind=kind, path=tmp_path / f"{kind}.png")
    assert chart.panels and (tmp_path / f"{kind}.png").stat().st_size > 1000
    pytest.importorskip("plotly")
    pair.plot(kind=kind, path=tmp_path / f"{kind}.html")
    assert "plotly" in (tmp_path / f"{kind}.html").read_text()[:200000].lower()


def test_step_trace_flags_a_drifting_run(tmp_path: Path) -> None:
    rep = cta.analyze(_trace(tmp_path / "a.json", steps=12, slow=60.0), label="drift", cache_dir=tmp_path / "c")
    chart, _ = build_chart(DataContext(rep), "step_trace")
    assert "drifts" in chart.notes and any("trend" in (t.name or "") for t in chart.panels[0].traces)


def test_inapplicable_plots_raise_not_applicable_which_is_a_value_error(pair) -> None:  # type: ignore[no-untyped-def]
    for kind in ("power_trace", "rank_heatmap", "straggler", "roofline", "gates", "history"):
        with pytest.raises(ValueError):
            pair.plot(kind=kind)
    with pytest.raises(NotApplicable):
        build_chart(DataContext(pair), "power_trace")


def test_waterfall_adds_up(pair) -> None:  # type: ignore[no-untyped-def]
    chart, _ = build_chart(DataContext(pair), "waterfall")
    totals = chart.panels[0].traces[0]
    assert len(totals.x or []) >= 3 and (totals.y or [0])[-1] > (totals.y or [0])[0]  # candidate is slower than baseline


def test_gates_plot_uses_configured_thresholds(tmp_path: Path) -> None:
    a, b = _trace(tmp_path / "a.json"), _trace(tmp_path / "b.json", slow=60.0)
    rep = cta.compare(a, b, gates={"mean_step_ms": 5.0}, cache_dir=tmp_path / "c", labels=["a", "b"])
    chart, _ = build_chart(DataContext(rep), "gates")
    assert [t.name for t in chart.panels[0].traces if t.mark == "barh"] == ["violated"]


def _write_reports(root: Path, values: list[float]) -> str:
    for i, v in enumerate(values):
        d = root / f"run{i:02d}"
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps({
            "created_at": f"2026-01-{i + 1:02d}T00:00:00+00:00", "groups": ["a", "b"],
            "aspects": {"latency_throughput": [{"metric": "mean_step_ms", "baseline": 1.0, "candidate": v}]}}))  # fmt: skip
    return str(root / "*" / "report.json")


def test_history_control_chart_flags_the_regression(tmp_path: Path) -> None:
    pattern = _write_reports(tmp_path, [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 10.0, 15.0])
    chart, _ = build_chart(DataContext(None, history=pattern), "history")
    flagged = [t for t in chart.panels[0].traces if t.name == "regression"]
    assert flagged and len(flagged[0].y or []) == 1 and (flagged[0].y or [0])[0] == 15.0
    assert "latest 15" in chart.notes
    with pytest.raises(NotApplicable, match="need at least 3"):
        build_chart(DataContext(None, history=str(tmp_path / "nothing" / "*.json")), "history")


def test_report_json_carries_created_at_for_history(pair, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pair.export(tmp_path / "r", formats=["json"])
    doc = json.loads((tmp_path / "r" / "report.json").read_text())
    assert doc["created_at"] and doc["cta_version"]


def test_custom_plot_from_sql_and_report_table(pair, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    ctx = DataContext(pair)
    chart, _ = build_chart(ctx, "custom", {"sql": "SELECT idx, label, duration_ms FROM steps", "mark": "line", "x": "idx", "y": "duration_ms", "color": "label"})
    assert [t.name for t in chart.panels[0].traces] == ["base", "cand"]
    chart, _ = build_chart(ctx, "custom", {"source": "kernels", "mark": "bar", "x": "category", "y": "duration_ms", "agg": "sum", "facet": "label"})
    assert len(chart.panels) == 2
    chart, _ = build_chart(ctx, "custom", {"source": "report:kernel_diff.by_family", "mark": "barh", "x": "delta", "y": "op_family"})
    assert chart.panels[0].traces
    with pytest.raises(ValueError, match="not a column"):
        build_chart(ctx, "custom", {"source": "kernels", "mark": "line", "x": "nope", "y": "duration_ms"})


def test_dashboard_has_tabs_filters_and_skips_what_does_not_apply(pair, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("plotly")
    out = pair.dashboard(tmp_path / "d.html")
    html = out.read_text()
    assert 'data-tab="steps"' in html and 'class="f-label"' in html and "plots not applicable" in html
    assert "Plotly.newPlot" in html and out.stat().st_size > 1_000_000  # plotly.js embedded: works offline
    cdn = pair.dashboard(tmp_path / "cdn.html", cdn=True, kinds=["step_trace"])
    assert cdn.stat().st_size < 500_000 and "cdn.plot.ly" in cdn.read_text()


def test_cli_list_kinds_spec_html_and_errors(pair, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    r = runner.invoke(app, ["plot", "--list-kinds"])
    assert r.exit_code == 0 and "kernel_drift" in r.output and "history" in r.output
    a, b = _trace(tmp_path / "a.json"), _trace(tmp_path / "b.json", slow=40.0)
    spec = tmp_path / "s.yaml"
    spec.write_text("- {kind: step_dist, mode: hist}\n- {kind: pareto, top_k: 3, name: top3}\n")
    r = runner.invoke(app, ["plot", str(a), str(b), "--spec", str(spec), "-o", str(tmp_path / "f"), "-f", "png", "--cache-dir", str(tmp_path / "c")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "f" / "step_dist.png").exists() and (tmp_path / "f" / "top3.png").exists()
    r = runner.invoke(app, ["plot", str(a), "--kind", "step_trace", "-O", "rollin=3", "--cache-dir", str(tmp_path / "c")])
    assert r.exit_code == 2 and "not accepted" in r.output
    r = runner.invoke(app, ["plot", str(a), "--sql", "SELECT idx, duration_ms FROM steps", "--mark", "line", "--x", "idx", "--y", "duration_ms",
                            "-o", str(tmp_path / "f"), "-f", "svg", "--cache-dir", str(tmp_path / "c")])
    assert r.exit_code == 0 and (tmp_path / "f" / "custom.svg").exists()
    r = runner.invoke(app, ["plot", "--kind", "history", "--history", _write_reports(tmp_path / "h", [10, 10.1, 9.9, 10, 10.2]), "-o", str(tmp_path / "f"), "-f", "png"])
    assert r.exit_code == 0 and (tmp_path / "f" / "history.png").exists()


def test_manifest_plots_block(tmp_path: Path) -> None:
    from cta.core.manifest import load_manifest

    a = _trace(tmp_path / "a.json")
    m = tmp_path / "m.yaml"
    m.write_text(f"groups:\n  - {{id: a, label: A, path: {a}}}\nconfig: {{cache_dir: {tmp_path / 'c'}}}\nplots:\n  - {{kind: step_trace, rolling: 3}}\n")
    _, cfg = load_manifest(m)
    assert cfg.plots == [{"kind": "step_trace", "rolling": 3}]
    r = CliRunner().invoke(app, ["plot", "-m", str(m), "-o", str(tmp_path / "f"), "-f", "png"])
    assert r.exit_code == 0 and (tmp_path / "f" / "step_trace.png").exists()


@pytest.mark.skipif(not (REAL / "e").exists(), reason="real DDP fixtures not generated")
def test_real_ddp_rank_plots(tmp_path: Path) -> None:
    rep = cta.analyze(REAL / "e", label="ddp", cache_dir=tmp_path / "c")
    for kind in ("rank_heatmap", "straggler", "timeline", "step_trace"):
        chart, _ = build_chart(DataContext(rep), kind)
        assert chart.panels
    k = DataContext(rep).frame("kernels")
    assert k.group_by("rank").agg(pl.col("t_ms").min()).filter(pl.col("t_ms") < 0).is_empty()


def test_steps_only_bundle_gets_the_same_warmup_slicing_as_the_aspects(tmp_path: Path) -> None:
    from cta.synth.generators import write_trtexec_timing

    p = tmp_path / "timing.json"
    write_trtexec_timing(p, queries=12)
    rep = cta.analyze(p, label="trt", warmup_steps=2, cache_dir=tmp_path / "c")
    rows = {r["metric"]: r for r in rep.to_polars("latency_throughput").iter_rows(named=True)}
    assert DataContext(rep).frame("steps").height == int(rows["total_steps"]["trt"])


def test_multi_trial_groups_keep_trials_apart(tmp_path: Path) -> None:
    rep = cta.analyze([_trace(tmp_path / "a.json", steps=5), _trace(tmp_path / "b.json", steps=5, slow=30.0)], label="g", cache_dir=tmp_path / "c")
    ctx = DataContext(rep)
    chart, _ = build_chart(ctx, "step_trace")
    lines = [t for t in chart.panels[0].traces if t.mark == "line" and "rolling" not in (t.name or "") and "trend" not in (t.name or "")]
    assert sorted(t.name or "" for t in lines) == ["g #0", "g #1"]  # one line per trial: idx restarts per trial, so merging would zig-zag
    assert all(list(t.x or []) == sorted(t.x or []) for t in lines)
    tl, _ = build_chart(ctx, "timeline")
    assert {y for tr in tl.panels[0].traces for y in tr.y or []} == {"t0/r0/gpu0/s7", "t0/r0/gpu0/s8", "t1/r0/gpu0/s7", "t1/r0/gpu0/s8"}


def test_ncu_plots_do_not_pool_groups(tmp_path: Path) -> None:
    from cta.synth.generators import write_ncu_csv

    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_ncu_csv(a)
    write_ncu_csv(b, gemm_dur_ns=250_000.0)
    rep = cta.compare(a, b, labels=["base", "cand"], cache_dir=tmp_path / "c")
    ctx = DataContext(rep)
    per_label = ctx.frame("ncu_kernels").group_by("label").len().sort("label")["len"].to_list()
    for kind in ("ncu_util", "stalls", "occupancy"):
        chart, _ = build_chart(ctx, kind)
        assert [p.title for p in chart.panels] == ["base", "cand"]
    util, _ = build_chart(ctx, "ncu_util")
    assert [len(p.traces[0].y or []) for p in util.panels] == per_label  # kernel ids restart per report: both groups keep all kernels


def test_dashboard_filters_in_the_page(pair, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("plotly")
    page = pair.dashboard(tmp_path / "d.html").read_text()
    assert "const LABELS=new Set([\"base\", \"cand\"])" in page and "keep=cat.map" in page  # category-axis group filter
    assert "meta.step_axes" in page


def _two_gpu_trace(path: Path, steps: int = 8) -> Path:
    """One rank, two GPUs: device 0 runs a 100 us kernel per step, device 1 a 300 us one."""
    ev: list[dict] = []
    for i in range(steps):
        s = i * 1000.0
        ev.append({"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{i}", "ts": s, "dur": 600.0, "pid": 1, "tid": 1})
        for dev, dur in ((0, 100.0), (1, 300.0)):
            c = 2 * i + dev
            ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": s + 10 + dev, "dur": 2.0, "pid": 1, "tid": 1, "args": {"correlation": c}})
            ev.append({"ph": "X", "cat": "kernel", "name": f"gemm_d{dev}", "ts": s + 20, "dur": dur, "pid": dev, "tid": 7, "args": {"correlation": c, "device": dev, "stream": 7}})
    path.write_text(json.dumps({"traceEvents": ev, "distributedInfo": {"rank": 0}}))
    return path


def _metric(rep, metric: str) -> float:  # type: ignore[no-untyped-def]
    for name in rep.aspects:
        s = rep.aspects[name].summary
        if metric in s["metric"].to_list():
            return float(s.filter(pl.col("metric") == metric).row(0, named=True)[rep.groups[0].label])
    raise KeyError(metric)


def test_analysis_can_be_restricted_to_devices_ranks_and_the_filter_reaches_plots(tmp_path: Path) -> None:
    p = _two_gpu_trace(tmp_path / "t.json")
    full = cta.analyze(p, label="g", cache_dir=tmp_path / "c")
    d1 = cta.analyze(p, label="g", devices=[1], cache_dir=tmp_path / "c")
    assert _metric(full, "kernel_launches") == 16.0 and _metric(d1, "kernel_launches") == 8.0
    assert _metric(d1, "total_kernel_ms") == pytest.approx(8 * 0.3)
    assert set(DataContext(d1).frame("kernels")["device"].to_list()) == {1}  # raw-table plots see the same slice
    r3 = cta.analyze(p, label="g", ranks=[3], cache_dir=tmp_path / "c")  # no such rank: empty slice, not a crash
    assert "kernel_diff" not in r3.aspects or _metric(r3, "kernel_launches") == 0.0


def test_live_session_recomputes_aggregates_for_device_and_step_window(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    from cta.viz.server import Filters, LiveSession

    rep = cta.analyze(_two_gpu_trace(tmp_path / "t.json"), label="g", cache_dir=tmp_path / "c")
    s = LiveSession(rep, [PlotSpec("family_share"), PlotSpec("breakdown"), PlotSpec("step_trace")])
    assert s.choices["devices"] == [0, 1] and s.choices["nsteps"] == 8
    assert s.report_for(Filters()) is rep  # no filter: nothing is recomputed
    assert _metric(s.report_for(Filters(devices=(0,))), "total_kernel_ms") == pytest.approx(8 * 0.1)
    assert _metric(s.report_for(Filters(steps=(2, 5))), "total_steps") == 3.0
    assert _metric(s.report_for(Filters(devices=(1,), steps=(0, 4))), "kernel_launches") == 4.0
    card = next(c for c in s.cards if c.stem == "breakdown")
    out = s.plot(card, Filters(devices=(0,)))
    fig = json.loads(out["figure"])
    assert fig["data"] and "error" not in out
    assert "error" in s.plot(card, Filters(labels=()))  # nothing selected is reported, not a crash


def test_live_server_answers_filter_queries_over_http(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    import threading
    import urllib.request

    from cta.viz.server import make_server

    rep = cta.analyze(_two_gpu_trace(tmp_path / "t.json"), label="g", cache_dir=tmp_path / "c")
    httpd, session = make_server(rep, [PlotSpec("family_share"), PlotSpec("step_trace")])
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert 'class="f-device"' in page and "/api/plot" in page and "live:" in page
        meta = json.loads(urllib.request.urlopen(base + "/api/meta").read())
        assert meta["devices"] == [0, 1]

        def bars(q: str) -> list[float]:
            j = json.loads(urllib.request.urlopen(f"{base}/api/plot?id=0&{q}").read())
            return [y for t in json.loads(j["figure"])["data"] for y in t["y"]]

        assert sum(bars("devices=0")) < sum(bars("devices=1")) < sum(bars(""))
        assert "error" in json.loads(urllib.request.urlopen(base + "/api/plot?id=99").read())
    finally:
        httpd.shutdown()
        httpd.server_close()
