from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import cta
from cta.adapters.jax.reader import JaxAdapter, select_sources
from cta.adapters.kineto.reader import KinetoAdapter
from cta.core.bundle import resolve_bundle
from cta.core.models import TraceArtifact

REAL = Path(__file__).resolve().parents[1] / "data_real" / "f"


def _meta(devices: int) -> list[dict[str, Any]]:
    ev: list[dict[str, Any]] = [{"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/host:CPU"}}]
    ev += [{"ph": "M", "pid": 10 + d, "name": "process_name", "args": {"name": f"/device:GPU:{d}"}} for d in range(devices)]
    return ev


def _op(dev: int, name: str, ts: float, dur: float, run: int, tf_op: str = "jit(f)/dot_general") -> dict[str, Any]:
    return {"ph": "X", "pid": 10 + dev, "tid": 3, "name": name, "ts": ts, "dur": dur,
            "args": {"hlo_op": name, "hlo_module": "jit_f", "run_id": str(run), "tf_op": tf_op}}  # fmt: skip


def _events(*, annotate: bool, devices: int = 2, steps: int = 3) -> list[dict[str, Any]]:
    ev = _meta(devices)
    for i in range(steps):
        s = i * 1000.0
        if annotate:
            ev.append({"ph": "X", "pid": 1, "tid": 1, "name": "train", "ts": s, "dur": 600.0, "args": {"step_num": str(i)}})
        else:  # host-side execution of the jitted module (what PJRT emits per call)
            ev.append({"ph": "X", "pid": 1, "tid": 1, "name": "CommonPjRtLoadedExecutable::Execute (jit_f)", "ts": s, "dur": 20.0,
                       "args": {"name": "jit_f", "run_id": str(i)}})  # fmt: skip
        for d in range(devices):
            ev.append(_op(d, "fusion.1", s + 10, 100, i))
            ev.append(_op(d, "all-reduce.2", s + 120, 50, i, tf_op="jit(f)/psum"))
            ev.append(_op(d, "copy-start.3", s + 200, 5, i, tf_op=""))
    return ev


def _write(path: Path, events: list[dict[str, Any]], *, head: str = '"displayTimeUnit":"ns",') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as gz:
        gz.write("{" + head + '"traceEvents":' + json.dumps(events) + "}")
    return path


def _ingest(path: Path, out: Path) -> dict[str, pl.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    art = TraceArtifact(format="jax", paths=(path,), world_size=1)
    return {k: pl.read_parquet(v) for k, v in JaxAdapter().ingest(art, out).items()}


def test_annotated_steps_devices_categories_and_scope(tmp_path: Path) -> None:
    t = _ingest(_write(tmp_path / "h.trace.json.gz", _events(annotate=True)), tmp_path / "o")
    k, s = t["kernels"], t["steps"]
    assert s["step"].to_list() == [0, 1, 2]
    assert sorted(k["device"].unique().to_list()) == [0, 1]
    assert set(k["step"].to_list()) == {0, 1, 2}
    cat = dict(zip(k["name"].to_list(), k["category"].to_list(), strict=True))
    assert cat == {"fusion.1": "compute", "all-reduce.2": "comm", "copy-start.3": "memcpy"}
    row = k.filter(pl.col("name") == "all-reduce.2").row(0, named=True)
    assert (row["scope"], row["scope_path"]) == ("psum", "jit(f) > psum")
    assert k.filter(pl.col("name") == "copy-start.3")["scope"][0] == "jit(f)"  # no name stack -> the module, spelled like one


def test_derived_xprof_lines_do_not_double_count(tmp_path: Path) -> None:
    ev = _events(annotate=True, devices=1, steps=1)
    ev += [{"ph": "M", "pid": 10, "tid": 3, "name": "thread_name", "args": {"name": "Stream #7(Compute)"}},
           {"ph": "M", "pid": 10, "tid": 4, "name": "thread_name", "args": {"name": "XLA Ops"}}]  # fmt: skip
    ev += [{**e, "tid": 4} for e in ev if e.get("ph") == "X" and e["pid"] == 10]  # derived copy of every op
    t = _ingest(_write(tmp_path / "h.trace.json.gz", ev), tmp_path / "o")
    assert t["kernels"].height == 3


def test_steps_inferred_from_module_runs_with_warning(tmp_path: Path) -> None:
    with pytest.warns(RuntimeWarning, match="StepTraceAnnotation"):
        t = _ingest(_write(tmp_path / "h.trace.json.gz", _events(annotate=False)), tmp_path / "o")
    assert t["steps"]["step"].to_list() == [0, 1, 2]  # one per host-side execution
    assert (t["kernels"]["step"] >= 0).all()


def test_ops_belong_to_the_step_that_launched_them(tmp_path: Path) -> None:
    ev = _meta(1)
    for i in range(2):
        s = i * 1000.0
        ev.append({"ph": "X", "pid": 1, "tid": 1, "name": "train", "ts": s, "dur": 100.0, "args": {"step_num": str(i)}})
        ev.append({"ph": "X", "pid": 1, "tid": 1, "name": "cuLaunchKernel", "ts": s + 10, "dur": 5.0, "args": {"correlation_id": str(i)}})
    for i in range(2):  # both kernels run after step 1 has started on the host
        op = _op(0, f"fusion.{i}", 1500.0 + i * 10, 5, i)
        op["args"]["correlation_id"] = str(i)
        ev.append(op)
    t = _ingest(_write(tmp_path / "h.trace.json.gz", ev), tmp_path / "o")
    assert t["kernels"].sort("start_ns")["step"].to_list() == [0, 1]


def test_host_launch_calls_fill_the_runtime_table(tmp_path: Path) -> None:
    ev = _events(annotate=True, devices=1, steps=1)
    ev.append({"ph": "X", "pid": 1, "tid": 1, "name": "cuLaunchKernel", "ts": 5.0, "dur": 7.0, "args": {"correlation_id": "1"}})
    ev.append({"ph": "X", "pid": 1, "tid": 1, "name": "cuLaunchKernel", "ts": 15.0, "dur": 3.0, "args": {"correlation_id": "2"}})
    rt = _ingest(_write(tmp_path / "h.trace.json.gz", ev), tmp_path / "o")["runtime"]
    assert rt.to_dicts() == [{"name": "cuLaunchKernel", "duration_ns": 10_000, "calls": 2}]


def test_work_outside_every_step_is_dropped(tmp_path: Path) -> None:
    ev = _events(annotate=True, steps=1) + [_op(0, "autotune.9", 50000.0, 5, 9)]
    t = _ingest(_write(tmp_path / "h.trace.json.gz", ev), tmp_path / "o")
    assert "autotune.9" not in t["kernels"]["name"].to_list()


def test_one_file_per_host_is_claimed_and_kineto_leaves_jax_alone(tmp_path: Path) -> None:
    d = tmp_path / "logdir" / "plugins" / "profile" / "2026_01_01"
    _write(d / "host.trace.json.gz", _events(annotate=True))
    _write(d / "perfetto_trace.json.gz", _events(annotate=True), head='"displayTimeUnit": "ns", ')
    (d / "host.xplane.pb").write_bytes(b"\x00")
    b = resolve_bundle(tmp_path / "logdir")
    assert list(b.artifacts) == ["jax"]
    assert [p.name for p in select_sources(b.artifacts["jax"].paths)] == ["host.trace.json.gz"]
    t = _ingest(d / "perfetto_trace.json.gz", tmp_path / "o")  # an explicitly passed file is accepted too
    assert t["steps"].height == 3
    assert not KinetoAdapter().detect(d / "perfetto_trace.json.gz")


def test_perfetto_then_xplane_fallbacks(tmp_path: Path) -> None:
    d = tmp_path / "a"
    _write(d / "perfetto_trace.json.gz", _events(annotate=True))
    (d / "h.xplane.pb").write_bytes(b"\x00")
    assert [p.name for p in select_sources(resolve_bundle(d).artifacts["jax"].paths)] == ["perfetto_trace.json.gz"]
    only = tmp_path / "b"
    only.mkdir()
    (only / "h.xplane.pb").write_bytes(b"\x00")
    assert [p.name for p in select_sources(resolve_bundle(only).artifacts["jax"].paths)] == ["h.xplane.pb"]


def test_kineto_trace_is_not_claimed_by_jax(tmp_path: Path) -> None:
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"schemaVersion": 1, "traceEvents": []}))
    assert not JaxAdapter().detect(p) and KinetoAdapter().detect(p)


def test_xplane_without_xprof_names_the_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "xprof", None)
    pb = tmp_path / "h.xplane.pb"
    pb.write_bytes(b"\x00")
    with pytest.raises(ImportError, match=r"cta\[jax\]"):
        _ingest(pb, tmp_path / "o")


def test_analyze_end_to_end_and_scope_depth(tmp_path: Path) -> None:
    p = _write(tmp_path / "h.trace.json.gz", _events(annotate=True))
    rep = cta.analyze(p, label="j", cache_dir=tmp_path / "c", scope_depth=1)
    rows = {r["metric"]: r for r in rep.to_polars("latency_throughput").iter_rows(named=True)}
    assert rows["total_steps"]["j"] == 3.0
    assert set(rep.to_polars("kernel_diff", table="by_scope")["scope"].to_list()) == {"jit(f)"}  # depth 1 = the jitted function


@pytest.mark.skipif(not REAL.exists(), reason="real JAX fixtures (tests/data_real/f) not generated")
def test_real_gpu_traces_two_h200(tmp_path: Path) -> None:
    import glob

    from cta.core.registry import ADAPTERS

    def tables(src: Path) -> dict[str, pl.DataFrame]:
        bundle = resolve_bundle(src)
        out = tmp_path / src.name
        out.mkdir()
        art = bundle.artifacts["jax"]
        return {k: pl.read_parquet(v) for k, v in ADAPTERS["jax"].ingest(art, out).items()}

    ann = tables(REAL / "annotated")
    k, s = ann["kernels"], ann["steps"]
    assert s["step"].to_list() == list(range(8))  # StepTraceAnnotation, not the compile/warm-up call
    assert sorted(k["device"].unique().to_list()) == [0, 1]
    assert k.group_by("step").len()["len"].unique().to_list() == [32]  # 16 ops per device and step, no autotune noise
    assert k.filter(pl.col("category") == "comm").height == 16 and "AllReduce" in k.filter(pl.col("category") == "comm")["name"][0]
    assert k["scope_path"].str.starts_with("jit(train_step)").any()

    with pytest.warns(RuntimeWarning, match="StepTraceAnnotation"):
        plain = tables(REAL / "plain")
    assert plain["steps"].height == 9  # eight steps plus the compile/warm-up call, one host-side execution each

    pb = Path(glob.glob(str(REAL / "annotated" / "plugins" / "profile" / "*" / "*.xplane.pb"))[0])
    pytest.importorskip("xprof")
    out = tmp_path / "pb"
    out.mkdir()
    via_pb = {k2: pl.read_parquet(v) for k2, v in JaxAdapter().ingest(TraceArtifact(format="jax", paths=(pb,), world_size=1), out).items()}
    assert via_pb["kernels"].height == k.height  # xprof's derived lines are not counted twice
