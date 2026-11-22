"""Tests for the JAX adapter: steps, devices, scopes, formats and tables.

Uses synthetic XProf/XPlane/Perfetto traces, plus real two-GPU H200 profiles
from ``tests/data_real/f`` when they have been generated.
"""

from __future__ import annotations

from collections.abc import Mapping
import glob
import gzip
import json
import pathlib
import sys
from typing import Any

import polars as pl
import pytest

import cta
from cta import models
from cta import plugins
from cta import table_schemas
from cta.adapters.jax import reader as jax_reader
from cta.adapters.kineto import reader as kineto_reader
from cta.ingest import discovery
from cta.synth import jax as synth_jax

REAL_JAX_DIR = pathlib.Path(__file__).resolve().parents[2] / "data_real" / "f"


def _process_name_events(devices: int) -> list[dict[str, Any]]:
    """Returns `process_name` metadata for the host and `devices` GPUs.

    The host is process 1 and GPU d is process 10 + d.
    """
    events: list[dict[str, Any]] = [
        {
            "ph": "M",
            "pid": 1,
            "name": "process_name",
            "args": {"name": "/host:CPU"},
        }
    ]
    events += [
        {
            "ph": "M",
            "pid": 10 + device,
            "name": "process_name",
            "args": {"name": f"/device:GPU:{device}"},
        }
        for device in range(devices)
    ]
    return events


def _thread_name_event(pid: int, tid: int, name: str) -> dict[str, Any]:
    """Returns a `thread_name` metadata event."""
    return {
        "ph": "M",
        "pid": pid,
        "tid": tid,
        "name": "thread_name",
        "args": {"name": name},
    }


def _device_op(
    device: int,
    name: str,
    start_us: float,
    duration_us: float,
    run_id: int,
    tf_op: str = "jit(f)/dot_general",
) -> dict[str, Any]:
    """Returns an XLA op that ran on a GPU, as XProf exports it."""
    op_args = {
        "hlo_op": name,
        "hlo_module": "jit_f",
        "run_id": str(run_id),
        "tf_op": tf_op,
    }
    return {
        "ph": "X",
        "pid": 10 + device,
        "tid": 3,
        "name": name,
        "ts": start_us,
        "dur": duration_us,
        "args": op_args,
    }


def _trace_events(
    *, has_step_annotations: bool, devices: int = 2, steps: int = 3
) -> list[dict[str, Any]]:
    """Returns the events of a JAX trace that runs the same ops every step."""
    events = _process_name_events(devices)
    for i in range(steps):
        step_start_us = i * 1000.0
        if has_step_annotations:
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 1,
                    "name": "train",
                    "ts": step_start_us,
                    "dur": 600.0,
                    "args": {"step_num": str(i)},
                }
            )
        else:
            events.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 1,
                    "name": "CommonPjRtLoadedExecutable::Execute (jit_f)",
                    "ts": step_start_us,
                    "dur": 20.0,
                    "args": {"name": "jit_f", "run_id": str(i)},
                }
            )
        for device in range(devices):
            events.append(
                _device_op(device, "fusion.1", step_start_us + 10, 100, i)
            )
            events.append(
                _device_op(
                    device,
                    "all-reduce.2",
                    step_start_us + 120,
                    50,
                    i,
                    tf_op="jit(f)/psum",
                )
            )
            events.append(
                _device_op(
                    device, "copy-start.3", step_start_us + 200, 5, i, tf_op=""
                )
            )
    return events


def _write_trace(
    path: pathlib.Path,
    events: list[dict[str, Any]],
    *,
    leading_fields: str = '"displayTimeUnit":"ns",',
) -> pathlib.Path:
    """Writes events as a gzipped Chrome trace document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(
            "{" + leading_fields + '"traceEvents":' + json.dumps(events) + "}"
        )
    return path


def _read_tables(
    table_paths: Mapping[str, pathlib.Path],
) -> dict[str, pl.DataFrame]:
    """Reads the Parquet tables an adapter wrote, keyed by table name."""
    return {
        table: pl.read_parquet(parquet_path)
        for table, parquet_path in table_paths.items()
    }


def _ingest(
    trace_path: pathlib.Path, out_dir: pathlib.Path
) -> dict[str, pl.DataFrame]:
    """Ingests one JAX trace file and reads back every table it produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = models.TraceArtifact(
        format="jax", paths=(trace_path,), world_size=1
    )
    return _read_tables(jax_reader.JaxAdapter().ingest(artifact, out_dir))


def test_annotated_steps_devices_categories_and_scope(
    tmp_path: pathlib.Path,
) -> None:
    trace_path = _write_trace(
        tmp_path / "h.trace.json.gz", _trace_events(has_step_annotations=True)
    )
    tables = _ingest(trace_path, tmp_path / "o")
    kernels, steps = tables["kernels"], tables["steps"]
    assert steps["step"].to_list() == [0, 1, 2]
    assert sorted(kernels["device"].unique().to_list()) == [0, 1]
    assert set(kernels["step"].to_list()) == {0, 1, 2}
    category_by_name = dict(
        zip(
            kernels["name"].to_list(),
            kernels["category"].to_list(),
            strict=True,
        )
    )
    assert category_by_name == {
        "fusion.1": "compute",
        "all-reduce.2": "comm",
        "copy-start.3": "memcpy",
    }
    row = kernels.filter(pl.col("name") == "all-reduce.2").row(0, named=True)
    assert (row["scope"], row["scope_path"]) == ("psum", "jit(f) > psum")
    copy_kernels = kernels.filter(pl.col("name") == "copy-start.3")
    assert copy_kernels["scope"][0] == "jit(f)"
    # Non-NCCL XLA collective populates network_ops with api="xla".
    net = tables["network_ops"]
    assert net.schema == table_schemas.NETWORK_OPS
    assert net.height == 6
    assert set(net["api"].to_list()) == {"xla"}
    assert set(net["op"].to_list()) == {"allreduce"}


def test_derived_xprof_lines_do_not_double_count(
    tmp_path: pathlib.Path,
) -> None:
    events = _trace_events(has_step_annotations=True, devices=1, steps=1)
    events += [
        _thread_name_event(10, 3, "Stream #7(Compute)"),
        _thread_name_event(10, 4, "XLA Ops"),
    ]
    derived_ops = [
        {**event, "tid": 4}
        for event in events
        if event.get("ph") == "X" and event["pid"] == 10
    ]
    events += derived_ops
    trace_path = _write_trace(tmp_path / "h.trace.json.gz", events)
    tables = _ingest(trace_path, tmp_path / "o")
    assert tables["kernels"].height == 3


def test_steps_inferred_from_module_runs_with_warning(
    tmp_path: pathlib.Path,
) -> None:
    events = _trace_events(has_step_annotations=False)
    with pytest.warns(RuntimeWarning, match="StepTraceAnnotation"):
        tables = _ingest(
            _write_trace(tmp_path / "h.trace.json.gz", events),
            tmp_path / "o",
        )
    assert tables["steps"]["step"].to_list() == [0, 1, 2]
    assert (tables["kernels"]["step"] >= 0).all()


def test_ops_belong_to_the_step_that_launched_them(
    tmp_path: pathlib.Path,
) -> None:
    events = _process_name_events(1)
    for i in range(2):
        step_start_us = i * 1000.0
        events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "name": "train",
                "ts": step_start_us,
                "dur": 100.0,
                "args": {"step_num": str(i)},
            }
        )
        events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "name": "cuLaunchKernel",
                "ts": step_start_us + 10,
                "dur": 5.0,
                "args": {"correlation_id": str(i)},
            }
        )
    for i in range(2):
        op = _device_op(0, f"fusion.{i}", 1500.0 + i * 10, 5, i)
        op["args"]["correlation_id"] = str(i)
        events.append(op)
    trace_path = _write_trace(tmp_path / "h.trace.json.gz", events)
    kernels = _ingest(trace_path, tmp_path / "o")["kernels"]
    assert kernels.sort("start_ns")["step"].to_list() == [0, 1]


def test_host_launch_calls_fill_runtime_and_runtime_calls_tables(
    tmp_path: pathlib.Path,
) -> None:
    events = _trace_events(has_step_annotations=True, devices=1, steps=1)
    events.append(
        {
            "ph": "X",
            "pid": 1,
            "tid": 1,
            "name": "cuLaunchKernel",
            "ts": 5.0,
            "dur": 7.0,
            "args": {"correlation_id": "1"},
        }
    )
    events.append(
        {
            "ph": "X",
            "pid": 1,
            "tid": 1,
            "name": "cuLaunchKernel",
            "ts": 15.0,
            "dur": 3.0,
            "args": {"correlation_id": "2"},
        }
    )
    # Out-of-step launch: excluded from runtime, step=-1 in runtime_calls.
    events.append(
        {
            "ph": "X",
            "pid": 1,
            "tid": 1,
            "name": "cuLaunchKernel",
            "ts": 5000.0,
            "dur": 9.0,
            "args": {"correlation_id": "3"},
        }
    )
    trace_path = _write_trace(tmp_path / "h.trace.json.gz", events)
    tables = _ingest(trace_path, tmp_path / "o")
    assert tables["runtime"].to_dicts() == [
        {"name": "cuLaunchKernel", "duration_ns": 10_000, "calls": 2}
    ]
    calls = tables["runtime_calls"]
    assert calls.schema == table_schemas.RUNTIME_CALLS
    assert calls.height == 3
    assert calls["step"].to_list() == [0, 0, -1]
    assert calls["correlation"].to_list() == [1, 2, 3]


def test_runtime_counts_calls_inside_an_enclosing_step(
    tmp_path: pathlib.Path,
) -> None:
    events = _process_name_events(1)
    for step_num, start_us, duration_us in (
        (0, 0.0, 1000.0),
        (1, 100.0, 100.0),
    ):
        events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "name": "train",
                "ts": start_us,
                "dur": duration_us,
                "args": {"step_num": str(step_num)},
            }
        )
    for call_start_us in (500.0, 1500.0):
        events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "name": "cuLaunchKernel",
                "ts": call_start_us,
                "dur": 4.0,
                "args": {"correlation_id": str(int(call_start_us))},
            }
        )
    events.append(_device_op(0, "fusion.1", 10.0, 5, 0))
    trace_path = _write_trace(tmp_path / "h.trace.json.gz", events)
    runtime = _ingest(trace_path, tmp_path / "o")["runtime"]
    assert runtime.to_dicts() == [
        {"name": "cuLaunchKernel", "duration_ns": 4_000, "calls": 1}
    ]


def test_work_outside_every_step_is_dropped(tmp_path: pathlib.Path) -> None:
    autotune_op = _device_op(0, "autotune.9", 50000.0, 5, 9)
    events = _trace_events(has_step_annotations=True, steps=1) + [autotune_op]
    trace_path = _write_trace(tmp_path / "h.trace.json.gz", events)
    kernels = _ingest(trace_path, tmp_path / "o")["kernels"]
    assert "autotune.9" not in kernels["name"].to_list()


def test_one_file_per_host_is_claimed_and_kineto_leaves_jax_alone(
    tmp_path: pathlib.Path,
) -> None:
    profile_dir = tmp_path / "logdir" / "plugins" / "profile" / "2026_01_01"
    _write_trace(
        profile_dir / "host.trace.json.gz",
        _trace_events(has_step_annotations=True),
    )
    perfetto_path = _write_trace(
        profile_dir / "perfetto_trace.json.gz",
        _trace_events(has_step_annotations=True),
        leading_fields='"displayTimeUnit": "ns", ',
    )
    synth_jax.write_pftrace(profile_dir / "trace.pftrace")
    synth_jax.write_xplane(profile_dir / "host.xplane.pb")
    bundle = discovery.resolve_bundle(tmp_path / "logdir")
    assert list(bundle.artifacts) == ["jax"]
    sources = jax_reader.select_sources(bundle.artifacts["jax"].paths)
    assert [source.name for source in sources] == ["host.trace.json.gz"]
    tables = _ingest(perfetto_path, tmp_path / "o")
    assert tables["steps"].height == 3
    assert not kineto_reader.KinetoAdapter().detect(perfetto_path)


def test_perfetto_pftrace_then_xplane_fallbacks(tmp_path: pathlib.Path) -> None:
    perfetto_dir = tmp_path / "a"
    _write_trace(
        perfetto_dir / "perfetto_trace.json.gz",
        _trace_events(has_step_annotations=True),
    )
    synth_jax.write_pftrace(perfetto_dir / "trace.pftrace")
    (perfetto_dir / "h.xplane.pb").write_bytes(b"\x00")
    jax_paths = discovery.resolve_bundle(perfetto_dir).artifacts["jax"].paths
    sources = jax_reader.select_sources(jax_paths)
    assert [source.name for source in sources] == ["perfetto_trace.json.gz"]

    pftrace_dir = tmp_path / "b"
    synth_jax.write_pftrace(pftrace_dir / "trace.pftrace.gz")
    (pftrace_dir / "h.xplane.pb").write_bytes(b"\x00")
    jax_paths = discovery.resolve_bundle(pftrace_dir).artifacts["jax"].paths
    sources = jax_reader.select_sources(jax_paths)
    assert [source.name for source in sources] == ["trace.pftrace.gz"]

    xplane_only_dir = tmp_path / "c"
    xplane_only_dir.mkdir()
    (xplane_only_dir / "h.xplane.pb").write_bytes(b"\x00")
    jax_paths = discovery.resolve_bundle(xplane_only_dir).artifacts["jax"].paths
    sources = jax_reader.select_sources(jax_paths)
    assert [source.name for source in sources] == ["h.xplane.pb"]


def test_pftrace_detect_validates_wire_header(tmp_path: pathlib.Path) -> None:
    valid = synth_jax.write_pftrace(tmp_path / "good.pftrace")
    valid_gz = synth_jax.write_pftrace(tmp_path / "good.pftrace.gz")
    garbage = tmp_path / "not_proto.pftrace"
    garbage.write_text("plain text, not a protobuf")
    empty = tmp_path / "empty.pftrace"
    empty.write_bytes(b"")
    adapter = jax_reader.JaxAdapter()
    assert adapter.detect(valid)
    assert adapter.detect(valid_gz)
    assert not adapter.detect(garbage)
    assert not adapter.detect(empty)


def test_kineto_trace_is_not_claimed_by_jax(tmp_path: pathlib.Path) -> None:
    trace_path = tmp_path / "k.json"
    trace_path.write_text(json.dumps({"schemaVersion": 1, "traceEvents": []}))
    assert not jax_reader.JaxAdapter().detect(trace_path)
    assert kineto_reader.KinetoAdapter().detect(trace_path)


def test_xplane_reads_natively_without_xprof(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "xprof", None)
    xplane_path = synth_jax.write_xplane(
        tmp_path / "h.xplane.pb", steps=3, devices=2
    )
    tables = _ingest(xplane_path, tmp_path / "o")
    assert tables["steps"]["step"].to_list() == [0, 1, 2]
    assert tables["kernels"].height == 24
    assert tables["runtime"].height == 3
    assert tables["runtime_calls"].height == 25
    assert tables["network_ops"].height == 12
    assert set(tables["network_ops"]["api"].to_list()) == {"nccl", "p2p"}


def test_analyze_end_to_end_scope_depth_and_network_aspect(
    tmp_path: pathlib.Path,
) -> None:
    trace_path = synth_jax.write_trace(
        tmp_path / "h.trace.json.gz", steps=3, devices=2
    )
    report = cta.analyze(
        trace_path, label="j", cache_dir=tmp_path / "c", scope_depth=1
    )
    latency = report.to_polars("latency")
    rows_by_metric = {
        row["metric"]: row for row in latency.iter_rows(named=True)
    }
    assert rows_by_metric["total_steps"]["j"] == 3.0
    by_scope = report.to_polars("kernels", table="by_scope")
    assert set(by_scope["scope"].to_list()) == {"jit(train_step)"}
    net = report.to_polars("network")
    assert net.height > 0
    assert any("nccl" in m for m in net["metric"].to_list())
    assert any("p2p" in m for m in net["metric"].to_list())


@pytest.mark.skipif(
    not REAL_JAX_DIR.exists(),
    reason="real JAX fixtures (tests/data_real/f) not generated",
)
def test_real_gpu_traces_two_h200(tmp_path: pathlib.Path) -> None:
    def ingest_profile_dir(source_dir: pathlib.Path) -> dict[str, pl.DataFrame]:
        bundle = discovery.resolve_bundle(source_dir)
        out_dir = tmp_path / source_dir.name
        out_dir.mkdir()
        artifact = bundle.artifacts["jax"]
        return _read_tables(plugins.ADAPTERS["jax"].ingest(artifact, out_dir))

    annotated = ingest_profile_dir(REAL_JAX_DIR / "annotated")
    kernels, steps = annotated["kernels"], annotated["steps"]
    assert steps["step"].to_list() == list(range(8))
    assert sorted(kernels["device"].unique().to_list()) == [0, 1]
    ops_per_step = kernels.group_by("step").len()["len"]
    assert ops_per_step.unique().to_list() == [32]
    comm_kernels = kernels.filter(pl.col("category") == "comm")
    assert comm_kernels.height == 16
    assert "AllReduce" in comm_kernels["name"][0]
    assert kernels["scope_path"].str.starts_with("jit(train_step)").any()
    assert annotated["runtime_calls"].height == 4764
    assert annotated["network_ops"].height == 16
    assert set(annotated["network_ops"]["api"].to_list()) == {"nccl"}
    assert set(annotated["network_ops"]["op"].to_list()) == {"allreduce"}

    with pytest.warns(RuntimeWarning, match="StepTraceAnnotation"):
        plain = ingest_profile_dir(REAL_JAX_DIR / "plain")
    assert plain["steps"].height == 9
    assert plain["network_ops"].height == 114

    # Native .xplane.pb reading requires no xprof package and matches JSON
    # across all five tables.
    profiles_dir = REAL_JAX_DIR / "annotated" / "plugins" / "profile"
    xplane_paths = glob.glob(str(profiles_dir / "*" / "*.xplane.pb"))
    xplane_path = pathlib.Path(xplane_paths[0])
    out_dir = tmp_path / "pb"
    out_dir.mkdir()
    xplane_artifact = models.TraceArtifact(
        format="jax", paths=(xplane_path,), world_size=1
    )
    xplane_tables = _read_tables(
        jax_reader.JaxAdapter().ingest(xplane_artifact, out_dir)
    )
    for table_name in (
        "kernels",
        "steps",
        "runtime",
        "runtime_calls",
        "network_ops",
    ):
        assert (
            xplane_tables[table_name].write_csv()
            == annotated[table_name].write_csv()
        ), table_name
