"""Tests for the JAX adapter: steps, devices, scopes and source-file choice.

Uses small synthetic XProf trace exports, plus real two-GPU H200 profiles from
`tests/data_real/f` when they have been generated.
"""

from __future__ import annotations

from collections.abc import Mapping
import gzip
import json
import pathlib
from typing import Any

import polars as pl
import pytest

import cta
from cta.adapters.jax import reader as jax_reader
from cta.adapters.kineto import reader as kineto_reader
from cta.core import bundle as core_bundle
from cta.core import models

REAL_JAX_DIR = pathlib.Path(__file__).resolve().parents[1] / "data_real" / "f"


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
    """Returns an XLA op that ran on a GPU, as XProf exports it.

    Args:
        device: GPU index; the op goes to process 10 + device, thread 3.
        name: HLO op name.
        start_us: Start time in microseconds.
        duration_us: Duration in microseconds.
        run_id: Run id of the module execution that issued the op.
        tf_op: Name stack of the op; empty for none.

    Returns:
        The trace event.
    """
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
    """Returns the events of a JAX trace that runs the same ops every step.

    In every step each device runs a fusion, an all-reduce and a copy.

    Args:
        has_step_annotations: Mark each step with a host range that carries
            a `step_num` argument, as `StepTraceAnnotation` does. Otherwise
            only the host-side executions of the jitted module remain.
        devices: Number of GPUs.
        steps: Number of steps, 1000 us apart.

    Returns:
        The trace events, process metadata first.
    """
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
            # The host-side execution of the jitted module, which PJRT emits
            # once per call.
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
    """Writes events as a gzipped Chrome trace document.

    Args:
        path: File to write; missing parent directories are created.
        events: The trace events.
        leading_fields: Raw JSON text put before `traceEvents` in the
            top-level object, trailing comma included.

    Returns:
        `path`.
    """
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
    """Ingests one JAX trace file and reads back every table it produced.

    Args:
        trace_path: Trace file to ingest.
        out_dir: Directory for the Parquet tables; created if missing.

    Returns:
        The tables, keyed by table name.
    """
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
    # Without a name stack the scope is the module, spelled like one.
    copy_kernels = kernels.filter(pl.col("name") == "copy-start.3")
    assert copy_kernels["scope"][0] == "jit(f)"


def test_derived_xprof_lines_do_not_double_count(
    tmp_path: pathlib.Path,
) -> None:
    events = _trace_events(has_step_annotations=True, devices=1, steps=1)
    events += [
        _thread_name_event(10, 3, "Stream #7(Compute)"),
        _thread_name_event(10, 4, "XLA Ops"),
    ]
    # The derived "XLA Ops" line holds a copy of every op.
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
    # One step per host-side execution.
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
    # Both kernels run after step 1 has started on the host.
    for i in range(2):
        op = _device_op(0, f"fusion.{i}", 1500.0 + i * 10, 5, i)
        op["args"]["correlation_id"] = str(i)
        events.append(op)
    trace_path = _write_trace(tmp_path / "h.trace.json.gz", events)
    kernels = _ingest(trace_path, tmp_path / "o")["kernels"]
    assert kernels.sort("start_ns")["step"].to_list() == [0, 1]


def test_host_launch_calls_fill_the_runtime_table(
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
    trace_path = _write_trace(tmp_path / "h.trace.json.gz", events)
    runtime = _ingest(trace_path, tmp_path / "o")["runtime"]
    assert runtime.to_dicts() == [
        {"name": "cuLaunchKernel", "duration_ns": 10_000, "calls": 2}
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
    (profile_dir / "host.xplane.pb").write_bytes(b"\x00")
    bundle = core_bundle.resolve_bundle(tmp_path / "logdir")
    assert list(bundle.artifacts) == ["jax"]
    sources = jax_reader.select_sources(bundle.artifacts["jax"].paths)
    assert [source.name for source in sources] == ["host.trace.json.gz"]
    # An explicitly passed file is accepted too.
    tables = _ingest(perfetto_path, tmp_path / "o")
    assert tables["steps"].height == 3
    assert not kineto_reader.KinetoAdapter().detect(perfetto_path)


def test_perfetto_then_xplane_fallbacks(tmp_path: pathlib.Path) -> None:
    perfetto_dir = tmp_path / "a"
    _write_trace(
        perfetto_dir / "perfetto_trace.json.gz",
        _trace_events(has_step_annotations=True),
    )
    (perfetto_dir / "h.xplane.pb").write_bytes(b"\x00")
    jax_paths = core_bundle.resolve_bundle(perfetto_dir).artifacts["jax"].paths
    sources = jax_reader.select_sources(jax_paths)
    assert [source.name for source in sources] == ["perfetto_trace.json.gz"]
    xplane_only_dir = tmp_path / "b"
    xplane_only_dir.mkdir()
    (xplane_only_dir / "h.xplane.pb").write_bytes(b"\x00")
    jax_paths = (
        core_bundle.resolve_bundle(xplane_only_dir).artifacts["jax"].paths
    )
    sources = jax_reader.select_sources(jax_paths)
    assert [source.name for source in sources] == ["h.xplane.pb"]


def test_kineto_trace_is_not_claimed_by_jax(tmp_path: pathlib.Path) -> None:
    trace_path = tmp_path / "k.json"
    trace_path.write_text(json.dumps({"schemaVersion": 1, "traceEvents": []}))
    assert not jax_reader.JaxAdapter().detect(trace_path)
    assert kineto_reader.KinetoAdapter().detect(trace_path)


def test_xplane_without_xprof_names_the_extra(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "xprof", None)
    xplane_path = tmp_path / "h.xplane.pb"
    xplane_path.write_bytes(b"\x00")
    with pytest.raises(ImportError, match=r"cta\[jax\]"):
        _ingest(xplane_path, tmp_path / "o")


def test_analyze_end_to_end_and_scope_depth(tmp_path: pathlib.Path) -> None:
    trace_path = _write_trace(
        tmp_path / "h.trace.json.gz", _trace_events(has_step_annotations=True)
    )
    report = cta.analyze(
        trace_path, label="j", cache_dir=tmp_path / "c", scope_depth=1
    )
    latency = report.to_polars("latency_throughput")
    rows_by_metric = {
        row["metric"]: row for row in latency.iter_rows(named=True)
    }
    assert rows_by_metric["total_steps"]["j"] == 3.0
    # Depth 1 is the jitted function.
    by_scope = report.to_polars("kernel_diff", table="by_scope")
    assert set(by_scope["scope"].to_list()) == {"jit(f)"}


@pytest.mark.skipif(
    not REAL_JAX_DIR.exists(),
    reason="real JAX fixtures (tests/data_real/f) not generated",
)
def test_real_gpu_traces_two_h200(tmp_path: pathlib.Path) -> None:
    import glob

    from cta.core import registry

    def ingest_profile_dir(source_dir: pathlib.Path) -> dict[str, pl.DataFrame]:
        """Ingests the JAX profile under a directory through the registry."""
        bundle = core_bundle.resolve_bundle(source_dir)
        out_dir = tmp_path / source_dir.name
        out_dir.mkdir()
        artifact = bundle.artifacts["jax"]
        return _read_tables(registry.ADAPTERS["jax"].ingest(artifact, out_dir))

    annotated = ingest_profile_dir(REAL_JAX_DIR / "annotated")
    kernels, steps = annotated["kernels"], annotated["steps"]
    # Steps come from StepTraceAnnotation, not from the compile/warm-up call.
    assert steps["step"].to_list() == list(range(8))
    assert sorted(kernels["device"].unique().to_list()) == [0, 1]
    # 16 ops per device and step, and no autotune noise.
    ops_per_step = kernels.group_by("step").len()["len"]
    assert ops_per_step.unique().to_list() == [32]
    comm_kernels = kernels.filter(pl.col("category") == "comm")
    assert comm_kernels.height == 16
    assert "AllReduce" in comm_kernels["name"][0]
    assert kernels["scope_path"].str.starts_with("jit(train_step)").any()

    with pytest.warns(RuntimeWarning, match="StepTraceAnnotation"):
        plain = ingest_profile_dir(REAL_JAX_DIR / "plain")
    # Eight steps plus the compile/warm-up call: one host-side execution each.
    assert plain["steps"].height == 9

    profiles_dir = REAL_JAX_DIR / "annotated" / "plugins" / "profile"
    xplane_paths = glob.glob(str(profiles_dir / "*" / "*.xplane.pb"))
    xplane_path = pathlib.Path(xplane_paths[0])
    pytest.importorskip("xprof")
    out_dir = tmp_path / "pb"
    out_dir.mkdir()
    xplane_artifact = models.TraceArtifact(
        format="jax", paths=(xplane_path,), world_size=1
    )
    xplane_tables = _read_tables(
        jax_reader.JaxAdapter().ingest(xplane_artifact, out_dir)
    )
    # XProf's derived lines are not counted twice.
    assert xplane_tables["kernels"].height == kernels.height
