"""Tests of the C++ Kineto reader (`cta._native.read_kineto`).

The tables the adapter builds from each fixture were hashed with the Python
reader this code replaced, so a mismatch means the adapter no longer produces
what it did. The fixtures cover the cases a synthetic trace does not: escaped
quotes and braces in names, half-nanosecond timestamps, upper-case categories,
missing fields, repeated correlation ids, GPU events that come before the call
that launched them, text pids and tids, and a bare list as the top level.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
from typing import Any

import numpy as np
import polars as pl
import pytest

from cta import _native
from cta import models
from cta import table_schemas
from cta.adapters.kineto import reader
from cta.synth import kineto as synth_kineto

# Microseconds. The second base is a Unix-epoch timestamp, which is too large
# for a double to hold in nanoseconds.
_BASES_US = {"small": 1_000_000.0, "epoch": 1_700_000_000_000_000.0}


def _x(
    name: str,
    cat: str,
    ts: float,
    dur: float | None,
    pid: Any = 0,
    tid: Any = 7,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds one complete ("X") event; None leaves a field out."""
    event: dict[str, Any] = {"ph": "X", "cat": cat, "name": name, "ts": ts}
    if dur is not None:
        event["dur"] = dur
    if pid is not None:
        event["pid"] = pid
    if tid is not None:
        event["tid"] = tid
    if args is not None:
        event["args"] = args
    return event


def adversarial_events(base_us: float) -> list[Any]:
    """Returns trace events that exercise the reader's edge cases."""
    t = base_us
    return [
        # GPU work first: its launches come later in the file.
        _x(
            'void gemm<float>(int, "x") {tail}',
            "kernel",
            t + 60,
            20.0,
            args={"correlation": 100, "stream": 7, "bytes": 0, "device": 1},
        ),
        _x(
            "Memcpy HtoD (Pageable -> Device)",
            "Gpu_Memcpy",
            t + 70.0005,
            5.0,
            args={"bytes": 4096, "correlation": 101},
        ),
        _x(
            "ncclKernel_AllReduce_RING_LL",
            "KERNEL",
            t + 90,
            30.25,
            pid=2,
            tid=9,
            args={"correlation": 102, "stream": 0, "device": 1.0},
        ),
        _x("kernel without args", "kernel", t + 130, 4.0),
        _x("zero duration", "kernel", t + 140, 0.0),
        _x("negative duration", "kernel", t + 141, -1.0),
        _x("no duration", "kernel", t + 142, None),
        _x(
            "ProfilerStep#9 kernel",
            "kernel",
            t + 150,
            3.0,
            args={"correlation": 108},
        ),
        _x(
            "memset é日本",
            "gpu_memset",
            t + 160,
            2.0,
            args={"bytes": "512", "stream": "3", "correlation": 107},
        ),
        # The step range, and the GPU-side copy of it, which does not count.
        _x("ProfilerStep#7", "user_annotation", t, 1000.0, pid=1, tid=10),
        _x("ProfilerStep#7", "gpu_user_annotation", t + 0.5, 999.0),
        _x("profilerstep# 8 later", "Operator", t + 1000, 400.0, pid=1, tid=10),
        # CPU ops: an outer op and a nested one on one thread.
        _x(
            'aten::mm "quoted" {brace} \\path',
            "CPU_OP",
            t + 10,
            500.0,
            pid=1,
            tid=10,
        ),
        _x("aten::inner", "cpu_op", t + 20, 100.0, pid=1, tid=10),
        _x(
            "annotated",
            "user_annotation",
            t + 5,
            600.0,
            pid="Trace",
            tid="thread 1",
        ),
        # Launches: a repeated id (the later call wins), a driver call, a call
        # on a text thread, a call without args.
        _x(
            "cudaLaunchKernel",
            "Cuda_Runtime",
            t + 30.0005,
            5.0004,
            1,
            10,
            {"correlation": 100},
        ),
        _x(
            "cudaLaunchKernel",
            "cuda_runtime",
            t + 40,
            6.0,
            1,
            10,
            {"correlation": 100},
        ),
        _x(
            "cudaMemcpyAsync",
            "cuda_runtime",
            t + 41,
            3.5,
            1,
            10,
            {"correlation": 101},
        ),
        _x(
            "cudaLaunchKernel",
            "cuda_runtime",
            t + 42,
            4.0,
            2,
            9,
            {"correlation": 102},
        ),
        _x(
            "cuLaunchKernel",
            "cuda_driver",
            t + 43,
            2.0,
            1,
            10,
            {"correlation": 104},
        ),
        _x(
            "cudaMemsetAsync",
            "cuda_runtime",
            t + 44,
            2.0,
            "Trace",
            "thread 1",
            {"correlation": 107},
        ),
        _x("cudaDeviceSynchronize", "cuda_runtime", t + 45, 9.0, 1, 10),
        _x(
            "cudaLaunchKernel",
            "cuda_runtime",
            t + 46,
            1.0,
            1,
            10,
            {"correlation": 108},
        ),
        _x("unmatched", "kernel", t + 170, 1.0, args={"correlation": 999}),
        # Ignored events.
        {"ph": "B", "cat": "cpu_op", "name": "begin", "ts": t},
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "main"}},
        {"ph": "i", "name": "instant", "ts": t, "s": "g"},
        5,
        None,
        "text",
        [1, 2],
    ]


def adversarial_trace(
    path: pathlib.Path,
    *,
    base_us: float,
    layout: str = "after",
    ensure_ascii: bool = True,
) -> pathlib.Path:
    """Writes the adversarial events in one of several top-level layouts.

    Args:
        path: File to write; a `.gz` suffix gzips it.
        base_us: Start time of the trace in microseconds.
        layout: "after" (distributedInfo after traceEvents), "before",
            "float_rank" (rank 3.0, which is not an integer), "none" (no
            distributedInfo) or "list" (a bare list of events).
        ensure_ascii: Whether non-ASCII characters are written as escapes.

    Returns:
        `path`.
    """
    events = adversarial_events(base_us)
    info = {
        "backend": "nccl",
        "rank": 3,
        "world_size": 4,
        "nested": {"a": [1, "}"]},
    }
    extra = {
        "schemaVersion": 1,
        "deviceProperties": [{"name": 'A100 "x" {y}', "id": 0}],
    }
    top: Any
    if layout == "list":
        top = events
    elif layout == "before":
        top = {"distributedInfo": info, **extra, "traceEvents": events}
    elif layout == "float_rank":
        top = {"traceEvents": events, "distributedInfo": {**info, "rank": 3.0}}
    elif layout == "none":
        top = {"traceEvents": events, **extra}
    else:
        top = {"traceEvents": events, **extra, "distributedInfo": info}
    text = json.dumps(top, indent=1, ensure_ascii=ensure_ascii)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _fixtures(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Writes every fixture trace and returns them by name."""
    out: dict[str, pathlib.Path] = {}
    for base_name, base in _BASES_US.items():
        for layout in ("after", "before", "float_rank", "none", "list"):
            name = f"adv_{base_name}_{layout}"
            out[name] = adversarial_trace(
                tmp_path / f"{name}.json", base_us=base, layout=layout
            )
    out["adv_small_gz"] = adversarial_trace(
        tmp_path / "adv_small_gz.json.gz", base_us=_BASES_US["small"]
    )
    out["adv_small_utf8"] = adversarial_trace(
        tmp_path / "adv_small_utf8.json",
        base_us=_BASES_US["small"],
        ensure_ascii=False,
    )
    out["compat_3"] = synth_kineto.write_trace(
        tmp_path / "compat_3.json", steps=3
    )
    out["compat_fp8"] = synth_kineto.write_trace(
        tmp_path / "compat_fp8.json.gz", steps=7, kernel_variant="fp8"
    )
    out["large"] = pathlib.Path(tmp_path / "large.json")
    synth_kineto.write_large_trace(
        out["large"], target_bytes=3 << 20, seed=5, rank=2, world_size=4
    )
    out["large_info_first"] = pathlib.Path(tmp_path / "large_first.json")
    synth_kineto.write_large_trace(
        out["large_info_first"],
        target_bytes=1 << 20,
        seed=6,
        distributed_info_after=False,
    )
    return out


def _tables(
    path: pathlib.Path, out_dir: pathlib.Path
) -> dict[str, pl.DataFrame]:
    """Ingests one trace file and reads the tables back."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = models.TraceArtifact(
        format="kineto", paths=(path,), world_size=1
    )
    paths = reader.KinetoAdapter().ingest(artifact, out_dir)
    return {name: pl.read_parquet(p) for name, p in paths.items()}


def table_digest(tables: dict[str, pl.DataFrame]) -> str:
    """Returns a hash of the three tables that is stable across runs."""
    digest = hashlib.sha256()
    for name in ("kernels", "steps", "runtime"):
        digest.update(name.encode())
        digest.update(tables[name].write_csv().encode())
    return digest.hexdigest()


# Hashes of the tables the Python reader produced from each fixture.
GOLDEN: dict[str, str] = {
    "adv_small_after": (
        "c21c3c7f2af0a6bc4db2af0bf45bc30e19eb684f3ecfaaf26027c9846b3718cc"
    ),
    "adv_small_before": (
        "c21c3c7f2af0a6bc4db2af0bf45bc30e19eb684f3ecfaaf26027c9846b3718cc"
    ),
    "adv_small_float_rank": (
        "edf4d24f36740fc3f52f59eb320bdcf111604bbb48bd46d39bf4d0469b1bd6ef"
    ),
    "adv_small_none": (
        "edf4d24f36740fc3f52f59eb320bdcf111604bbb48bd46d39bf4d0469b1bd6ef"
    ),
    "adv_small_list": (
        "edf4d24f36740fc3f52f59eb320bdcf111604bbb48bd46d39bf4d0469b1bd6ef"
    ),
    "adv_epoch_after": (
        "7f9565e6c20bc2ebd1f70a325d831f90f17066afab9c43edc272ef7d91262500"
    ),
    "adv_epoch_before": (
        "7f9565e6c20bc2ebd1f70a325d831f90f17066afab9c43edc272ef7d91262500"
    ),
    "adv_epoch_float_rank": (
        "5e85ca900fa78ebe9973dc7983fdc49ab244181b9ac32c4b9f76b5e33fb8f601"
    ),
    "adv_epoch_none": (
        "5e85ca900fa78ebe9973dc7983fdc49ab244181b9ac32c4b9f76b5e33fb8f601"
    ),
    "adv_epoch_list": (
        "5e85ca900fa78ebe9973dc7983fdc49ab244181b9ac32c4b9f76b5e33fb8f601"
    ),
    "adv_small_gz": (
        "c21c3c7f2af0a6bc4db2af0bf45bc30e19eb684f3ecfaaf26027c9846b3718cc"
    ),
    "adv_small_utf8": (
        "c21c3c7f2af0a6bc4db2af0bf45bc30e19eb684f3ecfaaf26027c9846b3718cc"
    ),
    "compat_3": (
        "1109cd0ea9c693824ded0396728a77659ba5b63b2eafc0715794954192bd8b0e"
    ),
    "compat_fp8": (
        "10cf4be74d883a1f34a74c8c74c60293320e222c78e66c9e0616e52fb9adee00"
    ),
    "large": (
        "c2f16d9eee731fecb0a9816b3501c210b5e34e9249fdd69888bed411583855ec"
    ),
    "large_info_first": (
        "d688290d7f5dc95367d23a79ef9a0d05b41efade00b956eca0c223fda7be030f"
    ),
}


def test_tables_match_the_python_reader_they_replaced(
    tmp_path: pathlib.Path,
) -> None:
    fixtures = _fixtures(tmp_path)
    actual = {
        name: table_digest(_tables(path, tmp_path / f"out_{name}"))
        for name, path in fixtures.items()
    }
    assert actual == GOLDEN


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        if key == "timing":  # seconds spent, different every run
            continue
        if isinstance(value, np.ndarray):
            if not np.array_equal(value, b[key], equal_nan=True):
                return False
        elif value != b[key]:
            return False
    return True


def test_result_does_not_depend_on_threads_chunk_or_block_size(
    tmp_path: pathlib.Path,
) -> None:
    for name, path in _fixtures(tmp_path).items():
        whole = _native.read_kineto(str(path), 1, 1 << 40, 1 << 30)
        # Tiny blocks cut the text at every kind of position: inside strings,
        # after a backslash, between a key and its value.
        tiny = 1 if path.stat().st_size < 100_000 else 3
        for threads, chunk, block in (
            (1, 64, tiny),
            (3, 100, 7),
            (8, 257, 64),
            (4, 5000, 4099),
        ):
            split = _native.read_kineto(str(path), threads, chunk, block)
            assert _same(whole, split), (name, threads, chunk, block)


def test_nesting_parents_match_a_plain_loop() -> None:
    rng = np.random.default_rng(3)
    starts = np.sort(rng.integers(0, 1000, size=500)).astype(np.int64)
    ends = starts + rng.integers(1, 200, size=500)
    order = np.lexsort((-ends, starts))
    starts, ends = starts[order], ends[order]
    expected = []
    open_ranges: list[int] = []
    for i in range(len(starts)):
        while open_ranges and ends[open_ranges[-1]] <= starts[i]:
            open_ranges.pop()
        expected.append(open_ranges[-1] if open_ranges else -1)
        open_ranges.append(i)
    assert _native.nesting_parents(starts, ends).tolist() == expected
    assert _native.nesting_parents(starts[:0], ends[:0]).tolist() == []


def test_distributed_info_rank_is_read_wherever_it_sits(
    tmp_path: pathlib.Path,
) -> None:
    ranks = {
        layout: _native.read_kineto(
            str(
                adversarial_trace(
                    tmp_path / f"{layout}.json",
                    base_us=1_000_000.0,
                    layout=layout,
                )
            ),
            2,
            200,
        )["rank"]
        for layout in ("after", "before", "float_rank", "none", "list")
    }
    assert ranks == {
        "after": 3,
        "before": 3,
        "float_rank": None,
        "none": None,
        "list": None,
    }


def test_late_launches_and_repeated_correlation_ids(
    tmp_path: pathlib.Path,
) -> None:
    path = adversarial_trace(tmp_path / "a.json", base_us=1_000_000.0)
    cols = _native.read_kineto(str(path), 2, 300)
    names = cols["names"]
    gemm = next(
        i
        for i, code in enumerate(cols["gpu_name"].tolist())
        if names[code].startswith("void gemm")
    )
    # The kernel precedes both launches with correlation id 100 in the file;
    # the second launch (at +40 us) replaces the first.
    assert cols["gpu_launch_ns"][gemm] == (1_000_000.0 + 40) * 1000
    unmatched = next(
        i
        for i, code in enumerate(cols["gpu_name"].tolist())
        if names[code] == "unmatched"
    )
    assert np.isnan(cols["gpu_launch_ns"][unmatched])
    assert cols["gpu_launch_thread"][unmatched] == -1


def test_gpu_rows_follow_the_rules_of_the_python_reader(
    tmp_path: pathlib.Path,
) -> None:
    path = adversarial_trace(tmp_path / "a.json", base_us=1_000_000.0)
    cols = _native.read_kineto(str(path), 2, 300)
    names = cols["names"]
    rows = {names[code]: i for i, code in enumerate(cols["gpu_name"].tolist())}
    # A missing, zero or negative duration drops the event.
    assert "zero duration" not in rows
    assert "negative duration" not in rows
    assert "no duration" not in rows
    # Stream and device fall back to the thread and process ids.
    memcpy = rows["Memcpy HtoD (Pageable -> Device)"]
    assert cols["gpu_stream"][memcpy] == 7
    assert cols["gpu_device"][memcpy] == 0
    assert cols["gpu_bytes"][memcpy] == 4096
    assert cols["gpu_category"][memcpy] == 2
    nccl = rows["ncclKernel_AllReduce_RING_LL"]
    assert cols["gpu_stream"][nccl] == 0  # present as 0, so no fallback
    assert (
        cols["gpu_device"][nccl] == 2
    )  # 1.0 is not an integer: fall back to pid
    assert cols["gpu_category"][nccl] == 1
    # Numeric text counts as a number.
    memset = rows["memset é日本"]
    assert cols["gpu_bytes"][memset] == 512
    assert cols["gpu_stream"][memset] == 3
    # A kernel named like a step is a GPU row and also a step.
    assert "ProfilerStep#9 kernel" in rows
    assert 9 in cols["step_id"].tolist()
    # The GPU-side copy of a step does not count, so ids 7, 8 and 9 appear once.
    assert sorted(cols["step_id"].tolist()) == [7, 8, 9]


def test_half_nanoseconds_round_to_even(tmp_path: pathlib.Path) -> None:
    # 62.5 ns and 187.5 ns are exact in binary; Python's round() takes the
    # even neighbour, 62 and 188.
    events = [_x("a", "kernel", 0.0625, 1.0), _x("b", "kernel", 0.1875, 1.0)]
    path = tmp_path / "half.json"
    path.write_text(json.dumps({"traceEvents": events}))
    cols = _native.read_kineto(str(path), 1, 1 << 20)
    assert cols["gpu_start"].tolist() == [62, 188]
    assert cols["gpu_end"].tolist() == [1062, 1188]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        '{"traceEvents": [',
        '{"traceEvents": [{"ph": "X"}',
        "[1, 2",
        "nope",
    ],
)
def test_broken_json_is_rejected(tmp_path: pathlib.Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="."):
        _native.read_kineto(str(path), 2, 100)


def test_bad_event_is_rejected_and_the_reader_stops(
    tmp_path: pathlib.Path,
) -> None:
    events = [_x(f"k{i}", "kernel", 1.0 + i, 1.0) for i in range(200)]
    events.insert(120, {"ph": "X", "cat": "kernel", "name": "bad", "ts": None})
    path = tmp_path / "bad_event.json"
    path.write_text(json.dumps({"traceEvents": events}))
    with pytest.raises(ValueError, match="numbers"):
        _native.read_kineto(str(path), 4, 200)


def test_a_missing_file_is_an_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(RuntimeError, match="cannot open"):
        _native.read_kineto(str(tmp_path / "missing.json"), 1, 100)


def test_a_trace_without_events_gives_empty_tables(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "empty.json"
    path.write_text('{"traceEvents": [], "distributedInfo": {"rank": 1}}')
    tables = _tables(path, tmp_path / "out")
    assert tables["kernels"].height == 0
    assert tables["steps"].height == 0
    assert tables["runtime"].height == 0


def test_every_runtime_call_is_listed(tmp_path: pathlib.Path) -> None:
    path = adversarial_trace(tmp_path / "a.json", base_us=1_000_000.0)
    cols = _native.read_kineto(str(path), 2, 300)
    calls = len(cols["rcall_name"])
    assert calls > 0
    for key in (
        "rcall_start",
        "rcall_end",
        "rcall_thread",
        "rcall_correlation",
    ):
        assert len(cols[key]) == calls
    names = [cols["names"][code] for code in cols["rcall_name"].tolist()]
    duration = (cols["rcall_end"] - cols["rcall_start"]).tolist()
    totals: dict[str, list[int]] = {}
    for name, ns in zip(names, duration, strict=True):
        entry = totals.setdefault(name, [0, 0])
        entry[0] += ns
        entry[1] += 1
    summary = {
        cols["names"][code]: [int(ns), int(count)]
        for code, ns, count in zip(
            cols["runtime_name"].tolist(),
            cols["runtime_duration"].tolist(),
            cols["runtime_calls"].tolist(),
            strict=True,
        )
    }
    assert totals == summary
    # A call without a correlation id says so with -1.
    assert -1 in cols["rcall_correlation"].tolist()


def test_runtime_calls_do_not_depend_on_threads_or_chunks(
    tmp_path: pathlib.Path,
) -> None:
    path = adversarial_trace(tmp_path / "a.json", base_us=1_000_000.0)
    one = _native.read_kineto(str(path), 1, 1 << 30)
    many = _native.read_kineto(str(path), 3, 200, 64)
    assert one["names"] == many["names"]
    for key in (
        "rcall_name",
        "rcall_start",
        "rcall_end",
        "rcall_thread",
        "rcall_correlation",
    ):
        assert one[key].tolist() == many[key].tolist(), key


def test_runtime_calls_table_is_written(tmp_path: pathlib.Path) -> None:
    path = adversarial_trace(tmp_path / "a.json", base_us=1_000_000.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    artifact = models.TraceArtifact(
        format="kineto", paths=(path,), world_size=1
    )
    written = reader.KinetoAdapter().ingest(artifact, out_dir)
    table = pl.read_parquet(written["runtime_calls"])
    assert table.schema == table_schemas.RUNTIME_CALLS
    assert table.height > 0
    assert table["start_ns"].is_sorted()
