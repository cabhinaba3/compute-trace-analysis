"""Tests of the C++ Nsight Systems reader (`cta._native.read_nsys_sqlite`).

The tables the adapter builds from each export were hashed with the Python
reader this code replaced, so a mismatch means the adapter no longer produces
what it did. The exports written here hold what generated ones never do:
kernels and copies launched from several threads, steps and scopes that nest
and overlap, launches that are missing or repeated, columns and tables an
export may lack, NULL fields, and a trace with no step markers.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import pathlib
import random
import sqlite3
import warnings

import numpy as np
import polars as pl
import pytest

from cta import _native
from cta.adapters.nsys import reader
from cta.adapters.nsys import runtime_calls
from cta.core import intervals
from cta.core import models
from cta.synth import nsys as synth_nsys

KERNEL = (
    "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (demangledName INTEGER, "
    "shortName INTEGER, start INTEGER, end INTEGER, deviceId INTEGER, "
    "streamId INTEGER, correlationId INTEGER)"
)
MEMCPY = (
    "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (copyKind INTEGER, start INTEGER,"
    " end INTEGER, bytes INTEGER, deviceId INTEGER, streamId INTEGER, "
    "correlationId INTEGER)"
)
MEMSET = (
    "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER, "
    "bytes INTEGER, deviceId INTEGER, streamId INTEGER, correlationId INTEGER)"
)
RUNTIME = (
    "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (nameId INTEGER, start INTEGER, "
    "end INTEGER, correlationId INTEGER, globalTid INTEGER)"
)
NVTX = (
    "CREATE TABLE NVTX_EVENTS (eventType INTEGER, start INTEGER, end INTEGER, "
    "text TEXT, textId INTEGER, globalTid INTEGER)"
)
STRINGS = "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)"
STRING_ROWS = [
    (1, "void gemm<float>(int)"),
    (2, "ncclDevKernel_AllReduce_Sum_f16_RING_LL"),
    (3, "cudaLaunchKernel"),
    (4, "ProfilerStep#4"),
    (5, "block"),
    (6, "NCCL in capitals"),
    (7, "step 9"),
]


def _build(
    path: pathlib.Path, fill: Callable[[sqlite3.Connection], None]
) -> pathlib.Path:
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    fill(connection)
    connection.commit()
    connection.close()
    return path


def _labels(c: sqlite3.Connection) -> None:
    """Steps and scopes on two threads, with launches from both."""
    c.executescript(";".join([STRINGS, KERNEL, MEMCPY, MEMSET, RUNTIME, NVTX]))
    c.executemany("INSERT INTO StringIds VALUES (?, ?)", STRING_ROWS)
    # (demangled, short, start, end, device, stream, correlation)
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
        [
            (1, 1, 120, 180, 0, 7, 1),  # inside inner on thread 1
            (2, 2, 130, 190, 0, 8, 2),  # same stream clock, comm kernel
            (1, 1, 250, 280, 0, 7, 3),  # inside inner2
            (1, 1, 130, 140, 1, 7, 4),  # starts together with kernel 2
            (6, 6, 1500, 1600, 0, 7, 5),  # second step, launched on thread 2
            (1, 1, 3500, 3600, 0, 7, 6),  # launch unknown: any-thread fallback
            (1, 1, 5000, 5100, 0, 7, 7),  # outside every range
            (1, 1, 4100, 4200, 0, 7, 8),  # step on thread 2 only
            (99, 99, 700, 710, 0, 7, 3),  # unnamed, and a repeated id
        ],
    )
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?)",
        [
            (1, 110, 120, 64, 0, 7, 1),
            (2, 400, 410, 128, 0, 7, 3),
            (8, 1010, 1020, 1, 0, 7, 5),
        ],
    )
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?,?,?,?,?,?)",
        [(30, 40, 256, 0, 7, 1), (50, 50, 1, 0, 7, 2)],
    )
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (3, ?, ?, ?, ?)",
        [
            (115, 119, 1, 1001),
            (125, 129, 2, 1001),
            (245, 249, 3, 1001),
            (246, 249, 3, 2002),  # the first row of an id wins
            (128, 129, 4, 1001),
            (1490, 1499, 5, 2002),
            (3400, 3409, 8, 2002),
            (1, 2, None, 1001),
        ],
    )
    c.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (59, ?, ?, ?, ?, ?)",
        [
            (0, 1000, None, 4, 1001),  # step 4, from StringIds
            (1000, 2000, "ProfilerStep#5", None, 1001),
            (10, 900, "outer", None, 1001),
            (20, 100, "inner", None, 1001),
            (110, 200, "inner", None, 1001),
            (240, 300, "inner2 é", None, 1001),
            (100, 150, None, 5, 1001),  # "block", from StringIds
            (0, 5000, "other thread", None, 2002),
            (3000, 4000, "iter_9", None, 2002),
            (3200, 3300, "", None, 2002),  # no text
            (40, 40, "zero length", None, 1001),
        ],
    )


def _tensorrt(c: sqlite3.Connection) -> None:
    """No step names; ExecutionContext ranges delimit the steps."""
    c.executescript(";".join([STRINGS, KERNEL, RUNTIME, NVTX]))
    c.executemany("INSERT INTO StringIds VALUES (?, ?)", STRING_ROWS)
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
        [(1, 1, 120 + 1000 * i, 180 + 1000 * i, 0, 7, i) for i in range(3)],
    )
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (3, ?, ?, ?, 1)",
        [(100 + 1000 * i, 110 + 1000 * i, i) for i in range(3)],
    )
    c.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (59, ?, ?, ?, NULL, 1)",
        [
            (1000 * i, 1000 * i + 900, "ExecutionContext::enqueue")
            for i in range(3)
        ]
        + [(1000 * i + 50, 1000 * i + 500, "layer") for i in range(3)],
    )


def _no_steps(c: sqlite3.Connection) -> None:
    """Kernels and a scope but no step marker: the whole trace is step 0."""
    c.executescript(";".join([STRINGS, KERNEL, RUNTIME, NVTX]))
    c.executemany("INSERT INTO StringIds VALUES (?, ?)", STRING_ROWS)
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
        [(1, 1, 120, 180, 0, 7, 1), (2, 2, 300, 340, 0, 8, 2)],
    )
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (3, ?, ?, ?, 1)",
        [(110, 119, 1), (290, 299, 2)],
    )
    c.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (59, ?, ?, ?, NULL, 1)",
        [(100, 200, "just a scope")],
    )


def _unnamed_ranges(c: sqlite3.Connection) -> None:
    """NVTX ranges exist but none has a text."""
    c.executescript(";".join([STRINGS, KERNEL, NVTX]))
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
        [(1, 1, 120, 180, 0, 7, 1)],
    )
    c.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (59, ?, ?, '', NULL, 1)", [(100, 200)]
    )


def _minimal(c: sqlite3.Connection) -> None:
    """No StringIds and none of the optional columns."""
    c.executescript(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (shortName INTEGER, start "
        "INTEGER, end INTEGER); CREATE TABLE NVTX_EVENTS (text TEXT, start "
        "INTEGER, end INTEGER)"
    )
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)",
        [(5, 1, 2), (6, 3, 4)],
    )
    c.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?,?,?)",
        [("ProfilerStep#1", 0, 10), ("scope", 1, 5)],
    )


def _empty(c: sqlite3.Connection) -> None:
    c.executescript(STRINGS)


def _metrics(c: sqlite3.Connection) -> None:
    """The modern GPU metrics tables, with a zero sample and an unknown id."""
    _labels(c)
    c.executescript(
        "CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER PRIMARY KEY, "
        "metricName TEXT); CREATE TABLE GPU_METRICS (metricId INTEGER, "
        "timestamp INTEGER, value REAL)"
    )
    c.executemany(
        "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?)",
        [(1, "SMs Active"), (2, "Tensor Active")],
    )
    c.executemany(
        "INSERT INTO GPU_METRICS VALUES (?,?,?)",
        [(1, 1000, 12.5), (2, 1000, 0.0), (1, 2000, 99.25), (9, 3000, 1.0)],
    )


def _fixtures(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Writes every export and returns them by name."""
    out = {
        name: _build(tmp_path / f"{name}.sqlite", fill)
        for name, fill in {
            "labels": _labels,
            "tensorrt": _tensorrt,
            "no_steps": _no_steps,
            "unnamed_ranges": _unnamed_ranges,
            "minimal": _minimal,
            "empty": _empty,
            "metrics": _metrics,
        }.items()
    }
    for name, legacy in (
        ("generated", False),
        ("generated_legacy_metrics", True),
    ):
        path = tmp_path / f"{name}.sqlite"
        synth_nsys.write_nsys_sqlite(path, steps=40, legacy_gpu_metrics=legacy)
        out[name] = path
    return out


def _tables(
    path: pathlib.Path, out_dir: pathlib.Path
) -> dict[str, pl.DataFrame]:
    """Ingests one export and reads the tables back."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = models.TraceArtifact(format="nsys", paths=(path,), world_size=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        written = reader.NsysAdapter().ingest(artifact, out_dir)
    # These tables are newer than the Python reader the digests come from.
    return {
        name: pl.read_parquet(p)
        for name, p in written.items()
        if name not in ("network_ops", "runtime_calls")
    }


def table_digest(tables: dict[str, pl.DataFrame]) -> str:
    """Returns a hash of the tables, each sorted by all its columns.

    Rows that tie on the start time have no defined order in the tables, so
    they are put in a fixed one first.
    """
    digest = hashlib.sha256()
    for name in sorted(tables):
        table = tables[name]
        digest.update(name.encode())
        digest.update(table.sort(table.columns).write_csv().encode())
    return digest.hexdigest()


# Hashes of the tables the Python reader produced from each export.
GOLDEN: dict[str, str] = {
    "labels": (
        "86f4db9adaf8f09d94c2f9787f4460990cd7f67d23b6ca8ef6e9a930afe14283"
    ),
    "tensorrt": (
        "b3d1da8de582f3fb62685f524e6bcd5726c246e7c8694fe7f81dbc75a4602f22"
    ),
    "no_steps": (
        "bc0c822f585672744c8b688f0632ed1754e59c77b4439c8c34ab7acf2953aedf"
    ),
    "unnamed_ranges": (
        "7107628bfc90a0422898f279a787b332d72242f27cfd688aedd2118dc4bda2e2"
    ),
    "minimal": (
        "70a54a41bc2bd934e5f1d6ea9dcf4a3af9f47b96033838176c4e322763523959"
    ),
    "empty": (
        "62e1486f4ed28bc639917e76b7c44c5abd6e02b8554d83a31a6ac51a48cf7f4a"
    ),
    "metrics": (
        "9f325ee7eca75ef2019ab1930bec4d1c49d70c824f676100776771792019d7cf"
    ),
    "generated": (
        "cd08651a27ee28d91eaf1cbbac6c36ca22aa2e25cf838fab121609a1e9f541c5"
    ),
    "generated_legacy_metrics": (
        "7a9c2520b68309ddb60d0f4cbc3c1f4bd8baea84a626d2d5fb7f11e5e4880cb0"
    ),
}


def test_tables_match_the_python_reader_they_replaced(
    tmp_path: pathlib.Path,
) -> None:
    actual = {
        name: table_digest(_tables(path, tmp_path / f"out_{name}"))
        for name, path in _fixtures(tmp_path).items()
    }
    assert actual == GOLDEN


def test_activities_come_in_table_order_with_launches_resolved(
    tmp_path: pathlib.Path,
) -> None:
    cols = _native.read_nsys_sqlite(str(_fixtures(tmp_path)["labels"]))
    names = [cols["names"][c] for c in cols["gpu_name"].tolist()]
    # Kernels (those that end after they start), then copies, then sets.
    assert names[:2] == [
        "void gemm<float>(int)",
        "ncclDevKernel_AllReduce_Sum_f16_RING_LL",
    ]
    assert cols["gpu_category"].tolist()[:2] == [0, 1]
    assert cols["names"][cols["gpu_name"][4]] == "NCCL in capitals"
    assert cols["gpu_category"][4] == 1  # any letter case counts as nccl
    assert names.count("Memset") == 1  # the zero-length set is skipped
    assert "Memcpy_D2D" in names
    # The first runtime row of a correlation id is the launch.
    third = 2
    assert cols["gpu_launch_ns"][third] == 245
    assert cols["gpu_launch_thread"][third] == 1001
    unknown = 5
    assert cols["gpu_launch_thread"][unknown] == -1
    assert (
        cols["gpu_launch_ns"][unknown] != cols["gpu_launch_ns"][unknown]
    )  # NaN


def test_ranges_keep_their_text_and_thread(tmp_path: pathlib.Path) -> None:
    cols = _native.read_nsys_sqlite(str(_fixtures(tmp_path)["labels"]))
    texts = [cols["nvtx_texts"][c] for c in cols["nvtx_text"].tolist()]
    assert "ProfilerStep#4" in texts  # text taken from StringIds
    assert "block" in texts
    assert "" in texts  # an unnamed range is kept, and filtered later
    assert "zero length" not in texts  # a range must end after it starts
    assert 2002 in cols["nvtx_thread"].tolist()


def test_steps_scopes_and_the_launching_thread(tmp_path: pathlib.Path) -> None:
    tables = _tables(_fixtures(tmp_path)["labels"], tmp_path / "out")
    kernels = tables["kernels"]
    by_start = {
        (row["start_ns"], row["name"]): row
        for row in kernels.iter_rows(named=True)
    }
    first = by_start[(120, "void gemm<float>(int)")]
    assert first["step"] == 4  # launched at 115 on thread 1, inside step 4
    assert first["scope"] == "inner"
    assert first["scope_path"] == "outer > block > inner"
    third = by_start[(250, "void gemm<float>(int)")]
    assert third["scope"] == "inner2 é"
    outside = by_start[(5000, "void gemm<float>(int)")]
    assert (outside["step"], outside["scope"]) == (-1, "other thread")
    assert sorted(tables["steps"]["step"].to_list()) == [4, 5, 9]


def test_without_step_markers_the_whole_trace_is_step_zero(
    tmp_path: pathlib.Path,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    artifact = models.TraceArtifact(
        format="nsys",
        paths=(_fixtures(tmp_path)["no_steps"],),
        world_size=1,
    )
    with pytest.warns(RuntimeWarning, match="no NVTX step markers"):
        written = reader.NsysAdapter().ingest(artifact, out_dir)
    steps = pl.read_parquet(written["steps"])
    assert steps["step"].to_list() == [0]
    assert (steps["start_ns"][0], steps["end_ns"][0]) == (120, 340)


def test_tensorrt_execution_contexts_are_the_steps(
    tmp_path: pathlib.Path,
) -> None:
    tables = _tables(_fixtures(tmp_path)["tensorrt"], tmp_path / "out")
    assert sorted(tables["steps"]["step"].to_list()) == [0, 1, 2]
    assert sorted(tables["kernels"]["step"].to_list()) == [0, 1, 2]


def test_gpu_metrics_come_from_both_table_forms(tmp_path: pathlib.Path) -> None:
    files = _fixtures(tmp_path)
    modern = _tables(files["metrics"], tmp_path / "m")["gpu_metrics"]
    # The sample with an unknown metric id is not joined; zero is kept.
    assert modern.height == 3
    assert 0.0 in modern["value"].to_list()
    legacy = _tables(files["generated_legacy_metrics"], tmp_path / "l")[
        "gpu_metrics"
    ]
    assert legacy.height == 6  # two samples of three metrics, as text
    assert _native.read_nsys_sqlite(str(files["metrics"]))["has_metrics"]
    assert not _native.read_nsys_sqlite(str(files["generated_legacy_metrics"]))[
        "has_metrics"
    ]


def test_a_missing_file_is_an_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(RuntimeError, match="cannot open"):
        _native.read_nsys_sqlite(str(tmp_path / "none.sqlite"))


def test_a_file_that_is_not_a_database_is_an_error(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "junk.sqlite"
    path.write_bytes(b"this is not a database" * 100)
    with pytest.raises(RuntimeError, match="sqlite"):
        _native.read_nsys_sqlite(str(path))


def test_chains_of_coded_names_equal_chains_of_a_list() -> None:
    """The level-by-level path builder agrees with the plain recursion."""
    rng = random.Random(4)
    for _ in range(25):
        count = rng.randrange(1, 300)
        starts = [rng.randrange(0, 1000) for _ in range(count)]
        ends = [s + rng.randrange(1, 400) for s in starts]
        texts = [f"scope{k}" for k in range(rng.randrange(1, 6))]
        codes = [rng.randrange(len(texts)) for _ in range(count)]
        threads = {}
        for thread in range(rng.randrange(1, 4)):
            members = [i for i in range(count) if i % 3 == thread % 3]
            if members:
                threads[thread] = intervals.build_thread_ranges(
                    [(starts[i], ends[i], i) for i in members]
                )
        wanted = np.array(
            [rng.randrange(-1, count) for _ in range(500)], dtype=np.int64
        )
        coded = intervals.range_paths(
            threads, wanted, intervals.CodedNames(texts, np.array(codes))
        )
        plain = intervals.range_paths(
            threads, wanted, [texts[c] for c in codes]
        )
        assert coded.tolist() == plain.tolist()


def test_runtime_calls_are_listed_and_add_up_to_the_summary(
    tmp_path: pathlib.Path,
) -> None:
    cols = _native.read_nsys_sqlite(str(_fixtures(tmp_path)["labels"]))
    calls = cols["runtime_calls"]
    assert len(calls["name"]) > 0
    names = [calls["names"][code] for code in calls["name"].tolist()]
    duration = calls["end"] - calls["start"]
    assert (duration > 0).all()
    totals: dict[str, list[int]] = {}
    for name, ns in zip(names, duration.tolist(), strict=True):
        entry = totals.setdefault(name, [0, 0])
        entry[0] += ns
        entry[1] += 1
    summary = {
        cols["runtime"]["names"][i]: [
            int(cols["runtime"]["duration_ns"][i]),
            int(cols["runtime"]["calls"][i]),
        ]
        for i in range(len(cols["runtime"]["names"]))
    }
    assert totals == summary


def test_runtime_calls_table_has_steps_and_threads(
    tmp_path: pathlib.Path,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    artifact = models.TraceArtifact(
        format="nsys", paths=(_fixtures(tmp_path)["labels"],), world_size=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        written = reader.NsysAdapter().ingest(artifact, out_dir)
    table = pl.read_parquet(written["runtime_calls"])
    assert table.schema == runtime_calls.RUNTIME_CALLS_SCHEMA
    assert table["start_ns"].is_sorted()
    assert (table["duration_ns"] == table["end_ns"] - table["start_ns"]).all()
    # The call at 115 on thread 1 launches a kernel and sits inside step 4.
    launch = table.filter(pl.col("start_ns") == 115)
    assert launch["thread"].to_list() == [1001]
    assert launch["step"].to_list() == [4]
