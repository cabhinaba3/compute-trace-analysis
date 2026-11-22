"""Tests of the C++20 JAX/XProf reader (``cta._native.read_jax``).

Covers Chrome-trace JSON (.json, .json.gz, bare array), XPlane protobuf
(.xplane.pb, .xplane.pb.gz) and binary Perfetto protobuf (.pftrace,
.pftrace.gz, TrackEvent and ChromeEventBundle), including block/chunk/thread
invariance down to 1-byte reads, adversarial edge cases, cross-format identity,
and golden hashes captured from the original Python reader on real two-GPU H200
profiles.
"""

from __future__ import annotations

import glob
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
from cta.adapters.jax import reader as jax_reader
from cta.synth import jax as synth_jax

REAL_JAX_DIR = pathlib.Path(__file__).resolve().parents[2] / "data_real" / "f"

_BASES_US = {"small": 1_000_000.0, "epoch": 1_700_000_000_000_000.0}

# Golden digests of (kernels, steps, runtime) captured from the original pure
# Python JAX adapter on the real two-GPU H200 fixtures.
REAL_GOLDEN: dict[str, str] = {
    "annotated": (
        "6b07e56205cf075e6ae854c972f969ae70a2078e1fc9c7e524426a5510c89e69"
    ),
    "plain": (
        "a2495e28f4757646505a87c98c1e8183bd469d31a61740673e8831e11d3b33a8"
    ),
}


def _same_columns(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        if key == "timing":
            continue
        if isinstance(value, np.ndarray):
            if not np.array_equal(value, b[key], equal_nan=True):
                return False
        elif value != b[key]:
            return False
    return True


def _ingest_tables(
    path: pathlib.Path, out_dir: pathlib.Path
) -> dict[str, pl.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = models.TraceArtifact(format="jax", paths=(path,), world_size=1)
    written = jax_reader.JaxAdapter().ingest(artifact, out_dir)
    return {name: pl.read_parquet(p) for name, p in written.items()}


def _digest(
    tables: dict[str, pl.DataFrame],
    names: tuple[str, ...] = ("kernels", "steps", "runtime"),
) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update(tables[name].write_csv().encode())
    return h.hexdigest()


def _adversarial_events(base_us: float) -> list[Any]:
    """Builds events exercising escaping, late metadata, rounding and args."""
    t = base_us
    return [
        # Device events come BEFORE process_name / thread_name metadata.
        {
            "ph": "X",
            "pid": 10,
            "tid": 7,
            "name": 'void gemm<{lambda}#1>("quoted" \\path)',
            "ts": t + 60.0625,
            "dur": 20.1875,
            "args": {
                "correlation_id": "100",
                "hlo_op": "dot.1",
                "module": "fallback_mod",
                "hlo_module": "jit_train",
                "name": "jit(train)/jvp()/dot_general",
            },
        },
        # Derived XLA Ops duplicate on tid=8 (dropped once late metadata loads).
        {
            "ph": "X",
            "pid": 10,
            "tid": 8,
            "name": "dot.1",
            "ts": t + 60.0625,
            "dur": 20.1875,
            "args": {"hlo_op": "dot.1", "hlo_module": "jit_train"},
        },
        # Memcpy with both memcpy_details and Memset_details (memcpy wins).
        {
            "ph": "X",
            "pid": 10,
            "tid": 7,
            "name": "MemcpyP2P é日本",
            "ts": t + 90.0,
            "dur": 5.0,
            "args": {
                "correlation_id": 101,
                "memcpy_details": "kind:dev size:4096 dest:1",
                "Memset_details": "kind:dev num_bytes:128",
                "module": "pjit_step",
            },
        },
        # Collective op (send-0) on negative tid to test Euclidean stream mod.
        {
            "ph": "X",
            "pid": 11,
            "tid": -5,
            "name": "send-0",
            "ts": t + 100.0,
            "dur": 12.0,
            "args": {"hlo_op": "send-0", "hlo_module": "jit_train"},
        },
        # XLA:CPU thunk on host with explicit device_ordinal.
        {
            "ph": "X",
            "pid": "host",
            "tid": "cpu-0",
            "name": "cpu_fusion.1",
            "ts": t + 115.0,
            "dur": 8.0,
            "args": {
                "hlo_op": "fusion.1",
                "hlo_module": "pmap_eval",
                "device_ordinal": "2",
            },
        },
        # Ignored: zero/negative/missing dur and non-X events.
        {"ph": "X", "pid": 10, "tid": 7, "name": "zero", "ts": t, "dur": 0.0},
        {"ph": "X", "pid": 10, "tid": 7, "name": "neg", "ts": t, "dur": -1.0},
        {"ph": "X", "pid": 10, "tid": 7, "name": "nodur", "ts": t},
        {"ph": "i", "pid": 10, "tid": 7, "name": "inst", "ts": t},
        # Step annotation (negative step_num allowed).
        {
            "ph": "X",
            "pid": "host",
            "tid": "main",
            "name": "train",
            "ts": t,
            "dur": 500.0,
            "args": {"step_num": "-1"},
        },
        # Repeated correlation_id=100 launches: FIRST host launch wins.
        {
            "ph": "X",
            "pid": "host",
            "tid": "main",
            "name": "cuLaunchKernel",
            "ts": t + 10.0,
            "dur": 4.0,
            "args": {"correlation_id": "100"},
        },
        {
            "ph": "X",
            "pid": "host",
            "tid": "main",
            "name": "cuLaunchKernel",
            "ts": t + 25.0,
            "dur": 6.0,
            "args": {"correlation_id": "100"},
        },
        {
            "ph": "X",
            "pid": "host",
            "tid": "main",
            "name": "MemcpyP2P",
            "ts": t + 30.0,
            "dur": 3.0,
            "args": {"correlation_id": 101},
        },
        # PJRT Execute span.
        {
            "ph": "X",
            "pid": "host",
            "tid": "main",
            "name": "CommonPjRtLoadedExecutable::Execute (jit_train)",
            "ts": t + 1.0,
            "dur": 15.0,
            "args": {"name": "jit_train"},
        },
        # Non-object array entries (ignored).
        None,
        42,
        "text",
        [1, 2],
        # Late metadata at the very end of the file!
        {
            "ph": "M",
            "pid": "host",
            "name": "process_name",
            "args": {"name": "/host:CPU"},
        },
        {
            "ph": "M",
            "pid": 10,
            "name": "process_name",
            "args": {"name": "/device:GPU:0"},
        },
        {
            "ph": "M",
            "pid": 10,
            "tid": 7,
            "name": "thread_name",
            "args": {"name": "Stream #7(Compute)"},
        },
        {
            "ph": "M",
            "pid": 10,
            "tid": 8,
            "name": "thread_name",
            "args": {"name": "XLA Ops"},
        },
        {
            "ph": "M",
            "pid": 11,
            "name": "process_name",
            "args": {"name": "/device:TPU:1"},
        },
        {
            "ph": "M",
            "pid": 11,
            "tid": -5,
            "name": "thread_name",
            "args": {"name": "XLA Ops"},
        },
    ]


def _write_adversarial(
    path: pathlib.Path,
    *,
    base_us: float,
    layout: str = "object",
    ensure_ascii: bool = True,
) -> pathlib.Path:
    events = _adversarial_events(base_us)
    payload: Any = (
        events
        if layout == "list"
        else {"displayTimeUnit": "ns", "traceEvents": events}
    )
    text = json.dumps(payload, indent=1, ensure_ascii=ensure_ascii)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_json_result_is_independent_of_threads_chunk_and_block_size(
    tmp_path: pathlib.Path,
) -> None:
    fixtures = [
        _write_adversarial(tmp_path / "small.json", base_us=_BASES_US["small"]),
        _write_adversarial(tmp_path / "epoch.json", base_us=_BASES_US["epoch"]),
        _write_adversarial(
            tmp_path / "list.json",
            base_us=_BASES_US["small"],
            layout="list",
        ),
        _write_adversarial(
            tmp_path / "gz.trace.json.gz",
            base_us=_BASES_US["small"],
            ensure_ascii=False,
        ),
        synth_jax.write_trace(tmp_path / "synth.trace.json.gz", steps=4),
    ]
    large = tmp_path / "large.json"
    synth_jax.write_large_trace(large, target_bytes=2 << 20, seed=3)
    fixtures.append(large)

    for path in fixtures:
        whole = _native.read_jax(str(path), "json", 1, 1 << 40, 1 << 30)
        tiny = 1 if path.stat().st_size < 100_000 else 13
        for threads, chunk, block in (
            (1, 64, tiny),
            (3, 100, 7),
            (8, 257, 64),
            (4, 5000, 4099),
        ):
            split = _native.read_jax(str(path), "json", threads, chunk, block)
            assert _same_columns(whole, split), (
                path.name,
                threads,
                chunk,
                block,
            )


def test_adversarial_semantics_match_python_rules(
    tmp_path: pathlib.Path,
) -> None:
    path = _write_adversarial(
        tmp_path / "adv.json", base_us=0.0, ensure_ascii=False
    )
    cols = _native.read_jax(str(path), "json", 2, 128, 7)
    assert cols["format"] == "json"
    names = cols["names"]
    op_names = [names[i] for i in cols["op_name"].tolist()]
    # Derived tid=8 ("dot.1") dropped because tid=7 ("Stream #7") won.
    assert "dot.1" not in op_names
    assert len(op_names) == 4

    gemm_idx = next(i for i, n in enumerate(op_names) if "gemm" in n)
    # 60.0625 us -> 60062 ns (round half to even); 20.1875 us -> 20188 ns.
    assert cols["op_start"][gemm_idx] == 60_062
    assert cols["op_end"][gemm_idx] == 60_062 + 20_188
    # First host launch (10 us = 10_000 ns) wins over second (25 us).
    assert cols["op_launch_ns"][gemm_idx] == 10_000.0
    assert names[cols["op_hlo_module"][gemm_idx]] == "jit_train"
    assert names[cols["op_scope"][gemm_idx]] == "dot_general"
    assert (
        names[cols["op_scope_path"][gemm_idx]]
        == "jit(train) > jvp() > dot_general"
    )
    assert cols["op_category"][gemm_idx] == 0

    p2p_idx = next(i for i, n in enumerate(op_names) if "MemcpyP2P" in n)
    assert cols["op_bytes"][p2p_idx] == 4096
    assert cols["op_category"][p2p_idx] == 2
    assert names[cols["op_scope"][p2p_idx]] == "pjit(step)"

    send_idx = op_names.index("send-0")
    assert cols["op_category"][send_idx] == 1
    assert cols["op_device"][send_idx] == 1
    assert cols["op_stream"][send_idx] == (-5) % (1 << 31)
    assert np.isnan(cols["op_launch_ns"][send_idx])

    cpu_idx = op_names.index("cpu_fusion.1")
    assert cols["op_device"][cpu_idx] == 2
    assert names[cols["op_scope"][cpu_idx]] == "pmap(eval)"

    assert cols["step_id"].tolist() == [-1]
    assert [names[i] for i in cols["exec_module"].tolist()] == ["jit_train"]
    assert len(cols["rcall_name"]) == 3


def test_cross_format_equivalence_json_xplane_and_pftrace(
    tmp_path: pathlib.Path,
) -> None:
    files = {
        "json": synth_jax.write_trace(tmp_path / "t.trace.json.gz", steps=3),
        "json_plain": synth_jax.write_trace(tmp_path / "t.json", steps=3),
        "xplane": synth_jax.write_xplane(tmp_path / "t.xplane.pb", steps=3),
        "xplane_gz": synth_jax.write_xplane(
            tmp_path / "t.xplane.pb.gz", steps=3
        ),
        "pftrace": synth_jax.write_pftrace(tmp_path / "t.pftrace", steps=3),
        "pftrace_gz": synth_jax.write_pftrace(
            tmp_path / "t.pftrace.gz", steps=3
        ),
        "pftrace_chrome": synth_jax.write_pftrace(
            tmp_path / "t_chrome.pftrace", steps=3, legacy_chrome=True
        ),
    }
    all_tables = (
        "kernels",
        "steps",
        "runtime",
        "runtime_calls",
        "network_ops",
    )
    digests = {
        key: _digest(_ingest_tables(path, tmp_path / f"out_{key}"), all_tables)
        for key, path in files.items()
    }
    assert len(set(digests.values())) == 1, digests

    # Also check thread-count invariance on XPlane and auto-detection without
    # format-specific extensions.
    xp_one = _native.read_jax(str(files["xplane"]), "xplane", 1)
    xp_many = _native.read_jax(str(files["xplane"]), "xplane", 4)
    assert _same_columns(xp_one, xp_many)

    for expected_fmt, src in (
        ("json", files["json_plain"]),
        ("xplane", files["xplane"]),
        ("pftrace", files["pftrace"]),
    ):
        renamed = tmp_path / f"opaque_{expected_fmt}.dat"
        renamed.write_bytes(src.read_bytes())
        cols = _native.read_jax(str(renamed), "auto")
        assert cols["format"] == expected_fmt
        assert len(cols["op_name"]) == len(xp_one["op_name"])


def test_cross_format_unannotated_module_step_inference(
    tmp_path: pathlib.Path,
) -> None:
    files = {
        "json": synth_jax.write_trace(
            tmp_path / "u.trace.json.gz", steps=2, annotated=False
        ),
        "xplane": synth_jax.write_xplane(
            tmp_path / "u.xplane.pb", steps=2, annotated=False
        ),
        "pftrace": synth_jax.write_pftrace(
            tmp_path / "u.pftrace", steps=2, annotated=False
        ),
    }
    digests: dict[str, str] = {}
    for key, path in files.items():
        with pytest.warns(RuntimeWarning, match="StepTraceAnnotation"):
            tables = _ingest_tables(path, tmp_path / f"out_u_{key}")
        assert tables["steps"]["step"].to_list() == [0, 1]
        digests[key] = _digest(
            tables, ("kernels", "steps", "runtime", "network_ops")
        )
    assert len(set(digests.values())) == 1


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
        _native.read_jax(str(path), "json", 2, 100)


def test_bad_event_and_bad_protobuf_and_args_are_rejected(
    tmp_path: pathlib.Path,
) -> None:
    bad_ts = tmp_path / "bad_ts.json"
    bad_ts.write_text(
        json.dumps(
            {
                "displayTimeUnit": "ns",
                "traceEvents": [
                    {"ph": "X", "pid": 1, "tid": 1, "name": "a", "ts": None}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="numbers"):
        _native.read_jax(str(bad_ts), "json", 2, 100)

    bad_pb = tmp_path / "bad.xplane.pb"
    bad_pb.write_bytes(b"\x00\xff\xff")
    with pytest.raises(ValueError, match="protobuf"):
        _native.read_jax(str(bad_pb), "xplane")

    bad_pf = tmp_path / "bad.pftrace"
    bad_pf.write_bytes(b"\x0a\x7ftruncated")
    with pytest.raises(ValueError, match="protobuf"):
        _native.read_jax(str(bad_pf), "pftrace")

    with pytest.raises(ValueError, match="unknown JAX format"):
        _native.read_jax(str(bad_ts), "bogus")

    with pytest.raises(RuntimeError, match="cannot open"):
        _native.read_jax(str(tmp_path / "missing.json"))


def _varint(value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _vi(field: int, value: int) -> bytes:
    return _varint((field << 3) | 0) + _varint(value)


def test_xplane_and_pftrace_proto_edge_cases(tmp_path: pathlib.Path) -> None:
    # XPlane: event_metadata.display_name + metadata.stats + num_occurrences
    # skipped + picosecond half-to-even rounding (62500 ps -> 62 ns).
    stat_meta_1 = _vi(1, 1) + _ld(2, b"hlo_op")
    stat_meta_2 = _vi(1, 2) + _ld(2, b"all-gather.1")
    meta_stat = _vi(1, 1) + _vi(7, 2)  # hlo_op -> ref_value 2
    ev_meta_1 = (
        _vi(1, 1)
        + _ld(2, b"raw_name")
        + _ld(3, b"display_allgather")
        + _ld(5, meta_stat)
    )
    ev_timed = _vi(1, 1) + _vi(2, 62_500) + _vi(3, 187_500)
    ev_occ = _vi(1, 1) + _vi(5, 99) + _vi(3, 1_000_000)  # num_occurrences
    line = (
        _vi(1, 7)
        + _ld(2, b"Stream #7")
        + _vi(3, 1000)
        + _ld(4, ev_timed)
        + _ld(4, ev_occ)
    )
    plane = (
        _vi(1, 0)
        + _ld(2, b"/device:GPU:0")
        + _ld(3, line)
        + _ld(4, _vi(1, 1) + _ld(2, ev_meta_1))
        + _ld(5, _vi(1, 1) + _ld(2, stat_meta_1))
        + _ld(5, _vi(1, 2) + _ld(2, stat_meta_2))
    )
    xplane_path = tmp_path / "edge.xplane.pb"
    xplane_path.write_bytes(_ld(1, plane))
    xcols = _native.read_jax(str(xplane_path), "xplane")
    names = xcols["names"]
    assert [names[i] for i in xcols["op_name"].tolist()] == [
        "display_allgather"
    ]
    assert xcols["op_start"].tolist() == [1062]
    assert xcols["op_end"].tolist() == [1062 + 188]
    assert xcols["op_category"].tolist() == [1]

    # Perfetto: TrackPacketDefaults + timestamp_delta_us + nested slices +
    # unclosed slice dropped + legacy ChromeEventBundle string_table (1-based).
    proc_td = (
        _vi(1, 100)
        + _ld(2, b"/device:TPU:3")
        + _ld(3, _vi(1, 10) + _ld(6, b"/device:TPU:3"))
    )
    line_td = (
        _vi(1, 101)
        + _vi(5, 100)
        + _ld(2, b"XLA Ops")
        + _ld(4, _vi(1, 10) + _vi(2, 5) + _ld(5, b"XLA Ops"))
    )
    defaults = _ld(11, _vi(11, 101))  # default_track_uuid = 101
    pkt_p = _vi(10, 7) + _ld(60, proc_td)
    pkt_t = _vi(10, 7) + _ld(60, line_td)
    # Outer slice [1000us, 1050us], inner slice [1010us, 1020us], unclosed slice
    # started at 1060us.
    ann = _ld(10, b"hlo_op") + _ld(6, b"reduce-scatter.0")
    pkt_b1 = (
        _vi(10, 7)
        + _vi(13, 1)  # SEQ_INCREMENTAL_STATE_CLEARED
        + _ld(59, defaults)
        + _vi(8, 1_000_000)
        + _ld(11, _vi(9, 1) + _ld(23, b"outer_rs") + _ld(4, ann))
    )
    pkt_b2 = (
        _vi(10, 7)
        + _ld(11, _vi(9, 1) + _vi(1, 10) + _ld(23, b"inner_op"))  # +10 us
    )
    pkt_e2 = _vi(10, 7) + _ld(11, _vi(9, 2) + _vi(1, 10))  # +10 us -> 1020 us
    pkt_e1 = _vi(10, 7) + _ld(11, _vi(9, 2) + _vi(1, 30))  # +30 us -> 1050 us
    pkt_unclosed = _vi(10, 7) + _ld(
        11, _vi(9, 1) + _vi(1, 10) + _ld(23, b"unclosed_dropped")
    )
    # Legacy ChromeEventBundle with 1-based string_table names.
    ce_meta = (
        _vi(3, ord("M"))
        + _vi(11, 99)
        + _vi(15, 1)  # "process_name"
        + _ld(14, _vi(9, 2) + _ld(7, b"/device:GPU:4"))
    )
    ce_ev = (
        _vi(3, ord("X"))
        + _vi(11, 99)
        + _vi(4, 9)
        + _vi(15, 3)  # "chrome_kernel"
        + _vi(2, 2000)
        + _vi(5, 25)
    )
    bundle = (
        _ld(1, ce_meta)
        + _ld(1, ce_ev)
        + _ld(2, b"process_name")
        + _ld(2, b"name")
        + _ld(2, b"chrome_kernel")
    )
    pkt_chrome = _ld(5, bundle)

    pf_path = tmp_path / "edge.pftrace"
    pf_path.write_bytes(
        b"".join(
            _ld(1, p)
            for p in (
                pkt_p,
                pkt_t,
                pkt_b1,
                pkt_b2,
                pkt_e2,
                pkt_e1,
                pkt_unclosed,
                pkt_chrome,
            )
        )
    )
    pcols = _native.read_jax(str(pf_path), "pftrace")
    pnames = [pcols["names"][i] for i in pcols["op_name"].tolist()]
    assert pnames == ["inner_op", "outer_rs", "chrome_kernel"]
    assert pcols["op_start"].tolist() == [1_010_000, 1_000_000, 2_000_000]
    assert pcols["op_end"].tolist() == [1_020_000, 1_050_000, 2_025_000]
    assert pcols["op_device"].tolist() == [3, 3, 4]
    assert pcols["op_stream"].tolist() == [5, 5, 9]
    assert pcols["op_category"].tolist() == [0, 1, 0]


@pytest.mark.skipif(
    not REAL_JAX_DIR.exists(),
    reason="real JAX fixtures (tests/data_real/f) not generated",
)
def test_real_h200_fixtures_match_python_golden_across_json_and_xplane(
    tmp_path: pathlib.Path,
) -> None:
    for mode, expected_digest in REAL_GOLDEN.items():
        base = REAL_JAX_DIR / mode / "plugins" / "profile"
        trace_js = pathlib.Path(
            glob.glob(str(base / "*" / "*.trace.json.gz"))[0]
        )
        perfetto_js = pathlib.Path(
            glob.glob(str(base / "*" / "perfetto_trace.json.gz"))[0]
        )
        xplane_pb = pathlib.Path(glob.glob(str(base / "*" / "*.xplane.pb"))[0])
        for tag, path in (
            ("json", trace_js),
            ("perfetto", perfetto_js),
            ("xplane", xplane_pb),
        ):
            if mode == "plain":
                with pytest.warns(RuntimeWarning, match="StepTraceAnnotation"):
                    tables = _ingest_tables(path, tmp_path / f"{mode}_{tag}")
            else:
                tables = _ingest_tables(path, tmp_path / f"{mode}_{tag}")
            assert _digest(tables) == expected_digest, (mode, tag)
