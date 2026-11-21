"""Regressions for the state handling of the native readers and solvers.

Each test pins one transition that used to go wrong: a state that was left
without being resolved, state kept from an earlier run, or a boundary (a chunk
edge, the end of a pipe) where the parser lost track of where it was.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import pathlib
import threading
from typing import Any

import pytest

from cta import _native as native
from cta.analysis import streaming
from cta.synth import chakra_proto
from cta.synth import goal as synth_goal

EVENT = {
    "ph": "X",
    "cat": "kernel",
    "name": "k",
    "ts": 1,
    "dur": 5,
    "pid": 0,
    "tid": 1,
}


def _et(path: pathlib.Path, nodes: list[object]) -> pathlib.Path:
    buf = bytearray()
    for msg in (chakra_proto.GlobalMetadataMsg(version="0.0.4"), *nodes):
        payload = msg.SerializeToString()  # type: ignore[attr-defined]
        buf += chakra_proto.encode_varint(len(payload)) + payload
    path.write_bytes(bytes(buf))
    return path


def _node(
    node_id: int, name: str, kind: int, *deps: int, us: int = 1, **attrs: int
) -> object:
    return chakra_proto.NodeMsg(
        id=node_id,
        name=name,
        type=kind,
        duration_micros=us,
        data_deps=list(deps),
        attr=[
            chakra_proto.AttributeProtoMsg(name=k, int32_val=v)
            for k, v in attrs.items()
        ],
    )


# --- GOAL: pipes and truncated input ----------------------------------------


def _feed(path: pathlib.Path, payload: bytes) -> threading.Thread:
    def run() -> None:
        try:
            with open(path, "wb", buffering=0) as writer:
                for i in range(0, len(payload), 100):
                    writer.write(payload[i : i + 100])
        except BrokenPipeError:
            pass

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def test_read_goal_reads_a_fifo_whole(tmp_path: pathlib.Path) -> None:
    """A pipe reports size 0; it used to read as an empty file."""
    text = (
        "num_ranks 1\nrank 0 {\n"
        + "".join(f"l{i}: calc {i + 1}\n" for i in range(500))
        + "}\n"
    )
    plain = tmp_path / "a.goal"
    plain.write_text(text)
    fifo = tmp_path / "p.goal"
    os.mkfifo(fifo)
    feeder = _feed(fifo, text.encode())

    got = native.read_goal(str(fifo), threads=1, chunk_bytes=1 << 20)
    feeder.join(timeout=10)

    expected = native.read_goal(str(plain), threads=1, chunk_bytes=1 << 20)
    assert len(got["node_id"]) == len(expected["node_id"]) == 500
    assert list(got["node_id"]) == list(expected["node_id"])


@pytest.mark.parametrize("keep", [0, 5, 13])
def test_a_piped_goal_bin_with_a_cut_header_is_an_error(
    tmp_path: pathlib.Path, keep: int
) -> None:
    """A cut header used to read as an empty graph; a mapped file refuses it."""
    full = tmp_path / "a.bin"
    native.write_goal_allreduce_bin(str(full), world_size=4, steps=2)
    payload = full.read_bytes()[:keep]
    fifo = tmp_path / "p.bin"
    os.mkfifo(fifo)
    feeder = _feed(fifo, payload)

    with pytest.raises(ValueError, match="GOAL binary"):
        list(native.open_goal_bin_stream(str(fifo), chunk_nodes=7))
    feeder.join(timeout=10)

    short = tmp_path / "short.bin"
    short.write_bytes(payload)
    with pytest.raises((ValueError, RuntimeError)):
        list(native.open_goal_bin_stream(str(short), chunk_nodes=7))


# --- Kineto: punctuation of the events array, gzip ---------------------------


def _kineto(text: str, tmp_path: pathlib.Path, chunk: int) -> int:
    path = tmp_path / "t.json"
    path.write_text(text)
    result = native.read_kineto(
        str(path), threads=2, chunk_bytes=chunk, block_bytes=5
    )
    return len(result["gpu_start"])


@pytest.mark.parametrize("chunk", [64, 1 << 20])
@pytest.mark.parametrize(
    "body",
    [
        "[EV,]",
        "[EV,EV,]",
        "[,EV]",
        "[EV,,EV]",
        "[EV EV]",
    ],
)
def test_a_bad_comma_in_the_events_is_refused_at_any_chunk_size(
    tmp_path: pathlib.Path, body: str, chunk: int
) -> None:
    """A comma at a chunk cut is dropped, so the bad ones used to pass there."""
    text = '{"traceEvents":' + body.replace("EV", json.dumps(EVENT)) + "}"

    with pytest.raises(ValueError):
        _kineto(text, tmp_path, chunk)


@pytest.mark.parametrize("chunk", [64, 1 << 20])
@pytest.mark.parametrize(
    "template",
    [
        '"abc"{"traceEvents":[EV]}',
        '{"traceEvents":[EV]},',
        '{"traceEvents":[EV]}:',
        '{"traceEvents":[EV]}"abc"',
        '{"traceEvents":[EV],"traceEvents":[EV]}',
    ],
)
def test_data_outside_the_trace_object_is_refused(
    tmp_path: pathlib.Path, template: str, chunk: int
) -> None:
    with pytest.raises(ValueError):
        _kineto(template.replace("EV", json.dumps(EVENT)), tmp_path, chunk)


@pytest.mark.parametrize("chunk", [64, 200, 1 << 20])
def test_a_good_trace_is_read_alike_at_every_chunk_size(
    tmp_path: pathlib.Path, chunk: int
) -> None:
    events = [dict(EVENT, name=f"k{i}", ts=i * 10) for i in range(40)]
    text = json.dumps({"traceEvents": events, "other": [1, 2, 3]})

    assert _kineto(text, tmp_path, chunk) == 40
    assert _kineto(json.dumps(events), tmp_path, chunk) == 40
    assert _kineto('{"traceEvents": [ ]}', tmp_path, chunk) == 0


def _gzip_trace() -> bytes:
    events = [dict(EVENT, name=f"k{i}", ts=i * 10) for i in range(2000)]
    return gzip.compress(json.dumps({"traceEvents": events}).encode())


def _read_gz(data: bytes, tmp_path: pathlib.Path) -> int:
    path = tmp_path / "t.json.gz"
    path.write_bytes(data)
    out = native.read_kineto(
        str(path), threads=1, chunk_bytes=4096, block_bytes=1000
    )
    return len(out["gpu_start"])


def test_a_whole_gzip_file_is_read(tmp_path: pathlib.Path) -> None:
    assert _read_gz(_gzip_trace(), tmp_path) == 2000


@pytest.mark.parametrize("cut", ["mid", "trailer", "crc"])
def test_a_gzip_file_cut_short_is_an_error(
    tmp_path: pathlib.Path, cut: str
) -> None:
    packed = _gzip_trace()
    data = {
        "mid": packed[: len(packed) // 2],
        "trailer": packed[:-8],
        "crc": packed[:-4],
    }[cut]

    with pytest.raises((ValueError, RuntimeError)):
        _read_gz(data, tmp_path)


# --- Frontier: what is left at the end of the trace --------------------------


def test_a_send_parked_behind_a_partial_barrier_still_matches_its_recv(
    tmp_path: pathlib.Path,
) -> None:
    """At EOF a recv waits on a send parked behind a barrier that never filled.

    The barrier is flushed first, which commits the send and matches the recv.
    The recv used to be counted as unmatched before that, which also changed
    the makespan.
    """
    comp, coll, send, recv = (
        chakra_proto.COMP_NODE,
        chakra_proto.COMM_COLL_NODE,
        chakra_proto.COMM_SEND_NODE,
        chakra_proto.COMM_RECV_NODE,
    )
    rank0 = [
        _node(0, "c0", comp, us=1000),
        _node(1, "ar", coll, 0, us=50, comm_type=0),
        _node(
            2, "send", send, 1, us=10, comm_dst=1, comm_tag=0, comm_size=1024
        ),
    ]
    rank1 = [
        _node(0, "recv", recv, us=10, comm_src=0, comm_tag=0, comm_size=1024),
        _node(1, "c1", comp, 0, us=5000),
    ]
    paths = [
        str(_et(tmp_path / "r0.et", rank0)),
        str(_et(tmp_path / "r1.et", rank1)),
    ]

    none = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier(paths, [0, 1], collective_sync="none")
    )
    barrier = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier(
            paths, [0, 1], collective_sync="barrier"
        )
    )

    # The grid engine skips a barrier group left with one participant; so does
    # the frontier solver.
    streams = [native.open_chakra_et_stream(paths, [0, 1]) for _ in range(2)]
    grid = {
        sync: native.solve_dag_stream(
            stream,
            engine="gridgraph",
            collective_sync=sync,
            world_size_hint=2,
            scratch_dir=str(tmp_path / f"s_{sync}"),
        )
        for sync, stream in zip(("none", "barrier"), streams, strict=True)
    }
    assert none.provenance.unmatched_p2p_recvs == 0
    assert barrier.provenance.unmatched_p2p_recvs == 0
    assert barrier.provenance.unmatched_p2p_sends == 0
    assert barrier.critical_path_ns == pytest.approx(
        none.critical_path_ns, rel=1e-9
    )
    assert none.num_edges == grid["none"]["num_edges"]
    assert barrier.num_edges == grid["barrier"]["num_edges"]
    assert barrier.critical_path_ns == pytest.approx(
        grid["barrier"]["makespan"], rel=1e-9
    )


# Four ranks, 30 ops, some sends with no recv. With a lookahead of 3 the
# frontier solver drops nodes whose parents have not arrived, which used to
# leave makespan_ub below the exact makespan.
LOOKAHEAD_GOAL = (
    "\n".join(
        [
            "num_ranks 4",
            "rank 0 {",
            "l1: send 8b to 1 tag 1",
            "l2: send 1024b to 2 tag 0",
            "l3: send 65536b to 3 tag 0",
            "l4: calc 767",
            "l5: recv 1024b from 2 tag 0",
            "l2 irequires l1",
            "l2 requires l1",
            "l2 requires l1",
            "l3 requires l1",
            "l4 requires l2",
            "l5 requires l3",
            "}",
            "rank 1 {",
            "l1: calc 385",
            "l2: send 1024b to 3 tag 0",
            "l3: calc 666",
            "l4: send 65536b to 3 tag 1",
            "l5: calc 124",
            "l6: calc 931",
            "l7: recv 1024b from 2 tag 1",
            "l8: calc 398",
            "l2 irequires l1",
            "l4 requires l1",
            "l4 requires l3",
            "l4 requires l2",
            "l5 requires l4",
            "l5 irequires l1",
            "l5 requires l2",
            "l6 requires l2",
            "l6 requires l4",
            "l6 irequires l2",
            "l7 requires l6",
            "l7 requires l6",
            "l8 requires l4",
            "}",
            "rank 2 {",
            "l1: recv 1024b from 3 tag 0",
            "l2: calc 386",
            "l3: calc 999",
            "l4: send 8b to 1 tag 0",
            "l5: send 1024b to 1 tag 1",
            "l6: send 1024b to 0 tag 0",
            "l3 requires l1",
            "l3 irequires l1",
            "l4 requires l2",
            "l5 requires l3",
            "l5 requires l3",
            "l5 requires l3",
            "l6 requires l2",
            "l6 requires l5",
            "}",
            "rank 3 {",
            "l1: recv 1024b from 1 tag 0",
            "l2: calc 533",
            "l3: calc 396",
            "l4: calc 347",
            "l5: recv 65536b from 1 tag 1",
            "l6: send 1024b to 2 tag 0",
            "l7: calc 143",
            "l8: send 65536b to 2 tag 0",
            "l9: send 1024b to 1 tag 1",
            "l10: recv 65536b from 0 tag 0",
            "l11: send 8b to 2 tag 1",
            "l2 requires l1",
            "l2 irequires l1",
            "l2 requires l1",
            "l3 requires l2",
            "l3 irequires l2",
            "l5 requires l1",
            "l5 requires l3",
            "l6 irequires l5",
            "l6 requires l2",
            "l6 requires l3",
            "l7 requires l2",
            "l8 requires l4",
            "l8 requires l1",
            "l8 requires l3",
            "l9 requires l8",
            "l10 requires l6",
            "l11 requires l6",
            "}",
        ]
    )
    + "\n"
)


def test_the_makespan_bound_still_holds_when_the_lookahead_cap_drops_nodes(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "g.goal"
    path.write_text(LOOKAHEAD_GOAL)

    def solve(engine: str, chunk: int, **options: Any) -> dict[str, Any]:
        stream = native.open_goal_text_stream(str(path), chunk_nodes=chunk)
        return native.solve_dag_stream(
            stream, engine=engine, scratch_dir=str(tmp_path / "s"), **options
        )

    exact = solve("gridgraph", 64)["makespan"]
    raw = solve("frontier", 5, max_lookahead=3, window_size=1_000_000)

    prov = raw["provenance"]
    assert prov["lookahead_drops"] > 0
    assert prov["exact"] is False
    assert prov["makespan_lb"] <= exact + 1e-6 <= prov["makespan_ub"] + 2e-6


# --- Out-of-core engines ------------------------------------------------


def _goal_text_stream(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    path = tmp_path / "g.goal"
    synth_goal.write_allreduce(path)
    return path, str(path)


@pytest.mark.parametrize("engine", ["graphchi", "xstream", "frontier"])
def test_an_engine_that_cannot_attribute_delays_refuses_the_request(
    tmp_path: pathlib.Path, engine: str
) -> None:
    path, text = _goal_text_stream(tmp_path)
    stream = native.open_goal_text_stream(text, chunk_nodes=16)

    with pytest.raises(ValueError, match="gridgraph"):
        native.solve_dag_stream(
            stream,
            engine=engine,
            delay_attribution=True,
            scratch_dir=str(tmp_path / "s"),
        )


def test_xstream_returns_the_critical_path_ops_like_the_other_engines(
    tmp_path: pathlib.Path,
) -> None:
    path, text = _goal_text_stream(tmp_path)

    def solve(engine: str) -> dict:
        stream = native.open_goal_text_stream(text, chunk_nodes=16)
        return native.solve_dag_stream(
            stream,
            engine=engine,
            memory_budget_bytes=32 * 1024,
            extract_critical_path_ops=True,
            scratch_dir=str(tmp_path / "s"),
        )

    reference = solve("gridgraph")
    got = solve("xstream")

    assert len(reference["cp_op_node_id"]) > 0
    for column in (
        "cp_op_rank",
        "cp_op_node_id",
        "cp_op_class",
        "cp_op_weight_ns",
    ):
        assert list(got[column]) == list(reference[column]), column
    assert math.isclose(got["makespan"], reference["makespan"], rel_tol=1e-12)


def test_a_scratch_dir_left_by_a_killed_run_is_never_reused(
    tmp_path: pathlib.Path,
) -> None:
    """Container pids repeat, so `run_<pid>_<n>` may hold stale spill files."""
    path = tmp_path / "chain.goal"
    path.write_text(
        "num_ranks 1\nrank 0 {\n"
        + "".join(f"l{i}: calc 1\n" for i in range(3))
        + "l1 requires l0\nl2 requires l1\n}\n"
    )
    scratch = tmp_path / "scratch"
    for seq in range(300):
        stale = scratch / f"run_{os.getpid()}_{seq}"
        stale.mkdir(parents=True)
        (stale / "eraw_0.bin").write_bytes(b"\0" * 24)  # one bogus edge record

    stream = native.open_goal_text_stream(str(path), chunk_nodes=16)
    raw = native.solve_dag_stream(
        stream, engine="gridgraph", scratch_dir=str(scratch)
    )

    assert raw["unordered"] == 0
    assert raw["num_nodes"] == 3
    assert raw["num_edges"] == 2


# --- Online aggregates --------------------------------------------------


def test_an_unknown_quantile_sketch_is_refused_even_when_nothing_spills(
    tmp_path: pathlib.Path,
) -> None:
    """The sketch is built only when the exact map overflows (rarely)."""
    path, text = _goal_text_stream(tmp_path)
    stream = native.open_goal_text_stream(text, chunk_nodes=16)

    with pytest.raises(ValueError, match="quantile_sketch"):
        native.stream_aggregates(stream, quantile_sketch="bogus")

    for kind in ("ddsketch", "kll", "gk"):
        native.stream_aggregates(
            native.open_goal_text_stream(text, chunk_nodes=16),
            quantile_sketch=kind,
        )


# --- Graph cache ---------------------------------------------------------


def test_a_graph_cache_with_sections_past_the_end_of_the_file_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    path, text = _goal_text_stream(tmp_path)
    cache = tmp_path / "g.cta_graph"
    native.build_graph_cache(
        native.open_goal_text_stream(text, chunk_nodes=16), str(cache)
    )
    data = cache.read_bytes()
    assert native.GraphCache(str(cache)).num_vertices > 0

    cut = tmp_path / "cut.cta_graph"
    cut.write_bytes(data[: len(data) - 64])
    with pytest.raises(RuntimeError, match="truncated"):
        native.GraphCache(str(cut))

    junk = tmp_path / "junk.cta_graph"
    junk.write_bytes(b"x" * 500)
    with pytest.raises((RuntimeError, ValueError)):
        native.GraphCache(str(junk))


# --- Numbers that do not fit ---------------------------------------------


@pytest.mark.parametrize("value", ["1e300", '"nan"', '"inf"', "9e18"])
@pytest.mark.parametrize("field", ["ts", "dur"])
def test_a_kineto_time_that_does_not_fit_in_nanoseconds_is_refused(
    tmp_path: pathlib.Path, field: str, value: str
) -> None:
    """Casting such a double is undefined and differs by CPU."""
    event = json.dumps(EVENT).replace(
        f'"{field}": {EVENT[field]}', f'"{field}": {value}'
    )
    assert value in event
    path = tmp_path / "t.json"
    path.write_text('{"traceEvents":[' + event + "]}")

    with pytest.raises(ValueError, match="ts and dur"):
        native.read_kineto(
            str(path), threads=1, chunk_bytes=1 << 20, block_bytes=1 << 20
        )


def test_a_host_trace_number_that_does_not_fit_in_an_int64_takes_its_default(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "host.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": 1,
                        "name": "aten::mm",
                        "dur_us": 1e300,
                        "comm_bytes": 1e300,
                    },
                    {
                        "id": 2,
                        "name": "aten::mm",
                        "dur_us": 10.0,
                        "data_deps": [1],
                    },
                ]
            }
        )
    )

    chunks = list(
        native.open_chakra_host_stream([str(path)], [0], chunk_nodes=8)
    )

    sizes = [int(x) for chunk in chunks for x in chunk["size_bytes"]]
    durations = [int(x) for chunk in chunks for x in chunk["duration_ns"]]
    assert sizes == [0, 0]
    assert durations == [0, 10_000]


def test_a_graph_cache_whose_build_died_early_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """The header is written first with every offset zero and patched last."""
    _, text = _goal_text_stream(tmp_path)
    cache = tmp_path / "g.cta_graph"
    native.build_graph_cache(
        native.open_goal_text_stream(text, chunk_nodes=16), str(cache)
    )
    data = bytearray(cache.read_bytes())
    # GraphCacheHeader is 128 bytes; its last five u64s are the section offsets.
    data[88:128] = bytes(40)
    zeroed = tmp_path / "zero.cta_graph"
    zeroed.write_bytes(bytes(data))

    with pytest.raises((RuntimeError, ValueError)):
        native.GraphCache(str(zeroed))
