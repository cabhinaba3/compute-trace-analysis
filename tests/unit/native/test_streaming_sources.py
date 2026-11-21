"""Unit tests for Phase 1 native streaming graph sources and chunked ingest.

Verifies exact parity across tiny and large chunk sizes for GOAL .bin,
GOAL .goal (legacy and strict forward-ref modes), Chakra .et, and Chakra
host JSON, plus unseekable FIFO pipes, reset(), error handling, and macOS/Linux
memory detection.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
from typing import Any

import numpy as np
import polars as pl
import pytest

from cta import _native as native
from cta.adapters import common
from cta.adapters.chakra import reader as chakra_reader
from cta.adapters.goal import reader as goal_reader
from cta.synth import chakra as synth_chakra
from cta.synth import chakra_proto
from cta.synth import goal as synth_goal
from cta.synth import training


def _collect_stream(
    stream: native.GraphStream,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Drains a native GraphStream into concatenated node and edge frames."""
    node_frames: list[pl.DataFrame] = []
    edge_frames: list[pl.DataFrame] = []
    for chunk in stream:
        nodes_df, edges_df = common.chunk_to_graph_frames(
            chunk, chakra_reader.NODES_SCHEMA, chakra_reader.EDGES_SCHEMA
        )
        if not nodes_df.is_empty():
            node_frames.append(nodes_df)
        if not edges_df.is_empty():
            edge_frames.append(edges_df)
    nodes = (
        pl.concat(node_frames)
        if node_frames
        else pl.DataFrame(schema=chakra_reader.NODES_SCHEMA)
    )
    edges = (
        pl.concat(edge_frames)
        if edge_frames
        else pl.DataFrame(schema=chakra_reader.EDGES_SCHEMA)
    )
    return nodes, edges


def test_available_memory_and_budget() -> None:
    avail = native.available_memory_bytes()
    assert avail > 0
    assert native.resolve_memory_budget(123456) == 123456
    auto = native.resolve_memory_budget(0)
    assert auto > 0
    assert auto <= avail


def test_goal_bin_stream_parity_and_chunk_invariance(
    tmp_path: pathlib.Path,
) -> None:
    spec = training.TrainingSpec(
        dp=2, pp=2, tp=2, microbatches=2, layers_per_stage=2, steps=2
    )
    goal_path = tmp_path / "train.goal"
    bin_path = tmp_path / "train.bin"
    training.write_goal(goal_path, spec)
    training.write_goal_bin(bin_path, spec)

    text_nodes, text_edges = goal_reader.parse_goal_text(goal_path)
    bin_nodes, bin_edges = goal_reader._parse_goal_bin(bin_path)

    # Node counts, weights, sizes, and edges must match text schedule exactly.
    assert bin_nodes.height == text_nodes.height
    assert bin_edges.height == text_edges.height
    assert bin_nodes["rank"].to_list() == text_nodes["rank"].to_list()
    assert bin_nodes["op_class"].to_list() == text_nodes["op_class"].to_list()
    assert (
        bin_nodes["duration_ns"].to_list()
        == text_nodes["duration_ns"].to_list()
    )
    assert (
        bin_nodes["size_bytes"].to_list() == text_nodes["size_bytes"].to_list()
    )
    assert set(bin_edges.iter_rows()) == set(text_edges.iter_rows())

    # Streaming across tiny and large chunk sizes produces identical tables.
    for chunk_size in (1, 7, 64, 100000):
        stream = native.open_goal_bin_stream(
            str(bin_path), chunk_nodes=chunk_size
        )
        assert stream.is_seekable is True
        assert stream.format_name == "goal_bin"
        assert stream.estimated_bytes == bin_path.stat().st_size
        nodes, edges = _collect_stream(stream)
        assert nodes.equals(bin_nodes)
        assert edges.equals(bin_edges)

        # Resetting a seekable stream yields the exact same data again.
        stream.reset()
        nodes2, edges2 = _collect_stream(stream)
        assert nodes2.equals(bin_nodes)
        assert edges2.equals(bin_edges)


def test_goal_bin_invalid_cookie_rejected(tmp_path: pathlib.Path) -> None:
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x00" * 32)
    with pytest.raises(ValueError, match="magic cookie"):
        native.open_goal_bin_stream(str(bad))


def test_goal_text_stream_legacy_vs_strict_forward_refs(
    tmp_path: pathlib.Path,
) -> None:
    schedule = (
        "num_ranks 2\n"
        "rank 0 {\n"
        "a: calc 10\n"
        "a requires b\n"
        "b: calc 20\n"
        "b irequires c\n"
        "c: send 1kb to 1 tag 4\n"
        "c requires missing_label\n"
        "}\n"
        "rank 1 {\n"
        "r0: recv 1kb from 0 tag 4\n"
        "}\n"
    )
    path = tmp_path / "fwd.goal"
    path.write_text(schedule, encoding="utf-8")

    # Legacy mode drops forward label refs (matching cta's read_goal).
    ref_nodes, ref_edges = goal_reader.parse_goal_text(path)
    for chunk_size in (1, 2, 100):
        stream = native.open_goal_text_stream(
            str(path), goal_semantics="cta_legacy", chunk_nodes=chunk_size
        )
        nodes, edges = _collect_stream(stream)
        assert nodes.equals(ref_nodes)
        assert edges.equals(ref_edges)
        assert edges.height == 0

    # Strict mode resolves both forward requires and forward irequires within }.
    strict_stream = native.open_goal_text_stream(
        str(path), goal_semantics="strict", chunk_nodes=1
    )
    s_nodes, s_edges = _collect_stream(strict_stream)
    assert s_nodes.equals(ref_nodes)
    assert set(s_edges.iter_rows()) == {
        (0, 1, 0, "requires"),
        (0, 2, 1, "irequires"),
    }


def test_chakra_et_stream_chunk_invariance_and_comm_group(
    tmp_path: pathlib.Path,
) -> None:
    n0 = chakra_proto.NodeMsg(
        id=10,
        name="gemm",
        type=chakra_proto.COMP_NODE,
        duration_micros=25,
    )
    n1 = chakra_proto.NodeMsg(
        id=20,
        name="allreduce",
        type=chakra_proto.COMM_COLL_NODE,
        duration_micros=40,
        data_deps=[10],
        ctrl_deps=[10],
        attr=[
            chakra_proto.AttributeProtoMsg(name="comm_type", int32_val=0),
            chakra_proto.AttributeProtoMsg(name="comm_size", int64_val=4096),
            chakra_proto.AttributeProtoMsg(name="pg_name", string_val="dp_pg"),
        ],
    )
    et_path = tmp_path / "trace.rank0.et"
    buf = bytearray()
    for msg in (chakra_proto.GlobalMetadataMsg(version="0.0.4"), n0, n1):
        payload = msg.SerializeToString()
        buf += chakra_proto.encode_varint(len(payload)) + payload
    et_path.write_bytes(bytes(buf))

    stream = native.open_chakra_et_stream([str(et_path)], [0], chunk_nodes=1)
    assert stream.is_seekable is True
    chunks = list(stream)
    assert len(chunks) == 2
    assert common.strings_from_ends(
        chunks[1]["comm_group_end"], chunks[1]["comm_group_bytes"]
    ).to_list() == ["dp_pg"]

    stream.reset()
    nodes, edges = _collect_stream(stream)
    assert nodes["node_id"].to_list() == [10, 20]
    assert nodes["duration_ns"].to_list() == [25_000, 40_000]
    assert edges.rows() == [(0, 10, 20, "data"), (0, 10, 20, "ctrl")]


def test_chakra_host_stream_parity(tmp_path: pathlib.Path) -> None:
    host_path = tmp_path / "host_et.rank2.json"
    payload = {
        "schema": "1.0.2-chakra.0.0.4",
        "nodes": [
            {
                "id": 1,
                "name": "aten::mm",
                "dur_us": 12.5,
                "tensor_bytes": 1024,
                "ctrl_deps": [],
                "data_deps": [],
            },
            {
                "id": 2,
                "name": "nccl:all_reduce",
                "duration_micros": 30,
                "comm_bytes": 8192,
                "ctrl_deps": [1],
                "data_deps": [1],
            },
            {
                "id": 3,
                "name": "nccl:send_peer",
                "dur_us": 5,
                "comm_bytes": 256,
                "comm_dst": 3,
                "comm_tag": 9,
                "ctrl_deps": [2],
            },
        ],
    }
    host_path.write_text(json.dumps(payload), encoding="utf-8")

    for chunk_size in (1, 2, 100):
        stream = native.open_chakra_host_stream(
            [str(host_path)], [2], chunk_nodes=chunk_size
        )
        nodes, edges = _collect_stream(stream)
        assert nodes["rank"].to_list() == [2, 2, 2]
        assert nodes["op_class"].to_list() == [
            "compute",
            "comm_coll",
            "comm_send",
        ]
        assert nodes["comm_type"].to_list() == ["", "ALL_REDUCE", "SEND"]
        assert nodes["duration_ns"].to_list() == [12_500, 30_000, 5_000]
        assert nodes["size_bytes"].to_list() == [1024, 8192, 256]
        assert nodes["src_rank"].to_list() == [-1, -1, 2]
        assert nodes["dst_rank"].to_list() == [-1, -1, 3]
        assert nodes["message_tag"].to_list() == [0, 0, 9]
        assert edges.rows() == [
            (2, 1, 2, "ctrl"),
            (2, 1, 2, "data"),
            (2, 2, 3, "ctrl"),
        ]


def test_unseekable_fifo_pipes(tmp_path: pathlib.Path) -> None:
    """Streams GOAL text, GOAL bin, Chakra .et, and host JSON over FIFOs."""
    goal_src = synth_goal.write_allreduce(
        tmp_path / "ring.goal", world_size=3, steps=2
    )
    bin_src = synth_goal.compile_to_bin(goal_src, tmp_path / "ring.bin")
    et_paths = synth_chakra.write_run(
        tmp_path / "et_dir", world_size=1, steps=2
    )
    host_src = synth_chakra.write_host_trace(
        tmp_path / "host.json", steps=2, compute_us=100.0, comm_bytes=4096
    )

    cases: list[tuple[str, pathlib.Path, Any]] = [
        (
            "goal_text",
            goal_src,
            lambda p: native.open_goal_text_stream(p, chunk_nodes=4),
        ),
        (
            "goal_bin",
            bin_src,
            lambda p: native.open_goal_bin_stream(p, chunk_nodes=4),
        ),
        (
            "chakra_et",
            et_paths[0],
            lambda p: native.open_chakra_et_stream([p], [0], chunk_nodes=4),
        ),
        (
            "chakra_host",
            host_src,
            lambda p: native.open_chakra_host_stream([p], [0], chunk_nodes=4),
        ),
    ]

    for name, src_file, opener in cases:
        fifo = tmp_path / f"{name}.fifo"
        os.mkfifo(fifo)
        raw = src_file.read_bytes()

        def _writer(target: pathlib.Path, data: bytes) -> None:
            with target.open("wb") as fh:
                fh.write(data)

        thread = threading.Thread(target=_writer, args=(fifo, raw))
        thread.start()
        try:
            stream = opener(str(fifo))
            assert stream.is_seekable is False
            nodes, edges = _collect_stream(stream)
            assert nodes.height > 0
            assert edges.height > 0
            with pytest.raises(RuntimeError, match="unseekable"):
                stream.reset()
        finally:
            thread.join(timeout=5.0)
            assert not thread.is_alive()


def test_streaming_parquet_writer_empty_and_chunked(
    tmp_path: pathlib.Path,
) -> None:
    empty_path = tmp_path / "empty.parquet"
    with common.StreamingParquetWriter(
        empty_path, chakra_reader.NODES_SCHEMA
    ) as writer:
        writer.write_frame(pl.DataFrame(schema=chakra_reader.NODES_SCHEMA))
    loaded = pl.read_parquet(empty_path)
    assert loaded.height == 0
    assert loaded.schema == chakra_reader.NODES_SCHEMA

    # Verify numpy arrays from chunked read round-trip cleanly.
    spec = training.TrainingSpec(dp=2, pp=1, tp=2, steps=1)
    bin_path = tmp_path / "small.bin"
    training.write_goal_bin(bin_path, spec)
    out_path = tmp_path / "nodes.parquet"
    with common.StreamingParquetWriter(
        out_path, chakra_reader.NODES_SCHEMA
    ) as writer:
        for chunk in native.open_goal_bin_stream(str(bin_path), chunk_nodes=5):
            nodes_df, _ = common.chunk_to_graph_frames(
                chunk, chakra_reader.NODES_SCHEMA, chakra_reader.EDGES_SCHEMA
            )
            writer.write_frame(nodes_df)
    full_nodes, _ = goal_reader._parse_goal_bin(bin_path)
    assert pl.read_parquet(out_path).equals(full_nodes)
    assert isinstance(
        np.asarray(full_nodes["duration_ns"].to_numpy()), np.ndarray
    )
