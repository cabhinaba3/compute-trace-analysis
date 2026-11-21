"""Unit tests for Phase 3 Engine B (1-pass frontier, Eviction-Epoch, barriers).

Tests exact parity of GOAL .bin frontier against solve_dag (including
concurrent same-tag sends, unmatched recvs, strict irequires, CP op chains, and
cycle witness), Certified Eviction-Epoch intervals [makespan_lb, makespan_ub]
on both dense monotonic and sparse 2^63 Chakra .et traces, dangling-dep
lookahead safety, and 2-node virtual collective barriers with strict
window_local vs global_exact slack certification.
"""

from __future__ import annotations

import pathlib

import pytest

from cta import _native as native
from cta import models
from cta.adapters.goal import reader as goal_reader
from cta.analysis import dag
from cta.analysis import streaming
from cta.synth import chakra as synth_chakra
from cta.synth import chakra_proto
from cta.synth import goal as synth_goal
from cta.synth import training


def _write_et(path: pathlib.Path, nodes: list[object]) -> pathlib.Path:
    buf = bytearray()
    for msg in (chakra_proto.GlobalMetadataMsg(version="0.0.4"), *nodes):
        payload = msg.SerializeToString()  # type: ignore[attr-defined]
        buf += chakra_proto.encode_varint(len(payload)) + payload
    path.write_bytes(bytes(buf))
    return path


def test_goal_bin_frontier_exact_parity_with_solve_dag(
    tmp_path: pathlib.Path,
) -> None:
    spec = training.TrainingSpec(
        dp=3,
        pp=2,
        tp=2,
        microbatches=2,
        layers_per_stage=2,
        steps=2,
        moe_every=2,
        ep_size=3,
    )
    bin_path = tmp_path / "train.bin"
    training.write_goal_bin(bin_path, spec)

    nodes, edges = goal_reader._parse_goal_bin(bin_path)
    ref = dag.solve_distributed_dag(nodes, edges)

    art = models.TraceArtifact(
        format="goal", paths=(bin_path,), world_size=spec.ranks
    )
    got = streaming.solve_artifact_frontier(
        art, extract_critical_path_ops=True, scratch_dir=str(tmp_path / "scr")
    )

    assert got.provenance.engine == "frontier"
    assert got.provenance.exact is True
    assert got.provenance.slack_scope == "unavailable"
    assert got.num_nodes == ref.num_nodes
    assert got.num_edges == ref.num_edges
    assert got.dag_depth == ref.dag_depth
    assert got.critical_path_ns == pytest.approx(
        ref.critical_path_ns, rel=1e-12
    )
    assert got.cp_compute_ns == pytest.approx(ref.cp_compute_ns, rel=1e-12)
    assert got.cp_exposed_comm_ns == pytest.approx(
        ref.cp_exposed_comm_ns, rel=1e-12
    )
    assert got.cp_comm_bytes == ref.cp_comm_bytes

    # Critical-path op chain must sum to the exact makespan and depth.
    assert got.critical_path_ops.height == ref.dag_depth
    assert float(got.critical_path_ops["weight_ns"].sum()) == pytest.approx(
        ref.critical_path_ns, rel=1e-12
    )
    assert float(
        got.critical_path_ops["earliest_finish_ns"][-1]
    ) == pytest.approx(ref.critical_path_ns, rel=1e-12)


def test_goal_bin_frontier_row_order_p2p_and_unmatched_recvs(
    tmp_path: pathlib.Path,
) -> None:
    """Row-order P2P pairing holds even when send #2 is ready before send #1."""
    schedule = (
        "num_ranks 2\n"
        "rank 0 {\n"
        "slow: calc 5000\n"
        "s0: send 100b to 1 tag 7\n"
        "s0 requires slow\n"
        "s1: send 100b to 1 tag 7\n"
        "}\n"
        "rank 1 {\n"
        "r0: recv 100b from 0 tag 7\n"
        "r1: recv 100b from 0 tag 7\n"
        "r_extra: recv 100b from 0 tag 7\n"
        "after0: calc 100\n"
        "after0 requires r0\n"
        "after_extra: calc 50\n"
        "after_extra requires r_extra\n"
        "}\n"
    )
    goal_path = tmp_path / "p2p_order.goal"
    goal_path.write_text(schedule, encoding="utf-8")
    bin_path = synth_goal.compile_to_bin(goal_path, tmp_path / "p2p_order.bin")

    nodes, edges = goal_reader._parse_goal_bin(bin_path)
    ref = dag.solve_distributed_dag(nodes, edges)

    raw = native.solve_goal_bin_frontier(str(bin_path))
    got = streaming.dag_result_from_native(raw)
    assert got.provenance.exact is True
    assert got.provenance.unmatched_p2p_recvs == 1
    assert got.provenance.unmatched_p2p_sends == 0
    assert got.critical_path_ns == pytest.approx(
        ref.critical_path_ns, rel=1e-12
    )
    assert got.dag_depth == ref.dag_depth
    assert got.num_edges == ref.num_edges


def test_goal_bin_frontier_strict_irequires(tmp_path: pathlib.Path) -> None:
    """Strict irequires (start-to-start) lets B start when A starts."""
    schedule = (
        "num_ranks 1\nrank 0 {\na: calc 1000\nb: calc 600\nb irequires a\n}\n"
    )
    goal_path = tmp_path / "ireq.goal"
    goal_path.write_text(schedule, encoding="utf-8")
    bin_path = synth_goal.compile_to_bin(goal_path, tmp_path / "ireq.bin")

    legacy = streaming.dag_result_from_native(
        native.solve_goal_bin_frontier(
            str(bin_path), goal_semantics="cta_legacy"
        )
    )
    strict = streaming.dag_result_from_native(
        native.solve_goal_bin_frontier(str(bin_path), goal_semantics="strict")
    )
    assert legacy.critical_path_ns == pytest.approx(1600.0)
    assert strict.critical_path_ns == pytest.approx(1000.0)
    assert strict.provenance.goal_semantics == "strict"


def test_chakra_eviction_epoch_certified_bounds_dense_and_sparse(
    tmp_path: pathlib.Path,
) -> None:
    """Tight window emits sound [makespan_lb, makespan_ub] on horizon misses."""
    # Dense monotonic IDs: node 0 -> 1 -> ... -> 49, PLUS skip edge 0 -> 49.
    dense_nodes = [
        chakra_proto.NodeMsg(
            id=i,
            name=f"op_{i}",
            type=chakra_proto.COMP_NODE,
            duration_micros=10,
            data_deps=([i - 1, 0] if i == 49 else ([i - 1] if i > 0 else [])),
        )
        for i in range(50)
    ]
    dense_et = _write_et(tmp_path / "dense.rank0.et", dense_nodes)
    # Full window -> exact (50 * 10us = 500_000 ns).
    exact_res = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier([str(dense_et)], [0], window_size=100)
    )
    assert exact_res.provenance.exact is True
    assert exact_res.provenance.evicted_dep_hits == 0
    assert exact_res.critical_path_ns == pytest.approx(500_000.0)

    # Tiny window (size=5) evicts node 0 long before node 49 references it.
    tight_res = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier([str(dense_et)], [0], window_size=5)
    )
    assert tight_res.provenance.exact is False
    assert tight_res.provenance.evicted_dep_hits == 1
    assert (
        tight_res.provenance.makespan_lb
        <= exact_res.critical_path_ns
        <= tight_res.provenance.makespan_ub
    )

    # Sparse huge 64-bit IDs + dangling dep on s4 so s0 commits and gets
    # evicted before s4 arrives referencing both evicted s0 and dangling ID.
    sparse_ids = [10, 2**60 + 3, 2**62 + 7, 2**63 + 11, 2**63 + 99]
    sparse_nodes = [
        chakra_proto.NodeMsg(
            id=sparse_ids[0],
            name="s0",
            type=chakra_proto.COMP_NODE,
            duration_micros=50,
            data_deps=[],
        ),
        chakra_proto.NodeMsg(
            id=sparse_ids[1],
            name="s1",
            type=chakra_proto.COMP_NODE,
            duration_micros=20,
            data_deps=[sparse_ids[0]],
        ),
        chakra_proto.NodeMsg(
            id=sparse_ids[2],
            name="s2",
            type=chakra_proto.COMP_NODE,
            duration_micros=20,
            data_deps=[sparse_ids[1]],
        ),
        chakra_proto.NodeMsg(
            id=sparse_ids[3],
            name="s3",
            type=chakra_proto.COMP_NODE,
            duration_micros=20,
            data_deps=[sparse_ids[2]],
        ),
        chakra_proto.NodeMsg(
            id=sparse_ids[4],
            name="s4",
            type=chakra_proto.COMP_NODE,
            duration_micros=30,
            data_deps=[sparse_ids[0], sparse_ids[3], 999_999_999],
        ),
    ]
    sparse_et = _write_et(tmp_path / "sparse.rank0.et", sparse_nodes)
    sp_exact = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier([str(sparse_et)], [0], window_size=50)
    )
    assert sp_exact.provenance.exact is True
    assert sp_exact.critical_path_ns == pytest.approx(140_000.0)

    sp_tight = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier([str(sparse_et)], [0], window_size=2)
    )
    assert sp_tight.provenance.exact is False
    assert sp_tight.provenance.evicted_dep_hits >= 1
    assert (
        sp_tight.provenance.makespan_lb
        <= sp_exact.critical_path_ns
        <= sp_tight.provenance.makespan_ub
    )

    # Also verify dangling dep on root cascades topologically at EOF.
    dangling_root = [
        chakra_proto.NodeMsg(
            id=1,
            name="r0",
            type=chakra_proto.COMP_NODE,
            duration_micros=50,
            data_deps=[777_777],
        ),
        chakra_proto.NodeMsg(
            id=2,
            name="r1",
            type=chakra_proto.COMP_NODE,
            duration_micros=30,
            data_deps=[1],
        ),
    ]
    dr_et = _write_et(tmp_path / "dangling_root.rank0.et", dangling_root)
    dr_res = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier([str(dr_et)], [0], window_size=50)
    )
    assert dr_res.provenance.exact is True
    assert dr_res.critical_path_ns == pytest.approx(80_000.0)


def test_chakra_virtual_collective_barriers_and_cut_certification(
    tmp_path: pathlib.Path,
) -> None:
    """Global barrier certifies global_exact slack iff no dead-end/straddle."""
    # Clean 2-step run with a straggler on rank 1 (2x compute time).
    clean_dir = tmp_path / "clean_et"
    paths = synth_chakra.write_run(
        clean_dir,
        world_size=2,
        compute_us=1000,
        allreduce_bytes=100_000,
        steps=2,
        straggler_rank=1,
        straggler_factor=2.0,
    )
    ranks = [0, 1]

    # Without barriers: each rank is independent; makespan = rank 1's total.
    no_bar = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier(
            [str(p) for p in paths], ranks, collective_sync="none"
        )
    )
    # With barriers: rank 0 waits for rank 1 at each step's AllReduce, and
    # every node reaches B_out with zero straddling edges -> global_exact slack!
    with_bar = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier(
            [str(p) for p in paths], ranks, collective_sync="barrier"
        )
    )
    assert with_bar.provenance.collective_sync == "barrier"
    assert with_bar.provenance.straddling_edges == 0
    assert with_bar.provenance.dead_end_sinks == 0
    assert with_bar.provenance.slack_scope == "global_exact"
    assert with_bar.provenance.exact is True
    assert with_bar.critical_path_ns == pytest.approx(
        no_bar.critical_path_ns, rel=1e-12
    )
    assert with_bar.mean_slack_ns > 0.0  # rank 0 compute has 1000us slack/step

    # Now add a dead-end local sink on rank 0 that never reaches the barrier:
    # slack MUST be tagged window_local with dead_end_sinks == 1!
    r0_nodes = [
        chakra_proto.NodeMsg(
            id=0, name="c0", type=chakra_proto.COMP_NODE, duration_micros=100
        ),
        chakra_proto.NodeMsg(
            id=1,
            name="async_log_sink",
            type=chakra_proto.COMP_NODE,
            duration_micros=10,
            data_deps=[0],
        ),
        chakra_proto.NodeMsg(
            id=2,
            name="ar0",
            type=chakra_proto.COMM_COLL_NODE,
            duration_micros=50,
            data_deps=[0],
            attr=[
                chakra_proto.AttributeProtoMsg(name="comm_type", int32_val=0)
            ],
        ),
        chakra_proto.NodeMsg(
            id=3,
            name="c1",
            type=chakra_proto.COMP_NODE,
            duration_micros=100,
            data_deps=[2],
        ),
    ]
    r1_nodes = [
        chakra_proto.NodeMsg(
            id=0, name="c0", type=chakra_proto.COMP_NODE, duration_micros=200
        ),
        chakra_proto.NodeMsg(
            id=1,
            name="ar0",
            type=chakra_proto.COMM_COLL_NODE,
            duration_micros=50,
            data_deps=[0],
            attr=[
                chakra_proto.AttributeProtoMsg(name="comm_type", int32_val=0)
            ],
        ),
        chakra_proto.NodeMsg(
            id=2,
            name="c1",
            type=chakra_proto.COMP_NODE,
            duration_micros=100,
            data_deps=[1],
        ),
    ]
    p0 = _write_et(tmp_path / "sink.rank0.et", r0_nodes)
    p1 = _write_et(tmp_path / "sink.rank1.et", r1_nodes)
    sink_res = streaming.dag_result_from_native(
        native.solve_chakra_et_frontier(
            [str(p0), str(p1)], [0, 1], collective_sync="barrier"
        )
    )
    assert sink_res.provenance.dead_end_sinks == 1
    assert sink_res.provenance.slack_scope == "window_local"
    assert sink_res.provenance.exact is False
