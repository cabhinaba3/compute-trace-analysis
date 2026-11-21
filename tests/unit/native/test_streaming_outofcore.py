"""Unit tests for Phase 4 pluggable out-of-core DAG engines and Scalasca delays.

Verifies bit-exact oracle parity across GridGraph (USENIX ATC '15), GraphChi
PSW (OSDI '12), and X-Stream Scatter-Gather (SOSP '13) against CTA's in-memory
solve_dag under artificially tight memory budgets (forcing multi-block /
fully-external mode), adversarial Theta(V) cutwidth, duplicate rank headers,
cycles + sample cycle witness + --fail-on-cycle, strict GOAL irequires exact
backward slack LF[u] <= LF[v] - w[v] + w[u], persistent .cta_graph cache reuse,
and Scalasca backward wait-state & root-cause delay attribution (IPDPS '12).
"""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from cta import _native as native
from cta import models
from cta.adapters.chakra import reader as chakra_reader
from cta.adapters.goal import reader as goal_reader
from cta.analysis import dag
from cta.analysis import streaming
from cta.synth import chakra as synth_chakra
from cta.synth import chakra_proto
from cta.synth import goal as synth_goal
from cta.synth import training


def test_available_graph_engines_lists_all_pluggable_papers() -> None:
    engines = native.available_graph_engines()
    assert set(engines) >= {"gridgraph", "frontier", "graphchi", "xstream"}


@pytest.mark.parametrize("engine_name", ["gridgraph", "graphchi", "xstream"])
@pytest.mark.parametrize("budget_bytes", [0, 64 * 1024])
def test_all_outofcore_engines_match_solve_dag_exact(
    tmp_path: pathlib.Path, engine_name: str, budget_bytes: int
) -> None:
    """GridGraph, GraphChi, and X-Stream match solve_dag under 64 KiB cap."""
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
    goal_bin = tmp_path / "train.bin"
    training.write_goal_bin(goal_bin, spec)
    nodes, edges = goal_reader._parse_goal_bin(goal_bin)
    ref = dag.solve_distributed_dag(nodes, edges)

    art = models.TraceArtifact(
        format="goal", paths=(goal_bin,), world_size=spec.ranks
    )
    got = streaming.solve_artifact_outofcore(
        art,
        engine=engine_name,
        memory_budget_bytes=budget_bytes,
        extract_critical_path_ops=True,
        scratch_dir=str(tmp_path / "scratch"),
    )

    assert got.provenance.engine == engine_name
    assert got.provenance.exact is True
    assert got.provenance.slack_scope == "global_exact"
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
    assert got.mean_slack_ns == pytest.approx(ref.mean_slack_ns, rel=1e-12)
    assert got.zero_slack_ratio == pytest.approx(
        ref.zero_slack_ratio, rel=1e-12
    )

    if engine_name == "gridgraph":
        assert got.critical_path_ops.height == ref.dag_depth
        assert float(got.critical_path_ops["weight_ns"].sum()) == pytest.approx(
            ref.critical_path_ns, rel=1e-12
        )


def test_outofcore_handles_out_of_order_sparse_ids_and_adversarial_cutwidth(
    tmp_path: pathlib.Path,
) -> None:
    """Tests reverse-ordered Chakra nodes, sparse 64-bit IDs, and skip edges."""
    # Reverse order in file: node 2000 depends on 0..1999 (adversarial fan-in)
    # and node 0 fans out to 1..2000 (adversarial cutwidth across blocks).
    n = 2200
    ids = [(i * 104729) + (1 << 40) for i in range(n)]
    nodes = []
    # Write nodes in REVERSE order to force out-of-core topological sorting!
    for i in range(n - 1, -1, -1):
        deps = []
        if i > 0:
            deps.append(ids[i - 1])
            if i % 17 == 0:
                deps.append(ids[0])  # long-range skip edge spanning blocks
                deps.append(ids[0])  # duplicate edge (must deduplicate!)
        nodes.append(
            chakra_proto.NodeMsg(
                id=ids[i],
                name=f"op_{i}",
                type=chakra_proto.COMP_NODE,
                duration_micros=5 + (i % 7),
                data_deps=deps,
            )
        )
    et_path = tmp_path / "rev.rank0.et"
    buf = bytearray()
    for msg in (chakra_proto.GlobalMetadataMsg(version="0.0.4"), *nodes):
        payload = msg.SerializeToString()
        buf += chakra_proto.encode_varint(len(payload)) + payload
    et_path.write_bytes(bytes(buf))

    art = models.TraceArtifact(format="chakra", paths=(et_path,), world_size=1)
    tables = chakra_reader.ChakraAdapter().ingest(art, tmp_path / "pq")
    ref = dag.solve_distributed_dag(
        pl.read_parquet(tables["nodes"]), pl.read_parquet(tables["edges"])
    )

    # Force multi-block out-of-core execution with a 32 KiB RAM budget!
    cache_file = tmp_path / "rev.cta_graph"
    got = streaming.solve_artifact_outofcore(
        art,
        engine="gridgraph",
        memory_budget_bytes=32 * 1024,
        extract_critical_path_ops=True,
        scratch_dir=str(tmp_path / "scr"),
        cache_path=cache_file,
    )
    assert cache_file.exists()
    assert got.provenance.exact is True
    assert got.num_nodes == ref.num_nodes == n
    assert got.num_edges == ref.num_edges
    assert got.dag_depth == ref.dag_depth == n
    assert got.critical_path_ns == pytest.approx(
        ref.critical_path_ns, rel=1e-12
    )
    assert got.mean_slack_ns == pytest.approx(ref.mean_slack_ns, rel=1e-12)
    assert got.zero_slack_ratio == pytest.approx(1.0, rel=1e-12)

    # Re-solving directly from the persisted .cta_graph cache matches.
    reused = streaming.solve_artifact_outofcore(
        art,
        engine="gridgraph",
        memory_budget_bytes=32 * 1024,
        cache_path=cache_file,
        scratch_dir=str(tmp_path / "scr2"),
    )
    assert reused.critical_path_ns == pytest.approx(got.critical_path_ns)
    assert reused.mean_slack_ns == pytest.approx(got.mean_slack_ns)


def test_strict_irequires_exact_backward_slack(tmp_path: pathlib.Path) -> None:
    """Verifies LF[u] <= LF[v] - w[v] + w[u] for start-to-start irequires."""
    # A (1000ns) and B (600ns) with B irequires A, followed by C (200ns)
    # requiring B: B starts at ES[A]=0, finishes at 600, C finishes at 800.
    # Makespan is 1000 (from A). B has LF[B] = 1000 - 200 = 800, LS[B] = 200.
    # For A: irequires constrains LS[A] <= LS[B] = 200 -> LF[A] <= 200 + 1000
    # = 1200, clamped to makespan = 1000 -> slack[A] = 0!
    # In contrast, if A had weight 300 and B had weight 800, C had weight 200:
    # makespan = 1000 (B->C), LF[B]=800, LS[B]=0 -> LS[A] <= 0 -> LF[A] <= 300
    # -> slack[A] = 0 (NOT 700 as if A had no outgoing edge, and NOT -500!).
    schedule = (
        "num_ranks 1\n"
        "rank 0 {\n"
        "a: calc 300\n"
        "b: calc 800\n"
        "c: calc 200\n"
        "b irequires a\n"
        "c requires b\n"
        "}\n"
    )
    goal_path = tmp_path / "strict_ireq.goal"
    goal_path.write_text(schedule, encoding="utf-8")
    bin_path = synth_goal.compile_to_bin(goal_path, tmp_path / "strict.bin")
    art = models.TraceArtifact(format="goal", paths=(bin_path,), world_size=1)

    strict = streaming.solve_artifact_outofcore(
        art,
        engine="gridgraph",
        goal_semantics="strict",
        scratch_dir=str(tmp_path / "scr"),
    )
    assert strict.provenance.goal_semantics == "strict"
    assert strict.critical_path_ns == pytest.approx(1000.0)
    # Every node (A, B, C) is critical under strict irequires -> mean_slack = 0!
    assert strict.mean_slack_ns == pytest.approx(0.0, abs=1e-9)
    assert strict.zero_slack_ratio == pytest.approx(1.0)


def test_cycle_witness_and_fail_on_cycle(tmp_path: pathlib.Path) -> None:
    schedule = (
        "num_ranks 1\n"
        "rank 0 {\n"
        "n0: calc 10\n"
        "n1: calc 20\n"
        "n2: calc 30\n"
        "n1 requires n0\n"
        "n2 requires n1\n"
        "n1 requires n2\n"
        "}\n"
    )
    path = tmp_path / "cycle.goal"
    path.write_text(schedule, encoding="utf-8")
    art = models.TraceArtifact(format="goal", paths=(path,), world_size=1)

    res = streaming.solve_artifact_outofcore(
        art,
        engine="gridgraph",
        goal_semantics="strict",
        scratch_dir=str(tmp_path / "scr"),
    )
    assert res.provenance.has_cycle is True
    assert res.provenance.exact is False
    assert res.unordered == 2
    assert len(res.provenance.cycle_witness) == 2
    assert set(res.provenance.cycle_witness) == {(0, 1), (0, 2)}

    with pytest.raises(RuntimeError, match="cycle detected"):
        streaming.solve_artifact_outofcore(
            art,
            engine="gridgraph",
            goal_semantics="strict",
            fail_on_cycle=True,
            scratch_dir=str(tmp_path / "scr2"),
        )


def test_scalasca_backward_delay_attribution_identifies_straggler(
    tmp_path: pathlib.Path,
) -> None:
    """Scalasca Pass 2 attributes collective barrier wait to straggler rank."""
    et_dir = tmp_path / "straggler_et"
    paths = synth_chakra.write_run(
        et_dir,
        world_size=4,
        compute_us=1000,
        allreduce_bytes=100_000,
        steps=2,
        straggler_rank=2,
        straggler_factor=2.5,
    )
    art = models.TraceArtifact(
        format="chakra", paths=tuple(sorted(paths)), world_size=4
    )
    res = streaming.solve_artifact_outofcore(
        art,
        engine="gridgraph",
        collective_sync="barrier",
        delay_attribution=True,
        scratch_dir=str(tmp_path / "scr"),
    )
    assert res.provenance.exact is True
    assert res.provenance.collective_sync == "barrier"
    assert res.provenance.delay_timing_basis == "measured"
    assert res.root_cause_delays.height > 0
    # Top root-cause culprit must be rank 2 (the 2.5x straggler compute op)!
    top = res.root_cause_delays.row(0, named=True)
    assert int(top["rank"]) == 2
    assert float(top["propagated_delay_ns"]) > 0.0


@pytest.mark.parametrize("engine_name", ["gridgraph", "graphchi", "xstream"])
def test_duplicate_rank_headers_and_duplicate_node_ids_match_oracle(
    tmp_path: pathlib.Path, engine_name: str
) -> None:
    """Duplicate rank blocks in GOAL text match solve_distributed_dag."""
    schedule = (
        "num_ranks 2\n"
        "rank 0 {\n"
        "c0: calc 100\n"
        "s0: send 64b to 1 tag 1\n"
        "s0 requires c0\n"
        "}\n"
        "rank 1 {\n"
        "r0: recv 64b from 0 tag 1\n"
        "c1: calc 250\n"
        "c1 requires r0\n"
        "}\n"
        "rank 0 {\n"
        "c2: calc 40\n"
        "s1: send 32b to 1 tag 2\n"
        "s1 requires c2\n"
        "}\n"
        "rank 1 {\n"
        "r1: recv 32b from 0 tag 2\n"
        "c3: calc 50\n"
        "c3 requires r1\n"
        "}\n"
    )
    goal_path = tmp_path / "dup_ranks.goal"
    goal_path.write_text(schedule, encoding="utf-8")
    nodes, edges = goal_reader.parse_goal_text(goal_path)
    ref = dag.solve_distributed_dag(nodes, edges)

    art = models.TraceArtifact(format="goal", paths=(goal_path,), world_size=2)
    got = streaming.solve_artifact_outofcore(
        art,
        engine=engine_name,
        scratch_dir=str(tmp_path / f"scr_{engine_name}"),
    )
    assert got.provenance.exact is True
    assert got.num_nodes == ref.num_nodes
    assert got.num_edges == ref.num_edges
    assert got.dag_depth == ref.dag_depth
    assert got.critical_path_ns == pytest.approx(
        ref.critical_path_ns, rel=1e-12
    )
    assert got.mean_slack_ns == pytest.approx(ref.mean_slack_ns, rel=1e-12)
    assert got.zero_slack_ratio == pytest.approx(
        ref.zero_slack_ratio, rel=1e-12
    )
