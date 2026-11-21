"""Unit tests for Phase 5 Engine D (streaming what-if & interconnect reprice).

Verifies bit-exact differential parity against CTA's in-memory what-if solver
(`DagPlan`, `run_scenarios`, `sweep`, `reprice`) across GOAL and Chakra traces,
all selector types (including `median`), all sweep groupings (`family`, `class`,
`rank`, `name`), interconnect repricing + warnings, multi-threaded determinism,
high fan-in (>= 256 predecessors exercising the 0xFF winner-log escape table),
and tight memory budgets forcing multi-block boundary spill.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from cta import _native as native
from cta import models
from cta.analysis import interconnect as ic
from cta.analysis import whatif
from cta.synth import chakra as synth_chakra
from cta.synth import chakra_proto
from cta.synth import training


def test_streaming_whatif_scenarios_match_in_memory_goal(
    tmp_path: pathlib.Path,
) -> None:
    """Scenarios, tie-breaking, and median rules match DagPlan bit-for-bit."""
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

    mem_model = whatif.load_model(goal_bin, cache_dir=tmp_path / "mem_cache")
    str_model = whatif.load_streaming_model(
        goal_bin,
        cache_dir=tmp_path / "str_cache",
        memory_budget_bytes=64 * 1024,
        scratch_dir=str(tmp_path / "scratch"),
    )

    scenarios = [
        whatif.parse_scenario("comm:2x"),
        whatif.parse_scenario("compute:1.5x;comm:remove"),
        whatif.parse_scenario("rank=0-3:2x;class=comm_send,comm_recv:0.5x"),
        whatif.parse_scenario("rank=*:median"),
        whatif.parse_scenario("all:remove"),
    ]
    ref = whatif.run_scenarios(mem_model, scenarios, threads=1)
    got_t1 = whatif.run_scenarios(
        str_model, scenarios, threads=1, memory_budget_bytes=64 * 1024
    )
    got_t4 = whatif.run_scenarios(
        str_model, scenarios, threads=4, memory_budget_bytes=512 * 1024
    )

    assert got_t1.table["scenario"].to_list() == ref.table["scenario"].to_list()
    assert got_t4.table["scenario"].to_list() == ref.table["scenario"].to_list()
    for col in (
        "makespan_ns",
        "cp_compute_ns",
        "cp_exposed_comm_ns",
        "cp_comm_bytes",
        "dag_depth",
    ):
        assert got_t1.table[col].to_list() == pytest.approx(
            ref.table[col].to_list(), rel=1e-12
        )
        assert got_t4.table[col].to_list() == got_t1.table[col].to_list()
    assert got_t1.critical_by_group.sum(axis=1) == pytest.approx(
        ref.critical_by_group.sum(axis=1), rel=1e-12
    )


@pytest.mark.parametrize("by", ["family", "class", "rank", "name"])
def test_streaming_sweep_matches_in_memory_chakra(
    tmp_path: pathlib.Path, by: str
) -> None:
    """Streaming sweep & ceiling table match in-memory sweep on Chakra .et."""
    et_dir = tmp_path / "chakra_et"
    synth_chakra.write_run(
        et_dir,
        world_size=4,
        compute_us=1000,
        allreduce_bytes=128_000,
        steps=3,
        straggler_rank=2,
        straggler_factor=1.8,
    )
    mem_model = whatif.load_model(et_dir, cache_dir=tmp_path / "mem_c")
    str_model = whatif.load_streaming_model(
        et_dir,
        cache_dir=tmp_path / "str_c",
        memory_budget_bytes=64 * 1024,
        scratch_dir=str(tmp_path / "scr"),
    )

    ref = whatif.sweep(mem_model, by=by, max_groups=6, factors=(1.5, 2.0))
    got = whatif.sweep(
        str_model,
        by=by,
        max_groups=6,
        factors=(1.5, 2.0),
        memory_budget_bytes=64 * 1024,
    )

    assert got.baseline_makespan_ns == pytest.approx(
        ref.baseline_makespan_ns, rel=1e-12
    )
    assert got.table["group"].to_list() == ref.table["group"].to_list()
    assert got.table["makespan_ns"].to_list() == pytest.approx(
        ref.table["makespan_ns"].to_list(), rel=1e-12
    )
    assert (
        got.table["bottleneck"].to_list() == ref.table["bottleneck"].to_list()
    )
    assert got.ceiling["group"].to_list() == ref.ceiling["group"].to_list()
    for col in (
        "time_share_pct",
        "critical_share_pct",
        "ceiling_speedup",
        "knee_factor",
    ):
        assert got.ceiling[col].to_list() == pytest.approx(
            ref.ceiling[col].to_list(), rel=1e-12
        )


def test_streaming_reprice_unrecorded_goal_and_recorded_chakra(
    tmp_path: pathlib.Path,
) -> None:
    """Streaming reprice matches in-memory reprice & warnings bit-for-bit."""
    goal_path = tmp_path / "t.goal"
    training.write_goal(goal_path, training.TrainingSpec(dp=4, tp=2, pp=2))
    mem_goal = whatif.load_model(goal_path, cache_dir=tmp_path / "mc")
    str_goal = whatif.load_streaming_model(
        goal_path, cache_dir=tmp_path / "sc", scratch_dir=str(tmp_path / "scr")
    )

    net = ic.parse(
        "gpus_per_node=4,intra=1us:300GB/s,inter=5us:25GB/s,algo=auto"
    )
    ref_g = whatif.run_scenarios(
        whatif.reprice(mem_goal, net), [whatif.parse_scenario("comm:2x")]
    )
    got_g = whatif.run_scenarios(
        whatif.reprice(str_goal, net), [whatif.parse_scenario("comm:2x")]
    )
    assert got_g.table["makespan_ns"].to_list() == pytest.approx(
        ref_g.table["makespan_ns"].to_list(), rel=1e-12
    )

    # Chakra trace with recorded collective durations: scales by new/base and
    # warns when base is omitted or collective group sizes default to world.
    et_dir = tmp_path / "rec_et"
    synth_chakra.write_run(
        et_dir, world_size=4, compute_us=800, allreduce_bytes=1 << 20, steps=2
    )
    mem_ch = whatif.load_model(et_dir, cache_dir=tmp_path / "mc2")
    str_ch = whatif.load_streaming_model(
        et_dir, cache_dir=tmp_path / "sc2", scratch_dir=str(tmp_path / "scr2")
    )

    with pytest.warns(RuntimeWarning, match="no baseline interconnect"):
        kept = whatif.reprice(str_ch, net)
    assert whatif.run_scenarios(kept, []).table["makespan_ns"][
        0
    ] == pytest.approx(
        whatif.run_scenarios(str_ch, []).table["makespan_ns"][0], rel=1e-12
    )

    base = ic.parse(
        "gpus_per_node=4,intra=1us:100GB/s,inter=4us:50GB/s,algo=ring,"
        "group.ALL_REDUCE=4"
    )
    target = ic.parse(
        "gpus_per_node=4,intra=1us:100GB/s,inter=10us:10GB/s,algo=tree,"
        "group.ALL_REDUCE=4"
    )
    ref_c = whatif.run_scenarios(
        whatif.reprice(mem_ch, target, base),
        [
            whatif.parse_scenario("type=ALL_REDUCE:2x"),
            whatif.parse_scenario("name~gemm:1.5x"),
        ],
    )
    got_c = whatif.run_scenarios(
        whatif.reprice(str_ch, target, base),
        [
            whatif.parse_scenario("type=ALL_REDUCE:2x"),
            whatif.parse_scenario("name~gemm:1.5x"),
        ],
    )
    assert got_c.table["makespan_ns"].to_list() == pytest.approx(
        ref_c.table["makespan_ns"].to_list(), rel=1e-12
    )
    assert got_c.table["cp_exposed_comm_ns"].to_list() == pytest.approx(
        ref_c.table["cp_exposed_comm_ns"].to_list(), rel=1e-12
    )


def test_streaming_whatif_high_fanin_escape_and_adversarial_cutwidth_spill(
    tmp_path: pathlib.Path,
) -> None:
    """Exercises 0xFF winner-log escape (deg > 300) and boundary disk spill."""
    # Construct a Chakra trace with 1800 nodes:
    # - Nodes 0..1499 are independent compute ops of varying durations.
    # - Node 1500 depends on ALL 1500 predecessors (deg = 1500 >> 255!), with
    #   the slowest predecessor at index 1400 (>= 255 -> triggers 0xFF escape!)
    #   and also creates max_cutwidth >= 1500 across topological blocks.
    # - Nodes 1501..1799 chain after 1500 and also reference earlier nodes.
    n_roots = 1500
    nodes = []
    for i in range(n_roots):
        dur = 500 if i == 1400 else 10 + (i % 50)
        nodes.append(
            chakra_proto.NodeMsg(
                id=i + 1,
                name=f"root_{i}",
                type=chakra_proto.COMP_NODE,
                duration_micros=dur,
            )
        )
    # Fan-in join node with 1500 predecessors!
    nodes.append(
        chakra_proto.NodeMsg(
            id=n_roots + 1,
            name="mega_join",
            type=chakra_proto.COMP_NODE,
            duration_micros=100,
            data_deps=list(range(1, n_roots + 1)),
        )
    )
    for j in range(1, 300):
        deps = [n_roots + j]
        if j % 5 == 0:
            deps.append((j * 3) % n_roots + 1)
        nodes.append(
            chakra_proto.NodeMsg(
                id=n_roots + 1 + j,
                name="tail_comm" if j % 4 == 0 else "tail_comp",
                type=(
                    chakra_proto.COMM_COLL_NODE
                    if j % 4 == 0
                    else chakra_proto.COMP_NODE
                ),
                duration_micros=25,
                data_deps=deps,
            )
        )
    et_path = tmp_path / "fanin.rank0.et"
    buf = bytearray()
    for msg in (chakra_proto.GlobalMetadataMsg(version="0.0.4"), *nodes):
        payload = msg.SerializeToString()
        buf += chakra_proto.encode_varint(len(payload)) + payload
    et_path.write_bytes(bytes(buf))

    mem_model = whatif.load_model(et_path, cache_dir=tmp_path / "mc")
    art = models.TraceArtifact(format="chakra", paths=(et_path,), world_size=1)
    cache_path = tmp_path / "fanin.cta_graph"
    # Build cache with tiny 32 KiB RAM budget -> creates multiple blocks!
    str_model = whatif.build_streaming_model_from_artifact(
        art,
        cache_path=cache_path,
        memory_budget_bytes=32 * 1024,
        scratch_dir=str(tmp_path / "scr"),
    )
    cache_hdr = native.GraphCache(str(cache_path))
    assert cache_hdr.num_blocks >= 2
    assert cache_hdr.max_cutwidth >= n_roots - 1

    # Direct native call with tight budget confirms boundary spill engages and
    # matches exact DagPlan answers bit-for-bit!
    groups = np.zeros(cache_hdr.num_total_vertices, dtype=np.uint32)
    scale = np.ones((2, 1), dtype=np.float64)
    scale[1, 0] = 0.5
    raw = native.solve_whatif_graph_cache(
        str(cache_path),
        "",
        scale,
        threads=1,
        memory_budget_bytes=64 * 1024,
        scratch_dir=str(tmp_path / "scr2"),
        inline_groups=groups,
    )
    assert raw["used_boundary_spill"] is True

    scenarios = [
        whatif.parse_scenario("name~root_1400:10x"),
        whatif.parse_scenario("name~tail_comm:remove"),
        whatif.parse_scenario("all:2x"),
    ]
    ref = whatif.run_scenarios(mem_model, scenarios, threads=1)
    got = whatif.run_scenarios(
        str_model, scenarios, threads=2, memory_budget_bytes=64 * 1024
    )
    for col in (
        "makespan_ns",
        "cp_compute_ns",
        "cp_exposed_comm_ns",
        "dag_depth",
    ):
        assert got.table[col].to_list() == pytest.approx(
            ref.table[col].to_list(), rel=1e-12
        )
