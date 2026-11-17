"""Tests of the 3D-parallel training graph generators (GOAL and Chakra)."""

from __future__ import annotations

import pathlib
import warnings

import polars as pl
import pytest

from cta import plugins
from cta.adapters.goal import reader as goal_reader
from cta.analysis import dag
from cta.ingest import cache
from cta.ingest import discovery
from cta.synth import goal
from cta.synth import training

_SMALL = training.TrainingSpec(
    dp=2,
    pp=2,
    tp=2,
    microbatches=3,
    layers_per_stage=2,
    steps=2,
    moe_every=2,
    ep_size=2,
    dp_buckets=2,
)


def _solve(nodes: pl.DataFrame, edges: pl.DataFrame) -> dag.DagStats:
    """Solves the graph and fails on the cycle warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return dag.solve_distributed_dag(nodes, edges)


def _read_chakra(
    directory: pathlib.Path, cache_dir: pathlib.Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    run = discovery.resolve_bundle(directory)
    tables = cache.CacheStore(cache_dir).get_or_ingest(
        run.artifacts["chakra_et"], plugins.ADAPTERS["chakra_et"]
    )
    return pl.read_parquet(tables["nodes"]), pl.read_parquet(tables["edges"])


def _assert_p2p_matched(nodes: pl.DataFrame) -> None:
    """Every send has a receive with the same (src, dst, tag) and vice versa."""
    channel = ["src_rank", "dst_rank", "message_tag"]
    sends = (
        nodes.filter(pl.col("op_class") == "comm_send")
        .group_by(channel)
        .len()
        .rename({"len": "sends"})
    )
    recvs = (
        nodes.filter(pl.col("op_class") == "comm_recv")
        .group_by(channel)
        .len()
        .rename({"len": "recvs"})
    )
    joined = sends.join(recvs, on=channel, how="full", coalesce=True)
    assert joined.height > 0
    assert joined["sends"].null_count() == 0
    assert joined["recvs"].null_count() == 0
    assert (joined["sends"] == joined["recvs"]).all()


def test_goal_binary_equals_compiled_text(tmp_path: pathlib.Path) -> None:
    text = tmp_path / "t.goal"
    node_count = training.write_goal_training(text, _SMALL)
    binary = tmp_path / "t.bin"
    assert training.write_goal_training_bin(binary, _SMALL) == node_count
    compiled = goal.compile_goal_to_bin(text, tmp_path / "compiled.bin")
    assert binary.read_bytes() == compiled.read_bytes()


def test_goal_graph_is_acyclic_and_every_message_is_matched(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "t.goal"
    node_count = training.write_goal_training(path, _SMALL)
    nodes, edges = goal_reader.parse_goal_text(path)
    assert nodes.height == node_count
    assert nodes["rank"].n_unique() == _SMALL.ranks
    _assert_p2p_matched(nodes)
    stats = _solve(nodes, edges)
    assert stats.num_nodes == node_count
    assert stats.num_edges > edges.height  # plus the send -> recv links
    assert stats.critical_path_ns > 0


def test_single_rank_has_only_compute_nodes(tmp_path: pathlib.Path) -> None:
    spec = training.TrainingSpec(
        dp=1, pp=1, tp=1, microbatches=3, layers_per_stage=4, steps=2
    )
    path = tmp_path / "one.goal"
    node_count = training.write_goal_training(path, spec)
    # Per step: forward and backward calcs for every layer and microbatch, and
    # one optimizer node.
    assert node_count == spec.steps * (2 * 3 * 4 + 1)
    nodes, _ = goal_reader.parse_goal_text(path)
    assert set(nodes["op_class"].unique()) == {"compute"}


def test_goal_moe_adds_all_to_all_traffic(tmp_path: pathlib.Path) -> None:
    dense = training.write_goal_training(
        tmp_path / "dense.goal",
        training.TrainingSpec(dp=2, pp=1, tp=1, moe_every=0),
    )
    moe = training.write_goal_training(
        tmp_path / "moe.goal",
        training.TrainingSpec(dp=2, pp=1, tp=1, moe_every=2, ep_size=2),
    )
    assert moe > dense


def test_chakra_collectives_are_single_nodes_and_pipeline_is_p2p(
    tmp_path: pathlib.Path,
) -> None:
    node_count = training.write_chakra_training(
        tmp_path / "et", _SMALL, threads=2
    )
    nodes, edges = _read_chakra(tmp_path / "et", tmp_path / "cache")
    assert nodes.height == node_count
    assert nodes["rank"].n_unique() == _SMALL.ranks
    _assert_p2p_matched(nodes)
    collectives = nodes.filter(pl.col("op_class") == "comm_coll")
    kinds = set(collectives["comm_type"].unique())
    assert {"ALL_REDUCE", "ALL_TO_ALL"} <= kinds
    # Tensor-parallel all-reduces: one per layer pass, forward and backward, on
    # every rank; the data-parallel buckets come on top.
    layer_passes = _SMALL.steps * _SMALL.microbatches * _SMALL.layers_per_stage
    per_rank = 2 * layer_passes + _SMALL.steps * _SMALL.dp_buckets
    all_reduces = collectives.filter(pl.col("comm_type") == "ALL_REDUCE")
    assert all_reduces.height == _SMALL.ranks * per_rank
    stats = _solve(nodes, edges)
    assert stats.num_nodes == node_count


def test_chakra_output_is_independent_of_the_thread_count(
    tmp_path: pathlib.Path,
) -> None:
    training.write_chakra_training(tmp_path / "a", _SMALL, threads=1)
    training.write_chakra_training(tmp_path / "b", _SMALL, threads=4)
    for rank in range(_SMALL.ranks):
        name = f"trace.rank{rank}.et"
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()


def test_a_hundred_million_nodes_fit_in_the_tag_space() -> None:
    # About 100M GOAL nodes (or Chakra nodes at 30 steps per rank-thousand).
    spec = training.TrainingSpec(dp=8, pp=8, tp=8, steps=1250, microbatches=8)
    spec.validate()
    assert spec.ranks == 512


def test_goal_output_is_deterministic(tmp_path: pathlib.Path) -> None:
    training.write_goal_training(tmp_path / "a.goal", _SMALL)
    training.write_goal_training(tmp_path / "b.goal", _SMALL)
    assert (tmp_path / "a.goal").read_bytes() == (
        tmp_path / "b.goal"
    ).read_bytes()


@pytest.mark.parametrize(
    "spec",
    [
        training.TrainingSpec(dp=0),
        training.TrainingSpec(dp=3, ep_size=2),
        training.TrainingSpec(layers_per_stage=2, dp_buckets=3),
        training.TrainingSpec(steps=10_000_000),
    ],
)
def test_invalid_specs_are_rejected(
    tmp_path: pathlib.Path, spec: training.TrainingSpec
) -> None:
    with pytest.raises(ValueError, match="."):
        training.write_goal_training(tmp_path / "x.goal", spec)
