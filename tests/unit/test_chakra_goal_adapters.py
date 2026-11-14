"""Tests for the Chakra execution-trace and GOAL schedule adapters.

Covers multi-rank and host Chakra-ET comparisons, exact parity between GOAL
text and its compiled binary form, Chakra-to-GOAL equivalence checks through
the SDK and the CLI, and parsing of storage and data-op nodes.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from typer import testing

import cta
from cta.cli import app as cli_app
from cta.core import models
from cta.synth import chakra
from cta.synth import goal

cli_runner = testing.CliRunner()


def _summary_by_metric(
    report: models.ComparisonReport, aspect_name: str
) -> dict[str, dict[str, Any]]:
    """Returns the summary rows of one aspect, keyed by metric name.

    Args:
        report: The comparison report to read.
        aspect_name: The aspect whose summary table is read.

    Returns:
        Each summary row as a dictionary, keyed by its `metric` value.
    """
    summary = report.to_polars(aspect_name)
    return {row["metric"]: row for row in summary.iter_rows(named=True)}


def test_chakra_et_multirank_comparison(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / ".cache"
    balanced_dir = tmp_path / "chakra_balanced"
    straggler_dir = tmp_path / "chakra_straggler"

    chakra.write_chakra_et_run(
        balanced_dir,
        world_size=4,
        compute_us=1000,
        allreduce_bytes=400_000,
        steps=2,
    )
    chakra.write_chakra_et_run(
        straggler_dir,
        world_size=4,
        compute_us=1000,
        allreduce_bytes=1_600_000,
        steps=2,
        straggler_rank=3,
        straggler_factor=2.0,
    )

    report = cta.compare(balanced_dir, straggler_dir, cache_dir=cache_dir)
    assert set(report.aspects) == {
        "dag_critical_path",
        "comm_collectives",
        "rank_skew",
        "traffic_matrix",
    }

    critical_path = _summary_by_metric(report, "dag_critical_path")
    assert critical_path["critical_path_us"]["delta_pct"] > 50.0
    assert critical_path["dag_depth"]["baseline"] == pytest.approx(4.0)

    comm = _summary_by_metric(report, "comm_collectives")
    assert comm["total_comm_mb"]["delta_pct"] == pytest.approx(300.0)

    skew = _summary_by_metric(report, "rank_skew")
    assert skew["compute_skew_ratio"]["baseline"] == pytest.approx(1.0)
    assert skew["compute_skew_ratio"]["candidate"] > 1.5


def test_chakra_host_et_comparison(tmp_path: pathlib.Path) -> None:
    slow_host_et_path = chakra.write_chakra_host_et(
        tmp_path / "chakra_host_a.json", compute_us=600.0, comm_bytes=100_000
    )
    fast_host_et_path = chakra.write_chakra_host_et(
        tmp_path / "chakra_host_b.json", compute_us=300.0, comm_bytes=100_000
    )
    report = cta.compare(
        slow_host_et_path, fast_host_et_path, cache_dir=tmp_path / ".cache"
    )
    critical_path = _summary_by_metric(report, "dag_critical_path")
    assert critical_path["cp_compute_us"]["delta_pct"] == pytest.approx(-50.0)


def test_goal_text_and_mmap_bin_exact_parity_and_algo_diff(
    tmp_path: pathlib.Path,
) -> None:
    cache_dir = tmp_path / ".cache"
    ring_goal = goal.write_goal_allreduce(
        tmp_path / "ring.goal",
        world_size=4,
        compute_ns=500_000,
        allreduce_bytes=400_000,
        algorithm="ring",
    )
    ring_bin = goal.compile_goal_to_bin(ring_goal, tmp_path / "ring.bin")
    direct_goal = goal.write_goal_allreduce(
        tmp_path / "direct.goal",
        world_size=4,
        compute_ns=500_000,
        allreduce_bytes=400_000,
        algorithm="direct",
    )

    # 1. Exact parity between the .goal text and the SPCL .bin binary schedule.
    parity = cta.compare(ring_goal, ring_bin, cache_dir=cache_dir)
    for aspect_name in (
        "dag_critical_path",
        "comm_collectives",
        "rank_skew",
        "traffic_matrix",
    ):
        summary = parity.to_polars(aspect_name)
        for delta_pct in summary["delta_pct"].to_list():
            assert abs(float(delta_pct)) < 1e-9

    # 2. Ring vs direct all-reduce.
    diff = cta.compare(ring_bin, direct_goal, cache_dir=cache_dir)
    traffic = _summary_by_metric(diff, "traffic_matrix")
    # Ring on 4 ranks has 4 directed links; a direct all-to-all has
    # 4 * 3 = 12.
    assert traffic["active_rank_links"]["baseline"] == pytest.approx(4.0)
    assert traffic["active_rank_links"]["candidate"] == pytest.approx(12.0)


def test_chakra_to_goal_equivalence_and_cli(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / ".cache"
    et_dir = tmp_path / "et_run"
    chakra.write_chakra_et_run(
        et_dir, world_size=4, compute_us=1000, allreduce_bytes=400_000, steps=2
    )

    good_goal = goal.write_goal_allreduce(
        tmp_path / "good.goal",
        world_size=4,
        compute_ns=1_000_000,
        allreduce_bytes=400_000,
        steps=2,
        algorithm="ring",
    )
    good_bin = goal.compile_goal_to_bin(good_goal, tmp_path / "good.bin")
    bad_goal = goal.write_goal_allreduce(
        tmp_path / "bad.goal",
        world_size=4,
        compute_ns=1_000_000,
        allreduce_bytes=200_000,
        steps=2,
        algorithm="ring",
    )

    matching = cta.verify_chakra_goal_equivalence(
        et_dir, good_bin, collective_algorithm="ring", cache_dir=cache_dir
    )
    assert matching.equivalent is True
    assert matching.comm_volume_rel_error == pytest.approx(0.0)

    mismatching = cta.verify_chakra_goal_equivalence(
        et_dir, bad_goal, collective_algorithm="ring", cache_dir=cache_dir
    )
    assert mismatching.equivalent is False
    assert mismatching.comm_volume_rel_error == pytest.approx(0.5)

    ok_result = cli_runner.invoke(
        cli_app.app,
        [
            "verify-equivalence",
            str(et_dir),
            str(good_goal),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert ok_result.exit_code == 0

    fail_result = cli_runner.invoke(
        cli_app.app,
        [
            "verify-equivalence",
            str(et_dir),
            str(bad_goal),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert fail_result.exit_code == 2


def test_chakra_storage_and_data_op_nodes(tmp_path: pathlib.Path) -> None:
    from cta.adapters.chakra import reader
    from cta.adapters.chakra import schema

    file_read = schema.NodeMsg(
        id=1, name="pread", type=schema.STORAGE_NODE, duration_micros=5
    )
    file_read.storage.fs.op = file_read.storage.fs.FS_OP_PREAD
    file_read.storage.fs.size_bytes = 4096
    block_io = schema.NodeMsg(
        id=2, name="bio", type=schema.STORAGE_NODE, duration_micros=1
    )
    block_io.storage.block.nr_sectors = 8
    checkpoint = schema.NodeMsg(
        id=3,
        name="ckpt",
        type=schema.DATA_OP_NODE,
        duration_micros=7,
        data_deps=[1],
    )
    metadata = schema.GlobalMetadataMsg(version="0.0.4")
    encoded = bytearray()
    for message in (metadata, file_read, block_io, checkpoint):
        payload = message.SerializeToString()
        encoded += schema.encode_varint(len(payload)) + payload
    et_path = tmp_path / "t.0.et"
    et_path.write_bytes(bytes(encoded))

    nodes, edges = reader._parse_single_et((0, et_path))
    nodes_by_name = {node["name"]: node for node in nodes}
    pread = nodes_by_name["pread"]
    assert (pread["op_class"], pread["size_bytes"]) == ("storage", 4096)
    assert nodes_by_name["bio"]["size_bytes"] == 8 * 512
    assert nodes_by_name["ckpt"]["op_class"] == "data_op"
    assert len(edges) == 1
