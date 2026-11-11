from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cta import compare, equiv
from cta.cli.app import app
from cta.synth.generators import (
    compile_goal_to_bin,
    write_chakra_et_run,
    write_chakra_host_et,
    write_goal_allreduce,
)

runner = CliRunner()


def test_chakra_et_multirank_comparison(tmp_path: Path) -> None:
    cache = tmp_path / ".cache"
    base_dir = tmp_path / "chakra_balanced"
    cand_dir = tmp_path / "chakra_straggler"

    write_chakra_et_run(base_dir, world_size=4, compute_us=1000, allreduce_bytes=400_000, steps=2)
    write_chakra_et_run(
        cand_dir,
        world_size=4,
        compute_us=1000,
        allreduce_bytes=1_600_000,
        steps=2,
        straggler_rank=3,
        straggler_factor=2.0,
    )

    rep = compare(base_dir, cand_dir, cache_dir=cache)
    assert set(rep.aspects) == {"dag_critical_path", "comm_collectives", "rank_skew", "traffic_matrix"}

    cp = rep.to_polars("dag_critical_path")
    cp_map = {r["metric"]: r for r in cp.iter_rows(named=True)}
    assert cp_map["critical_path_us"]["delta_pct"] > 50.0
    assert cp_map["dag_depth"]["baseline"] == pytest.approx(4.0)

    comm = rep.to_polars("comm_collectives")
    comm_map = {r["metric"]: r for r in comm.iter_rows(named=True)}
    assert comm_map["total_comm_mb"]["delta_pct"] == pytest.approx(300.0)

    skew = rep.to_polars("rank_skew")
    skew_map = {r["metric"]: r for r in skew.iter_rows(named=True)}
    assert skew_map["compute_skew_ratio"]["baseline"] == pytest.approx(1.0)
    assert skew_map["compute_skew_ratio"]["candidate"] > 1.5


def test_chakra_host_et_comparison(tmp_path: Path) -> None:
    a = write_chakra_host_et(tmp_path / "chakra_host_a.json", compute_us=600.0, comm_bytes=100_000)
    b = write_chakra_host_et(tmp_path / "chakra_host_b.json", compute_us=300.0, comm_bytes=100_000)
    rep = compare(a, b, cache_dir=tmp_path / ".cache")
    cp = {r["metric"]: r for r in rep.to_polars("dag_critical_path").iter_rows(named=True)}
    assert cp["cp_compute_us"]["delta_pct"] == pytest.approx(-50.0)


def test_goal_text_and_mmap_bin_exact_parity_and_algo_diff(tmp_path: Path) -> None:
    cache = tmp_path / ".cache"
    ring_goal = write_goal_allreduce(tmp_path / "ring.goal", world_size=4, compute_ns=500_000, allreduce_bytes=400_000, algo="ring")
    ring_bin = compile_goal_to_bin(ring_goal, tmp_path / "ring.bin")
    direct_goal = write_goal_allreduce(tmp_path / "direct.goal", world_size=4, compute_ns=500_000, allreduce_bytes=400_000, algo="direct")

    # 1. Exact parity between .goal text and SPCL .bin binary schedule
    parity = compare(ring_goal, ring_bin, cache_dir=cache)
    for asp_name in ("dag_critical_path", "comm_collectives", "rank_skew", "traffic_matrix"):
        df = parity.to_polars(asp_name)
        for d in df["delta_pct"].to_list():
            assert abs(float(d)) < 1e-9

    # 2. Ring vs Direct AllReduce comparison
    diff = compare(ring_bin, direct_goal, cache_dir=cache)
    tmat = {r["metric"]: r for r in diff.to_polars("traffic_matrix").iter_rows(named=True)}
    # Ring on 4 ranks has 4 directed links; Direct all-to-all has 4 * 3 = 12 directed links
    assert tmat["active_rank_links"]["baseline"] == pytest.approx(4.0)
    assert tmat["active_rank_links"]["candidate"] == pytest.approx(12.0)


def test_chakra_to_goal_equivalence_and_cli(tmp_path: Path) -> None:
    cache = tmp_path / ".cache"
    et_dir = tmp_path / "et_run"
    write_chakra_et_run(et_dir, world_size=4, compute_us=1000, allreduce_bytes=400_000, steps=2)

    good_goal = write_goal_allreduce(tmp_path / "good.goal", world_size=4, compute_ns=1_000_000, allreduce_bytes=400_000, steps=2, algo="ring")
    good_bin = compile_goal_to_bin(good_goal, tmp_path / "good.bin")
    bad_goal = write_goal_allreduce(tmp_path / "bad.goal", world_size=4, compute_ns=1_000_000, allreduce_bytes=200_000, steps=2, algo="ring")

    v_ok = equiv(et_dir, good_bin, collective_algo="ring", cache_dir=cache)
    assert v_ok.equivalent is True
    assert v_ok.comm_volume_rel_error == pytest.approx(0.0)

    v_bad = equiv(et_dir, bad_goal, collective_algo="ring", cache_dir=cache)
    assert v_bad.equivalent is False
    assert v_bad.comm_volume_rel_error == pytest.approx(0.5)

    res_ok = runner.invoke(app, ["equiv", str(et_dir), str(good_goal), "--cache-dir", str(cache)])
    assert res_ok.exit_code == 0

    res_fail = runner.invoke(app, ["equiv", str(et_dir), str(bad_goal), "--cache-dir", str(cache)])
    assert res_fail.exit_code == 2
