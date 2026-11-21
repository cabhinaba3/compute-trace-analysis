"""End-to-end integration tests for Phase 6 streaming dispatch & CLI."""

from __future__ import annotations

import json
import pathlib

import polars as pl
import pytest
from typer import testing

import cta
from cta import models
from cta.aspects.distributed import equivalence
from cta.cli import app as cli_app
from cta.ingest import cache
from cta.synth import chakra as synth_chakra
from cta.synth import goal as synth_goal
from cta.synth import training

cli_runner = testing.CliRunner()


def test_analyze_streaming_exact_matches_in_memory_and_bypasses_parquet(
    tmp_path: pathlib.Path,
) -> None:
    """Streaming exact mode matches in-memory metrics and skips Parquet."""
    spec = training.TrainingSpec(
        dp=2, pp=2, tp=2, microbatches=2, layers_per_stage=2, steps=2
    )
    goal_bin = tmp_path / "train.bin"
    training.write_goal_bin(goal_bin, spec)

    mem_cache = tmp_path / "mem_cache"
    str_cache = tmp_path / "str_cache"

    ref = cta.analyze(
        goal_bin,
        aspects=(
            "critical_path",
            "rank_skew",
            "traffic_matrix",
            "collectives",
            "whatif",
        ),
        cache_dir=mem_cache,
        streaming="off",
        label="run",
    )
    got = cta.analyze(
        goal_bin,
        aspects=(
            "critical_path",
            "rank_skew",
            "traffic_matrix",
            "collectives",
            "whatif",
        ),
        cache_dir=str_cache,
        streaming="exact",
        extract_critical_path_ops=True,
        label="run",
    )

    # Full nodes/edges Parquet was bypassed; .cta_graph cache was written!
    assert list(str_cache.rglob("nodes.parquet")) == []
    assert list(str_cache.rglob("edges.parquet")) == []
    assert len(list(str_cache.rglob("*.cta_graph"))) == 1

    # Summary metrics match bit-for-bit across every aspect!
    for aspect_name in (
        "critical_path",
        "rank_skew",
        "traffic_matrix",
        "collectives",
        "whatif",
    ):
        ref_tbl = ref.aspect(aspect_name).summary
        got_tbl = got.aspect(aspect_name).summary
        assert got_tbl["metric"].to_list() == ref_tbl["metric"].to_list()
        assert got_tbl["run"].to_list() == pytest.approx(
            ref_tbl["run"].to_list(), rel=1e-9
        )

    # Honest provenance and exact CP op chain are surfaced in details!
    cp_details = got.aspect("critical_path").details
    assert "provenance" in cp_details
    prov = cp_details["provenance"].row(0, named=True)
    assert prov["engine"] == "gridgraph"
    assert prov["exact"] is True
    assert "critical_path_ops" in cp_details
    assert cp_details["critical_path_ops"].height > 0

    # Lazy Parquet materialization works on demand if raw tables are needed.
    store = cache.CacheStore(str_cache)
    art = models.TraceArtifact(
        format="goal", paths=(goal_bin,), world_size=spec.ranks
    )
    pq_tables = store.ensure_parquet_tables(art)
    assert pq_tables["nodes"].exists()
    assert pq_tables["edges"].exists()


def test_auto_dispatch_parquet_bypass_and_scalasca_delays_in_report(
    tmp_path: pathlib.Path,
) -> None:
    """streaming='auto' bypasses Parquet and surfaces Scalasca delays."""
    et_dir = tmp_path / "chakra_straggler"
    synth_chakra.write_run(
        et_dir,
        world_size=4,
        compute_us=1000,
        allreduce_bytes=128_000,
        steps=2,
        straggler_rank=3,
        straggler_factor=2.5,
    )
    str_cache = tmp_path / "auto_cache"
    report = cta.analyze(
        et_dir,
        aspects=("critical_path", "rank_skew"),
        cache_dir=str_cache,
        streaming="auto",
        parquet_bypass_bytes=1,  # force bypass on small synthetic trace
        collective_sync="barrier",
        delay_attribution=True,
        label="straggler",
    )
    assert list(str_cache.rglob("nodes.parquet")) == []
    cp_details = report.aspect("critical_path").details
    prov = cp_details["provenance"].row(0, named=True)
    assert prov["exact"] is True
    assert prov["collective_sync"] == "barrier"
    assert prov["delay_timing_basis"] == "measured"
    delays = cp_details["root_cause_delays"]
    assert delays.height > 0
    assert int(delays.row(0, named=True)["rank"]) == 3


def test_streaming_aggregate_skips_dag_aspects_and_validates_explicit(
    tmp_path: pathlib.Path,
) -> None:
    """streaming='aggregate' runs 1-pass aspects and rejects DAG aspects."""
    path = tmp_path / "allreduce.goal"
    synth_goal.write_allreduce(path, world_size=4)

    report = cta.analyze(
        path,
        cache_dir=tmp_path / "c1",
        streaming="aggregate",
        quantile_sketch="kll",
    )
    assert "critical_path" not in report.aspects
    assert {"rank_skew", "traffic_matrix", "collectives"} <= set(report.aspects)
    assert list((tmp_path / "c1").rglob("nodes.parquet")) == []

    with pytest.raises(ValueError, match="streaming='aggregate'"):
        cta.analyze(
            path,
            aspects=("critical_path",),
            cache_dir=tmp_path / "c2",
            streaming="aggregate",
        )


def test_streaming_equivalence_matches_parquet_oracle(
    tmp_path: pathlib.Path,
) -> None:
    """verify_chakra_goal_equivalence streaming matches Parquet bit-for-bit."""
    et_dir = tmp_path / "et"
    synth_chakra.write_run(
        et_dir, world_size=4, compute_us=1000, allreduce_bytes=400_000, steps=2
    )
    goal_path = tmp_path / "sched.goal"
    synth_goal.write_allreduce(
        goal_path,
        world_size=4,
        compute_ns=1_000_000,
        allreduce_bytes=400_000,
        steps=2,
    )

    ref = equivalence.verify_chakra_goal_equivalence(
        et_dir,
        goal_path,
        cache_dir=tmp_path / "pq_cache",
        streaming_mode="off",
    )
    got = equivalence.verify_chakra_goal_equivalence(
        et_dir,
        goal_path,
        cache_dir=tmp_path / "str_cache",
        streaming_mode="on",
    )
    assert list((tmp_path / "str_cache").rglob("nodes.parquet")) == []
    assert got.equivalent == ref.equivalent is True
    assert got.world_size_chakra == ref.world_size_chakra
    assert got.world_size_goal == ref.world_size_goal
    assert got.compute_rel_error == pytest.approx(ref.compute_rel_error)
    assert got.comm_volume_rel_error == pytest.approx(ref.comm_volume_rel_error)


def test_cli_streaming_flags_stdin_compare_whatif_and_verify(
    tmp_path: pathlib.Path,
) -> None:
    """Exercises CLI streaming flags across analyze, compare, whatif, verify."""
    goal_path = tmp_path / "run.goal"
    synth_goal.write_allreduce(goal_path, world_size=4)
    goal_text = goal_path.read_text(encoding="utf-8")

    # 1. Stdin pipe with `cta analyze - --streaming aggregate`.
    stdin_res = cli_runner.invoke(
        cli_app.app,
        ["analyze", "-", "--streaming", "aggregate"],
        input=goal_text,
    )
    assert stdin_res.exit_code == 0, stdin_res.output
    stdin_doc = json.loads(stdin_res.stdout)
    assert stdin_doc["world_size"] == 4
    assert stdin_doc["num_nodes"] > 0

    # 2. `cta analyze --streaming exact --critical-path-ops --format json`.
    ana_res = cli_runner.invoke(
        cli_app.app,
        [
            "analyze",
            str(goal_path),
            "--streaming",
            "exact",
            "--engine",
            "gridgraph",
            "--critical-path-ops",
            "--cache-dir",
            str(tmp_path / "c_ana"),
            "--format",
            "json",
        ],
    )
    assert ana_res.exit_code == 0, ana_res.output
    assert list((tmp_path / "c_ana").rglob("nodes.parquet")) == []

    # 3. `cta compare --streaming exact --engine graphchi`.
    cmp_res = cli_runner.invoke(
        cli_app.app,
        [
            "compare",
            str(goal_path),
            str(goal_path),
            "--aspects",
            "critical_path,rank_skew",
            "--streaming",
            "exact",
            "--engine",
            "graphchi",
            "--cache-dir",
            str(tmp_path / "c_cmp"),
            "--format",
            "json",
        ],
    )
    assert cmp_res.exit_code == 0, cmp_res.output

    # 4. `cta whatif --streaming on --format json`.
    wi_res = cli_runner.invoke(
        cli_app.app,
        [
            "whatif",
            str(goal_path),
            "--streaming",
            "on",
            "--speed",
            "comm:2x",
            "--cache-dir",
            str(tmp_path / "c_wi"),
            "--format",
            "json",
        ],
    )
    assert wi_res.exit_code == 0, wi_res.output
    wi_doc = json.loads(wi_res.stdout)
    assert wi_doc["streaming"] is True
    assert wi_doc["vertices"] > 0
    assert len(wi_doc["scenarios"]) == 2
    assert list((tmp_path / "c_wi").rglob("nodes.parquet")) == []

    # 5. `cta verify-equivalence --streaming on`.
    et_dir = tmp_path / "et_cli"
    synth_chakra.write_run(
        et_dir, world_size=4, compute_us=1000, allreduce_bytes=400_000, steps=2
    )
    ver_res = cli_runner.invoke(
        cli_app.app,
        [
            "verify-equivalence",
            str(et_dir),
            str(goal_path),
            "--streaming",
            "on",
            "--cache-dir",
            str(tmp_path / "c_ver"),
        ],
    )
    assert ver_res.exit_code == 0, ver_res.output
    assert list((tmp_path / "c_ver").rglob("nodes.parquet")) == []


def test_frontier_windowed_provenance_reported_honestly(
    tmp_path: pathlib.Path,
) -> None:
    """Windowed mode certifies barrier slack and reports provenance."""
    et_dir = tmp_path / "et_win"
    synth_chakra.write_run(
        et_dir, world_size=4, compute_us=500, allreduce_bytes=65_536, steps=2
    )
    report = cta.analyze(
        et_dir,
        aspects=("critical_path",),
        cache_dir=tmp_path / "c_win",
        streaming="windowed",
        window_size=4096,
    )
    prov = (
        report.aspect("critical_path").details["provenance"].row(0, named=True)
    )
    assert prov["engine"] == "frontier"
    assert prov["collective_sync"] == "barrier"
    assert prov["slack_scope"] in ("global_exact", "window_local")
    assert isinstance(pl.DataFrame([prov]), pl.DataFrame)
