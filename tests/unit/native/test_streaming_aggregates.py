"""Unit tests for Phase 2 pluggable quantile sketches and Engine A aggregates.

Verifies exact parity of Engine A against CTA's existing distributed aspects
(rank_skew, collectives, traffic_matrix, equivalence), formal error bounds for
DDSketch (VLDB '19), KLL (FOCS '16), and Greenwald-Khanna (SIGMOD '01) on
heavy-tailed and high-multiplicity streams, and unseekable FIFO pipe streaming.
"""

from __future__ import annotations

import os
import pathlib
import threading

import numpy as np
import pytest

import cta
from cta import _native as native
from cta import models
from cta.analysis import streaming
from cta.aspects.distributed import equivalence
from cta.synth import chakra as synth_chakra
from cta.synth import goal as synth_goal
from cta.synth import training


def test_hybrid_quantile_exact_matches_numpy() -> None:
    rng = np.random.default_rng(42)
    values = rng.integers(0, 500, size=10_000, dtype=np.uint64)
    hq = native.HybridQuantile("ddsketch", exact_cap=65536, accuracy=0.001)
    hq.add_array(values)
    assert hq.is_exact is True
    assert hq.sketch_name == "exact"
    assert hq.error_bound == 0.0
    for q in (0.0, 0.1, 0.5, 0.95, 0.99, 1.0):
        expected = float(np.percentile(values, q * 100.0))
        assert hq.quantile(q) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_ddsketch_relative_error_bound_on_heavy_tail() -> None:
    """DDSketch guarantees |q_hat - q| <= alpha * q across 9 orders of mag."""
    rng = np.random.default_rng(7)
    # Log-uniform values spanning 8 B to 4 GB plus repeated cliff values.
    exps = rng.uniform(np.log(8.0), np.log(4e9), size=120_000)
    values = np.exp(exps).astype(np.uint64)
    alpha = 0.002
    hq = native.HybridQuantile("ddsketch", exact_cap=0, accuracy=alpha)
    hq.add_array(values)
    assert hq.is_exact is False
    assert hq.sketch_name == "ddsketch"
    assert hq.error_bound == pytest.approx(alpha)
    for q in (0.1, 0.5, 0.9, 0.95, 0.99):
        true_q = float(np.percentile(values, q * 100.0))
        est_q = hq.quantile(q)
        rel_err = abs(est_q - true_q) / max(true_q, 1.0)
        assert rel_err <= alpha * 1.5, (q, true_q, est_q, rel_err)


@pytest.mark.parametrize("sketch_kind", ["kll", "gk"])
def test_rank_error_sketches_stay_within_eps_rank(sketch_kind: str) -> None:
    """KLL (FOCS '16) and GK (SIGMOD '01) satisfy rank error <= eps."""
    rng = np.random.default_rng(13)
    values = rng.integers(1, 500_000, size=60_000, dtype=np.uint64)
    # Add a massive repeated value to verify multiplicity handling in GK & KLL.
    repeated = np.full(15_000, 123_456, dtype=np.uint64)
    data = np.concatenate([values, repeated])
    rng.shuffle(data)
    sorted_data = np.sort(data)

    eps = 0.01
    hq = native.HybridQuantile(sketch_kind, exact_cap=0, accuracy=eps)
    hq.add_array(data)
    assert hq.is_exact is False
    assert hq.sketch_name == sketch_kind
    assert hq.error_bound == pytest.approx(eps)

    n = len(sorted_data)
    for q in (0.1, 0.25, 0.5, 0.75, 0.95):
        est = hq.quantile(q)
        lo_rank = int(np.searchsorted(sorted_data, est, side="left"))
        hi_rank = int(np.searchsorted(sorted_data, est, side="right"))
        target_rank = q * (n - 1)
        # Distance from target_rank to the value's rank interval [lo, hi].
        rank_err = (
            0.0
            if lo_rank <= target_rank <= hi_rank
            else min(abs(target_rank - lo_rank), abs(target_rank - hi_rank)) / n
        )
        assert rank_err <= eps * 1.25, (sketch_kind, q, rank_err)


def test_hybrid_spills_honestly_when_cardinality_exceeds_cap() -> None:
    values = np.arange(1, 5001, dtype=np.uint64)
    hq = native.HybridQuantile("ddsketch", exact_cap=100, accuracy=0.005)
    hq.add_array(values)
    assert hq.is_exact is False
    assert hq.sketch_name == "ddsketch"
    assert abs(hq.quantile(0.5) - 2500.0) / 2500.0 <= 0.01


def _metric_map(report: models.Report, aspect: str) -> dict[str, float]:
    table = report.to_polars(aspect)
    col = "baseline" if "baseline" in table.columns else report.groups[0].label
    return {
        str(row["metric"]): float(row[col])
        for row in table.iter_rows(named=True)
    }


def test_online_aggregates_exact_parity_with_chakra_and_goal_aspects(
    tmp_path: pathlib.Path,
) -> None:
    cache_dir = tmp_path / ".cache"
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
    chakra_dir = tmp_path / "train_chakra"
    training.write_chakra(chakra_dir, spec)
    goal_bin = tmp_path / "train.bin"
    training.write_goal_bin(goal_bin, spec)
    goal_txt = tmp_path / "train.goal"
    training.write_goal(goal_txt, spec)

    for label, src_path, fmt, world_size in (
        ("chakra", chakra_dir, "chakra", spec.ranks),
        ("goal_bin", goal_bin, "goal", spec.ranks),
        ("goal_txt", goal_txt, "goal", spec.ranks),
    ):
        report = cta.analyze(
            src_path,
            aspects=("rank_skew", "collectives", "traffic_matrix"),
            cache_dir=cache_dir,
        )
        paths = (
            tuple(sorted(src_path.glob("*.et")))
            if src_path.is_dir()
            else (src_path,)
        )
        artifact = models.TraceArtifact(
            format=fmt, paths=paths, world_size=world_size
        )
        stream, inc_coll = streaming.open_stream_for_artifact(
            artifact, chunk_nodes=37
        )
        agg = streaming.run_streaming_aggregates(
            stream,
            include_collectives_in_traffic=inc_coll,
            min_world_size=world_size,
            chunk_nodes=37,
        )
        assert agg.provenance.engine == "online_agg"
        assert agg.provenance.exact is True
        assert agg.provenance.quantile_sketch == "exact"
        assert agg.provenance.slack_scope == "unavailable"

        skew = _metric_map(report, "rank_skew")
        assert agg.world_size == int(skew["world_size"]), label
        assert agg.max_rank_compute_us == pytest.approx(
            skew["max_rank_compute_us"], rel=1e-9
        )
        assert agg.mean_rank_compute_us == pytest.approx(
            skew["mean_rank_compute_us"], rel=1e-9
        )
        assert agg.compute_skew_ratio == pytest.approx(
            skew["compute_skew_ratio"], rel=1e-9
        )
        assert agg.max_rank_comm_mb == pytest.approx(
            skew["max_rank_comm_mb"], rel=1e-9
        )
        assert agg.comm_jains_fairness == pytest.approx(
            skew["comm_jains_fairness"], rel=1e-9
        )

        coll = _metric_map(report, "collectives")
        assert agg.total_comm_mb == pytest.approx(
            coll["total_comm_mb"], rel=1e-9
        )
        assert agg.comm_op_count == int(coll["comm_op_count"])
        assert agg.median_msg_kb == pytest.approx(
            coll["median_msg_kb"], rel=1e-9
        )
        assert agg.p95_msg_kb == pytest.approx(coll["p95_msg_kb"], rel=1e-9)
        assert agg.max_msg_kb == pytest.approx(coll["max_msg_kb"], rel=1e-9)

        traf = _metric_map(report, "traffic_matrix")
        assert agg.total_p2p_mb == pytest.approx(traf["total_p2p_mb"], rel=1e-9)
        assert agg.active_rank_links == int(traf["active_rank_links"])
        assert agg.mean_peer_degree == pytest.approx(
            traf["mean_peer_degree"], rel=1e-9
        )
        assert agg.hotspot_link_mb == pytest.approx(
            traf["hotspot_link_mb"], rel=1e-9
        )


def test_online_aggregates_equivalence_and_fifo_pipes(
    tmp_path: pathlib.Path,
) -> None:
    et_dir = tmp_path / "et_run"
    synth_chakra.write_run(
        et_dir, world_size=4, compute_us=1000, allreduce_bytes=400_000, steps=2
    )
    goal_path = synth_goal.write_allreduce(
        tmp_path / "ring.goal",
        world_size=4,
        compute_ns=1_000_000,
        allreduce_bytes=400_000,
        steps=2,
        algorithm="ring",
    )

    et_art = models.TraceArtifact(
        format="chakra", paths=tuple(sorted(et_dir.glob("*.et"))), world_size=4
    )
    et_stream, et_coll = streaming.open_stream_for_artifact(et_art)
    et_agg = streaming.run_streaming_aggregates(
        et_stream, include_collectives_in_traffic=et_coll, min_world_size=4
    )

    # Stream the GOAL text through an unseekable FIFO pipe directly to Engine A.
    fifo = tmp_path / "ring.fifo"
    os.mkfifo(fifo)
    raw = goal_path.read_bytes()

    def _feed() -> None:
        with fifo.open("wb") as fh:
            fh.write(raw)

    thread = threading.Thread(target=_feed)
    thread.start()
    try:
        pipe_stream, pipe_coll = streaming.open_stream_for_path(
            fifo, fmt="goal", chunk_nodes=7
        )
        assert pipe_stream.is_seekable is False
        goal_agg = streaming.run_streaming_aggregates(
            pipe_stream,
            include_collectives_in_traffic=pipe_coll,
            min_world_size=1,
        )
    finally:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert goal_agg.world_size == et_agg.world_size == 4
    assert goal_agg.total_compute_ns == et_agg.total_compute_ns
    assert goal_agg.send_count == goal_agg.recv_count

    # Verify Chakra -> GOAL decomposed send volume matches equivalence.py.
    expected_send_bytes = et_agg.total_send_bytes + sum(
        nbytes * equivalence._collective_send_factor(ctype, 4, "ring")
        for ctype, nbytes in et_agg.coll_bytes_by_type.items()
    )
    assert goal_agg.total_send_bytes == pytest.approx(
        expected_send_bytes, rel=1e-9
    )
