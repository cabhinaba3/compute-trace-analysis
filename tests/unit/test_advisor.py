"""Tests of the optimization advisor (`cta.advisor`).

The timeline rules are checked on hand-made kernel tables whose gaps, overlap
and copy times are known exactly, then on generated Nsight Systems and GOAL
traces through `cta.analyze`, where the ceilings must agree with what the
`overlap` aspect reports for the same run.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
import pytest

import cta
from cta import advisor
from cta import models
from cta.synth import nsys as synth_nsys
from cta.synth import training

US = 1000


def _kernels(rows: list[tuple]) -> pl.DataFrame:
    """Builds a kernels table from (category, name, start_us, end_us, bytes)."""
    return pl.DataFrame(
        {
            "rank": [0] * len(rows),
            "device": [0] * len(rows),
            "category": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "start_ns": [r[2] * US for r in rows],
            "end_ns": [r[3] * US for r in rows],
            "duration_ns": [(r[3] - r[2]) * US for r in rows],
            "bytes": [r[4] if len(r) > 4 else 0 for r in rows],
        }
    )


def _steps(bounds_us: list[tuple[int, int]]) -> pl.DataFrame:
    """Builds a one-rank steps table from (start_us, end_us) pairs."""
    return pl.DataFrame(
        {
            "rank": [0] * len(bounds_us),
            "step": list(range(len(bounds_us))),
            "start_ns": [s * US for s, _ in bounds_us],
            "end_ns": [e * US for _, e in bounds_us],
            "duration_ns": [(e - s) * US for s, e in bounds_us],
        }
    )


def _ops(calls: int, nbytes: int = 0) -> pl.DataFrame:
    """Builds ``calls`` allreduce rows of rank 0."""
    return pl.DataFrame(
        {
            "rank": [0] * calls,
            "op": ["allreduce"] * calls,
            "bytes": [nbytes] * calls,
            "duration_ns": [100 * US] * calls,
        }
    )


def _findings(
    kernels: pl.DataFrame, steps: pl.DataFrame, ops: pl.DataFrame | None = None
) -> dict[str, advisor.Finding]:
    ops = ops if ops is not None else _ops(0)
    return {f.rule: f for f in advisor.timeline_findings(steps, kernels, ops)}


def test_merge_intervals_joins_overlapping_and_touching_ones() -> None:
    starts, ends = advisor.merge_intervals(
        np.array([5, 0, 2, 20, 10]), np.array([8, 3, 4, 25, 12])
    )

    assert starts.tolist() == [0, 5, 10, 20]
    assert ends.tolist() == [4, 8, 12, 25]


def test_gaps_are_measured_between_the_union_of_streams() -> None:
    # Two streams: the second one covers the hole the first one leaves.
    length, inside = advisor.gaps_inside(
        np.array([0, 10, 12, 40]),
        np.array([10, 20, 18, 50]),
        np.array([0]),
        np.array([100]),
    )

    assert length.tolist() == [20]
    assert inside.tolist() == [20]


def test_a_gap_counts_only_where_it_lies_inside_a_step() -> None:
    # The gap 20..60 is half inside step 0 (0..40) and 10 inside step 1.
    length, inside = advisor.gaps_inside(
        np.array([0, 60]),
        np.array([20, 80]),
        np.array([0, 50]),
        np.array([40, 100]),
    )

    assert length.tolist() == [40]
    assert inside.tolist() == [20 + 10]


def test_overlapping_steps_are_counted_once() -> None:
    # Steps 0..50 and 20..80 cover 0..80; the gap 10..60 is inside, once.
    _, inside = advisor.gaps_inside(
        np.array([0, 60]),
        np.array([10, 70]),
        np.array([0, 20]),
        np.array([50, 80]),
    )

    assert inside.tolist() == [50]


def test_tiny_kernels_with_launch_gaps_are_launch_bound() -> None:
    # Two steps of ten 5 us kernels, each followed by a 15 us launch gap.
    rows = []
    for step in range(2):
        for i in range(10):
            start = step * 200 + i * 20
            rows.append(("compute", "k", start, start + 5))
    steps = _steps([(0, 200), (200, 400)])

    found = _findings(_kernels(rows), steps)["launch_bound"]

    # 19 gaps of 15 us between the 20 kernels, all inside a step.
    assert found.ceiling_ns == pytest.approx(19 * 15 * US / 2)
    assert found.evidence["tiny_kernel_share"] == 1.0
    assert found.evidence["median_kernel_us"] == 5.0
    assert found.evidence["kernels_per_step"] == 10.0


def test_long_kernels_are_not_launch_bound() -> None:
    rows = [("compute", "gemm", i * 600, i * 600 + 500) for i in range(6)]

    assert "launch_bound" not in _findings(
        _kernels(rows), _steps([(0, 1800), (1800, 3600)])
    )


def test_big_gaps_are_the_host_not_the_launches() -> None:
    # Tiny kernels but 5 ms apart: graphs would not close that gap.
    rows = [("compute", "k", i * 5000, i * 5000 + 5) for i in range(6)]

    assert "launch_bound" not in _findings(
        _kernels(rows), _steps([(0, 15000), (15000, 30000)])
    )


def test_launch_ceiling_is_the_exact_gap_time_per_step() -> None:
    # One step 0..100: kernels 0-10, 20-30, 40-50 leave two 10 us gaps.
    rows = [("compute", "k", s, s + 10) for s in (0, 20, 40)]

    found = _findings(_kernels(rows), _steps([(0, 100)]))["launch_bound"]

    assert found.ceiling_ns == 2 * 10 * US
    assert found.step_ns == 100 * US


def test_comm_beside_compute_is_not_exposed() -> None:
    rows = [
        ("compute", "gemm", 0, 1000),
        ("comm", "ncclDevKernel_AllReduce", 100, 900),
    ]

    assert "exposed_comm" not in _findings(_kernels(rows), _steps([(0, 1000)]))


def test_exposed_comm_is_the_part_no_compute_covers() -> None:
    rows = [
        ("compute", "gemm", 0, 800),
        ("comm", "ncclDevKernel_AllReduce", 700, 1000),
    ]

    found = _findings(_kernels(rows), _steps([(0, 1000)]))["exposed_comm"]

    assert found.ceiling_ns == 200 * US
    assert found.evidence["exposed_comm_ms_per_step"] == pytest.approx(0.2)


def test_many_small_calls_are_bucketing_few_large_ones_overlap() -> None:
    rows = [("compute", "gemm", 0, 100), ("comm", "nccl", 200, 400)]
    steps = _steps([(0, 500)])

    small = _findings(_kernels(rows), steps, _ops(20, nbytes=4096))
    large = _findings(_kernels(rows), steps, _ops(2, nbytes=64 << 20))
    unknown_size_many = _findings(_kernels(rows), steps, _ops(30, nbytes=0))
    unknown_size_few = _findings(_kernels(rows), steps, _ops(3, nbytes=0))

    assert "Bucket" in small["exposed_comm"].action
    assert "Overlap" in large["exposed_comm"].action
    assert "Bucket" in unknown_size_many["exposed_comm"].action
    assert "Overlap" in unknown_size_few["exposed_comm"].action
    assert (
        "median_message_bytes" not in unknown_size_many["exposed_comm"].evidence
    )


def test_only_host_device_copies_are_advised_on() -> None:
    rows = [
        ("compute", "gemm", 0, 300),
        ("memcpy", "Memcpy HtoD (Pageable -> Device)", 400, 500, 1 << 20),
        ("memcpy", "Memcpy_D2D", 600, 700, 1 << 20),
    ]

    found = _findings(_kernels(rows), _steps([(0, 1000)]))["exposed_copies"]

    assert found.ceiling_ns == 100 * US
    assert found.evidence["pageable_copies"] == 1.0
    assert found.evidence["copies_per_step"] == 1.0
    assert found.evidence["achieved_gb_per_s"] == pytest.approx(
        (1 << 20) / (100 * US)
    )


def test_pageable_copies_are_advised_to_be_pinned() -> None:
    rows = [
        ("compute", "gemm", 0, 300),
        ("memcpy", "Memcpy HtoD (Pageable -> Device)", 400, 500, 1 << 20),
    ]

    found = _findings(_kernels(rows), _steps([(0, 1000)]))["exposed_copies"]

    assert found.action.startswith("The copies come from pageable")
    assert "pin it" in found.action


def test_copies_not_known_to_be_pageable_are_advised_to_overlap() -> None:
    # nsys names say only the direction; 100 GB/s is no pageable copy.
    rows = [
        ("compute", "gemm", 0, 300),
        ("memcpy", "Memcpy_H2D", 400, 500, 10_000_000),
    ]

    found = _findings(_kernels(rows), _steps([(0, 1000)]))["exposed_copies"]

    assert found.action.startswith("Issue the copy early")
    assert "pageable_copies" not in found.evidence
    assert "If the host buffers are pageable" in found.action


def test_the_slowest_device_sets_the_ceiling() -> None:
    fast = _kernels([("compute", "g", 0, 800), ("comm", "n", 700, 800)])
    slow = _kernels([("compute", "g", 0, 500), ("comm", "n", 400, 900)])
    kernels = pl.concat(
        [fast, slow.with_columns(pl.lit(1, dtype=pl.Int64).alias("device"))]
    )

    found = _findings(kernels, _steps([(0, 1000)]))["exposed_comm"]

    assert found.ceiling_ns == 400 * US
    assert found.evidence["timeline"] == "rank 0 device 1"


def test_graph_ceilings_become_ranked_recommendations() -> None:
    ceiling = pl.DataFrame(
        {
            "group": ["compute", "NCCL_AllReduce", "SEND", "(other)"],
            "time_share_pct": [70.0, 20.0, 1.0, 9.0],
            "critical_share_pct": [60.0, 30.0, 0.5, 9.5],
            "ceiling_speedup": [2.5, 1.5, 1.001, 1.1],
            "knee_factor": [4.0, float("inf"), float("inf"), 2.0],
        }
    )
    sweep = pl.DataFrame({"makespan_ns": [8_000_000.0], "speedup": [1.25]})

    recs = advisor.whatif_recommendations("g", ceiling, sweep)

    # (other) is a pool, and SEND saves under MIN_CEILING_PCT.
    assert [r.rule for r in recs] == [
        "whatif:compute",
        "whatif:NCCL_AllReduce",
    ]
    compute = recs[0]
    assert compute.basis == "whatif-removal"
    assert compute.per == "run"
    # 8 ms * 1.25 = 10 ms unchanged; removing compute leaves 1/2.5 of it.
    assert compute.ceiling_ms == pytest.approx(10.0 * 0.6)
    assert compute.ceiling_pct == pytest.approx(60.0)
    assert compute.evidence["knee_factor"] == 4.0
    assert recs[1].evidence["knee_factor"] == "only removal"
    assert "Overlap" in recs[1].action
    assert "Speed up" in recs[0].action


def test_recommendations_are_ranked_and_small_ones_dropped() -> None:
    def rec(rule: str, ms: float) -> models.Recommendation:
        return models.Recommendation(
            "g", 0, rule, rule, "do", ms, ms, "step", "measured-removal", {}
        )

    ranked = advisor._rank("g", [rec("a", 1.0), rec("b", 5.0), rec("c", 3.0)])

    assert [(r.rule, r.priority) for r in ranked] == [
        ("b", 1),
        ("c", 2),
        ("a", 3),
    ]


def test_ceilings_of_a_generated_nsys_trace_match_the_overlap_aspect(
    tmp_path: pathlib.Path,
) -> None:
    path = synth_nsys.write_sqlite(
        tmp_path / "a.sqlite",
        steps=4,
        gemm_us=800,
        comm_us=400,
        comm_overlap_us=100,
    )

    report = cta.analyze(path, cache_dir=tmp_path / "cache")

    by_rule = {r.rule: r for r in report.recommendations}
    comm = by_rule["exposed_comm"]
    copies = by_rule.get("exposed_copies")
    overlap = report.aspects["overlap"].summary.filter(
        pl.col("metric") == "exposed_comm_ms"
    )
    overlap_total_ms = overlap.row(0, named=True)["a"]
    steps = 4
    advisor_total_ms = comm.evidence["exposed_comm_ms_per_step"] * steps + (
        copies.evidence["exposed_copy_ms_per_step"] * steps if copies else 0
    )
    # The aspect counts comm and copies as one class, so it is at most the
    # sum, and the same when the two never overlap each other.
    assert advisor_total_ms >= overlap_total_ms - 1e-9
    assert comm.ceiling_ms == pytest.approx(0.3)
    assert comm.per == "step"
    assert comm.basis == "measured-removal"
    assert 0 < comm.ceiling_pct <= 100
    step_ms = 1.15
    assert comm.ceiling_ms < step_ms


def test_per_step_ceilings_do_not_depend_on_how_many_steps_are_kept(
    tmp_path: pathlib.Path,
) -> None:
    path = synth_nsys.write_sqlite(
        tmp_path / "a.sqlite", steps=6, warmup_penalty_us=0
    )

    every = cta.analyze(path, cache_dir=tmp_path / "c1")
    some = cta.analyze(
        path, cache_dir=tmp_path / "c1", warmup_steps=1, max_steps=3
    )

    comm_every = next(
        r for r in every.recommendations if r.rule == "exposed_comm"
    )
    comm_some = next(
        r for r in some.recommendations if r.rule == "exposed_comm"
    )
    assert comm_some.ceiling_ms == pytest.approx(comm_every.ceiling_ms)


def test_no_ceiling_exceeds_its_step(tmp_path: pathlib.Path) -> None:
    path = synth_nsys.write_sqlite(tmp_path / "a.sqlite", steps=3)

    report = cta.analyze(path, cache_dir=tmp_path / "cache")

    assert report.recommendations
    for rec in report.recommendations:
        assert 0 < rec.ceiling_pct <= 100
    ranks = [r.priority for r in report.recommendations]
    assert ranks == list(range(1, len(ranks) + 1))
    ceilings = [r.ceiling_ms for r in report.recommendations]
    assert ceilings == sorted(ceilings, reverse=True)


def test_every_group_of_a_comparison_gets_its_own_advice(
    tmp_path: pathlib.Path,
) -> None:
    base = synth_nsys.write_sqlite(tmp_path / "base.sqlite", comm_us=400)
    cand = synth_nsys.write_sqlite(
        tmp_path / "cand.sqlite", comm_us=400, comm_overlap_us=400
    )

    report = cta.compare(base, cand, cache_dir=tmp_path / "cache")

    labels = {r.group_label for r in report.recommendations}
    assert "base" in labels
    base_comm = next(
        r
        for r in report.recommendations
        if r.group_label == "base" and r.rule == "exposed_comm"
    )
    # The candidate hides its collective behind the GEMM.
    assert not any(
        r.group_label == "cand" and r.rule == "exposed_comm"
        for r in report.recommendations
    )
    assert base_comm.ceiling_ms > 0


def test_the_report_shows_advice_in_markdown_and_json(
    tmp_path: pathlib.Path,
) -> None:
    path = synth_nsys.write_sqlite(tmp_path / "a.sqlite")
    report = cta.analyze(path, cache_dir=tmp_path / "cache")

    markdown = report.to_markdown()
    exported = json.loads(
        (report.export(tmp_path / "out", formats=["json"])["json"]).read_text()
    )

    assert "## Optimization Advice" in markdown
    assert "measured-removal" in markdown
    assert exported["recommendations"][0]["rule"] in {
        "exposed_comm",
        "exposed_copies",
        "launch_bound",
    }
    assert exported["recommendations"][0]["priority"] == 1


def test_graph_advice_needs_the_whatif_aspect(tmp_path: pathlib.Path) -> None:
    goal = tmp_path / "t.goal"
    training.write_goal(goal, training.TrainingSpec(dp=4, tp=2, pp=2))

    without = cta.analyze(goal, cache_dir=tmp_path / "cache")
    with_whatif = cta.analyze(
        goal, cache_dir=tmp_path / "cache", aspects=("whatif",)
    )

    assert without.recommendations == []
    assert with_whatif.recommendations
    assert {r.basis for r in with_whatif.recommendations} == {"whatif-removal"}
    assert {r.per for r in with_whatif.recommendations} == {"run"}
    assert [r.priority for r in with_whatif.recommendations] == list(
        range(1, len(with_whatif.recommendations) + 1)
    )


REAL_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data_real"
_REAL_NSYS = REAL_DATA_DIR / "a_nsys" / "single_fp16.sqlite"
_REAL_KINETO = REAL_DATA_DIR / "e" / "kineto"


@pytest.mark.skipif(
    not _REAL_NSYS.exists(), reason="real GPULab captures not present"
)
def test_real_single_gpu_capture_is_launch_bound(
    tmp_path: pathlib.Path,
) -> None:
    # fp16 matmuls of 3 us each, 10 a step, on an otherwise idle GPU.
    report = cta.analyze(
        _REAL_NSYS, cache_dir=tmp_path / "cache", warmup_steps=2
    )

    first = report.recommendations[0]
    assert first.rule == "launch_bound"
    assert first.evidence["tiny_kernel_share"] == 1.0
    assert 0 < first.ceiling_pct <= 100
    copies = [r for r in report.recommendations if r.rule == "exposed_copies"]
    # 38 GB/s is no pageable copy: the advice must not start with pinning.
    assert copies
    assert copies[0].evidence["achieved_gb_per_s"] > 20
    assert copies[0].action.startswith("Issue the copy early")
    for rec in report.recommendations:
        assert rec.ceiling_pct <= 100


@pytest.mark.skipif(
    not _REAL_KINETO.exists(), reason="real GPULab captures not present"
)
def test_real_two_rank_kineto_ceiling_agrees_with_the_overlap_aspect(
    tmp_path: pathlib.Path,
) -> None:
    report = cta.analyze(
        _REAL_KINETO, cache_dir=tmp_path / "cache", warmup_steps=1
    )

    tables = report.sources[report.groups[0].group_id][0]
    steps = pl.read_parquet(tables["steps"]).sort("rank", "start_ns")
    kept = steps.with_columns(
        pl.col("start_ns").cum_count().over("rank").alias("i")
    ).filter(pl.col("i") > 1)
    timelines = (
        pl.read_parquet(tables["kernels"]).select("rank", "device").unique()
    )
    total_ms = (
        report.aspects["overlap"]
        .summary.filter(pl.col("metric") == "exposed_comm_ms")
        .row(0, named=True)["kineto"]
    )
    mean_per_step_ms = total_ms / kept.height
    comm = next(r for r in report.recommendations if r.rule == "exposed_comm")

    # The aspect sums comm and copies over the kept steps of every rank, so
    # it is a mean over timelines; the advisor keeps the worst timeline.
    assert timelines.height == kept["rank"].n_unique()
    assert comm.ceiling_ms >= mean_per_step_ms * (1 - 1e-9)
    assert comm.ceiling_ms <= kept["duration_ns"].mean() / 1e6
