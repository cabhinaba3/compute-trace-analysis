"""Reproducible multi-million-node benchmark for CTA streaming graph engines.

Synthesizes a 3D-parallel GOAL `.bin` training trace and benchmarks:
  - Engine A: 1-pass online aggregates + DDSketch quantiles
  - Engine B: 1-pass frontier / Kahn DAG solver over mmapped GOAL .bin
  - Engine C: GridGraph out-of-core exact DAG solver (.cta_graph build + solve)
  - Engine D: Streaming multi-lane what-if solver (10 scenarios + sweep)

Usage:
    PYTHONPATH=src python examples/benchmark_streaming.py --scale medium
"""

from __future__ import annotations

import argparse
import pathlib
import resource
import sys
import tempfile
import time

from cta import models
from cta.analysis import streaming
from cta.analysis import whatif
from cta.synth import training


def _peak_rss_mib() -> float:
    """Returns process peak RSS in MiB (macOS reports bytes, Linux KiB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = float(1024 * 1024) if sys.platform == "darwin" else 1024.0
    return rss / divisor


def _spec_for_scale(scale: str) -> training.TrainingSpec:
    """Returns a 3D-parallel TrainingSpec for the requested benchmark size."""
    if scale == "small":
        return training.TrainingSpec(
            dp=4, pp=4, tp=4, microbatches=8, layers_per_stage=6, steps=3
        )
    if scale == "medium":
        return training.TrainingSpec(
            dp=8, pp=4, tp=4, microbatches=8, layers_per_stage=8, steps=4
        )
    return training.TrainingSpec(
        dp=8, pp=4, tp=8, microbatches=16, layers_per_stage=8, steps=4
    )


def main() -> None:
    """Synthesizes a large GOAL .bin graph and benchmarks Engines A–D."""
    parser = argparse.ArgumentParser(
        description="Benchmark CTA streaming graph engines A–D."
    )
    parser.add_argument(
        "--scale",
        choices=("small", "medium", "large"),
        default="medium",
        help="Synthetic 3D-parallel trace size.",
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=128,
        help="Explicit RAM budget in MiB for out-of-core engines C & D.",
    )
    args = parser.parse_args()
    budget = args.memory_mb * 1024 * 1024

    spec = _spec_for_scale(args.scale)
    with tempfile.TemporaryDirectory(prefix="cta_bench_") as tmp_str:
        tmp = pathlib.Path(tmp_str)
        bin_path = tmp / "train.bin"
        t0 = time.perf_counter()
        training.write_goal_bin(bin_path, spec)
        synth_s = time.perf_counter() - t0
        raw_mb = bin_path.stat().st_size / (1024 * 1024)
        art = models.TraceArtifact(
            format="goal", paths=(bin_path,), world_size=spec.ranks
        )
        print(
            f"Synthesized {bin_path.name} ({spec.ranks} ranks, "
            f"{raw_mb:.1f} MiB) in {synth_s:.2f}s"
        )

        t0 = time.perf_counter()
        agg = streaming.aggregate_artifact(art, quantile_sketch="ddsketch")
        t_agg = time.perf_counter() - t0
        print(
            f"[Engine A — Online Agg]   {agg.num_nodes:,} nodes in "
            f"{t_agg:.3f}s ({agg.num_nodes / max(t_agg, 1e-6) / 1e6:.2f} M/s) "
            f"| peak RSS {(_peak_rss_mib()):.1f} MiB"
        )

        t0 = time.perf_counter()
        front = streaming.solve_artifact_frontier(
            art, scratch_dir=str(tmp / "scr_b")
        )
        t_front = time.perf_counter() - t0
        print(
            f"[Engine B — Frontier]     {front.num_nodes:,} nodes, "
            f"{front.num_edges:,} edges in {t_front:.3f}s "
            f"({front.num_nodes / max(t_front, 1e-6) / 1e6:.2f} M/s) "
            f"| exact={front.provenance.exact} "
            f"| cp={front.critical_path_ns / 1e6:.3f} ms "
            f"| peak RSS {_peak_rss_mib():.1f} MiB"
        )

        cache_path = tmp / "train.cta_graph"
        t0 = time.perf_counter()
        grid = streaming.solve_artifact_outofcore(
            art,
            engine="gridgraph",
            memory_budget_bytes=budget,
            scratch_dir=str(tmp / "scr_c"),
            cache_path=cache_path,
        )
        t_grid = time.perf_counter() - t0
        cache_mb = cache_path.stat().st_size / (1024 * 1024)
        print(
            f"[Engine C — GridGraph]    build+solve ({args.memory_mb} MiB cap) "
            f"in {t_grid:.3f}s | cache={cache_mb:.1f} MiB "
            f"| exact={grid.provenance.exact} "
            f"| cp={grid.critical_path_ns / 1e6:.3f} ms "
            f"| peak RSS {_peak_rss_mib():.1f} MiB"
        )

        t0 = time.perf_counter()
        grid_warm = streaming.solve_artifact_outofcore(
            art,
            engine="gridgraph",
            memory_budget_bytes=budget,
            scratch_dir=str(tmp / "scr_c2"),
            cache_path=cache_path,
        )
        t_warm = time.perf_counter() - t0
        assert abs(grid_warm.critical_path_ns - front.critical_path_ns) <= 1e-6
        print(
            f"[Engine C — Warm Cache]   solve in {t_warm:.3f}s "
            f"({grid.num_nodes / max(t_warm, 1e-6) / 1e6:.2f} M/s) "
            f"| peak RSS {_peak_rss_mib():.1f} MiB"
        )

        model = whatif.build_streaming_model_from_artifact(
            art,
            cache_path=cache_path,
            memory_budget_bytes=budget,
            scratch_dir=str(tmp / "scr_d"),
        )
        scenarios = [
            whatif.parse_scenario(f"comm:{f}x")
            for f in (1.25, 1.5, 2.0, 3.0, 4.0)
        ] + [
            whatif.parse_scenario("compute:2x"),
            whatif.parse_scenario("rank=*:median"),
            whatif.parse_scenario("comm:remove"),
        ]
        t0 = time.perf_counter()
        run = whatif.run_scenarios(model, scenarios, memory_budget_bytes=budget)
        t_wi = time.perf_counter() - t0
        print(
            f"[Engine D — What-If]      {len(scenarios) + 1} scenarios "
            f"({run.lanes} lanes x {run.workers} threads) in {t_wi:.3f}s "
            f"| peak RSS {_peak_rss_mib():.1f} MiB"
        )


if __name__ == "__main__":
    main()
