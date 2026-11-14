"""Quickstart: distributed Chakra-ET and GOAL schedule comparison.

Builds a balanced and a straggler Chakra-ET run, checks that the balanced run
matches its compiled GOAL ring schedule, compares the two runs, and writes the
figures and LaTeX tables.
"""

from __future__ import annotations

import pathlib
import tempfile

import cta
from cta.synth import chakra
from cta.synth import goal


def main(work_dir: pathlib.Path | None = None) -> None:
    """Runs the quickstart and prints the equivalence check and the report.

    Args:
        work_dir: Directory for the traces, cache and generated files; a
            temporary directory (removed afterwards) when ``None``.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_dir = work_dir or pathlib.Path(tmp_dir)
        cache_dir = root_dir / ".cta_cache"

        balanced_dir = root_dir / "chakra_ring"
        straggler_dir = root_dir / "chakra_straggler"
        chakra.write_chakra_et_run(
            balanced_dir, world_size=4, compute_us=800, allreduce_bytes=512_000
        )
        chakra.write_chakra_et_run(
            straggler_dir,
            world_size=4,
            compute_us=800,
            allreduce_bytes=512_000,
            straggler_rank=1,
            straggler_factor=1.75,
        )

        goal_text_path = goal.write_goal_allreduce(
            root_dir / "ring.goal",
            world_size=4,
            compute_ns=800_000,
            allreduce_bytes=512_000,
            algorithm="ring",
        )
        goal_bin_path = goal.compile_goal_to_bin(
            goal_text_path, root_dir / "ring.bin"
        )

        verdict = cta.verify_chakra_goal_equivalence(
            balanced_dir,
            goal_bin_path,
            collective_algorithm="ring",
            cache_dir=cache_dir,
        )
        print(f"Chakra-ET <-> GOAL (.bin) equivalent: {verdict.equivalent}")
        print(verdict.checks)

        report = cta.compare(
            balanced_dir,
            straggler_dir,
            labels=["Balanced_4R", "Straggler_R1"],
            cache_dir=cache_dir,
        )
        print("\n" + report.to_markdown())

        figures_dir = root_dir / "figures"
        for kind in ("breakdown", "skew", "traffic_delta"):
            report.plot(kind=kind, path=figures_dir / f"{kind}.pdf")
        report.to_latex(figures_dir / "distributed_tables.tex")
        print(f"Generated LaTeX & vector PDF figures in {figures_dir}")


if __name__ == "__main__":
    main()
