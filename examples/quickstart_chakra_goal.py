"""Quickstart: Distributed Chakra-ET & GOAL schedule comparison and equivalence check."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cta
from cta.synth.generators import compile_goal_to_bin, write_chakra_et_run, write_goal_allreduce


def main(work_dir: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = work_dir or Path(tmp)
        cache_dir = root / ".cta_cache"

        chakra_base = root / "chakra_ring"
        chakra_skew = root / "chakra_straggler"
        write_chakra_et_run(chakra_base, world_size=4, compute_us=800, allreduce_bytes=512_000)
        write_chakra_et_run(
            chakra_skew,
            world_size=4,
            compute_us=800,
            allreduce_bytes=512_000,
            straggler_rank=1,
            straggler_factor=1.75,
        )

        goal_txt = write_goal_allreduce(
            root / "ring.goal", world_size=4, compute_ns=800_000, allreduce_bytes=512_000, algo="ring"
        )
        goal_bin = compile_goal_to_bin(goal_txt, root / "ring.bin")

        verdict = cta.equiv(chakra_base, goal_bin, collective_algo="ring", cache_dir=cache_dir)
        print(f"Chakra-ET <-> GOAL (.bin) equivalent: {verdict.equivalent}")
        print(verdict.checks)

        report = cta.compare(
            chakra_base,
            chakra_skew,
            labels=["Balanced_4R", "Straggler_R1"],
            cache_dir=cache_dir,
        )
        print("\n" + report.to_markdown())

        out_dir = root / "figures"
        for kind in ("breakdown", "skew", "traffic_delta"):
            report.plot(kind=kind, path=out_dir / f"{kind}.pdf")
        report.to_latex(out_dir / "distributed_tables.tex")
        print(f"Generated LaTeX & vector PDF figures in {out_dir}")


if __name__ == "__main__":
    main()
