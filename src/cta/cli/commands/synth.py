"""The ``synth`` command: writes small synthetic traces to try CTA on."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from cta.cli import output

# Manifest written next to the synthetic traces by ``cta synth``.
_DEMO_SWEEP_MANIFEST = """groups:
  - id: fp16
    label: Jetson_FP16
    path: gpu_bundle_fp16
  - id: int8
    label: Jetson_INT8
    path: gpu_bundle_int8
config:
  warmup_steps: 2
  gates:
    mean_step_ms: 5.0
"""


def synth_command(
    out_dir: Annotated[
        pathlib.Path,
        typer.Argument(
            help=(
                "Directory to populate with synthetic demo traces & sweep "
                "manifest."
            )
        ),
    ] = pathlib.Path("demo_traces"),
) -> None:
    """Generates small valid traces of all 7 ecosystems.

    For demos and offline testing.
    """
    from cta.synth import chakra
    from cta.synth import goal
    from cta.synth import jtop
    from cta.synth import kineto
    from cta.synth import ncu
    from cta.synth import nsys
    from cta.synth import trtexec

    chakra.write_chakra_et_run(
        out_dir / "chakra_ring",
        world_size=4,
        compute_us=900,
        allreduce_bytes=400_000,
    )
    chakra.write_chakra_et_run(
        out_dir / "chakra_skewed",
        world_size=4,
        compute_us=900,
        allreduce_bytes=800_000,
        straggler_rank=2,
        straggler_factor=1.8,
    )

    ring_goal_path = goal.write_goal_allreduce(
        out_dir / "goal" / "ring.goal",
        world_size=4,
        compute_ns=900_000,
        allreduce_bytes=400_000,
        algorithm="ring",
    )
    goal.compile_goal_to_bin(ring_goal_path, out_dir / "goal" / "ring.bin")
    goal.write_goal_allreduce(
        out_dir / "goal" / "direct.goal",
        world_size=4,
        compute_ns=900_000,
        allreduce_bytes=400_000,
        algorithm="direct",
    )

    for mode, compute_ms, power_mw, precision, is_fused in (
        ("fp16", 3.2, 22_000.0, "FP16", False),
        ("int8", 1.3, 14_000.0, "INT8", True),
    ):
        bundle_dir = out_dir / f"gpu_bundle_{mode}"
        nsys.write_nsys_sqlite(
            bundle_dir / "profile.sqlite",
            gemm_us=int(compute_ms * 300),
            comm_us=250,
        )
        trtexec.write_trtexec_timing(
            bundle_dir / "times.json", queries=40, compute_ms=compute_ms
        )
        trtexec.write_trtexec_layers(
            bundle_dir / "profile.json",
            bundle_dir / "engine.json",
            fused=is_fused,
            precision=precision,
        )
        jtop.write_jtop_trace(
            bundle_dir / "jtop.json", duration_s=4, power_mw=power_mw
        )

    kineto.write_kineto_trace(
        out_dir / "kineto" / "eager.json.gz",
        attn_us=450.0,
        gemm_us=600.0,
        kernel_variant="fp16",
    )
    kineto.write_kineto_trace(
        out_dir / "kineto" / "compiled_fp8.json.gz",
        attn_us=210.0,
        gemm_us=280.0,
        kernel_variant="fp8",
    )
    ncu.write_ncu_csv(
        out_dir / "ncu" / "baseline.csv", sm_pct=35.0, dram_pct=84.0
    )
    ncu.write_ncu_csv(
        out_dir / "ncu" / "optimized.csv",
        gemm_dur_ns=260_000.0,
        sm_pct=82.0,
        dram_pct=38.0,
        flops=4e10,
        dram_bytes=5e8,
    )

    (out_dir / "sweep.yaml").write_text(_DEMO_SWEEP_MANIFEST)
    output.console.print(
        "[green]✓ Generated synthetic multi-format demo workspace in:[/green] "
        f"{out_dir}"
    )


def register(app: typer.Typer) -> None:
    """Adds ``cta synth`` to the application."""
    app.command("synth")(synth_command)
