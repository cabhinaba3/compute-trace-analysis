"""The ``synth``, ``inspect`` and ``list`` commands."""

from __future__ import annotations

import pathlib
from typing import Annotated

from rich import pretty
from rich import table
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
    """Generate miniature valid traces across all 7 ecosystems for demos and offline testing."""  # noqa: E501
    from cta.synth import chakra
    from cta.synth import goal
    from cta.synth import jtop
    from cta.synth import kineto
    from cta.synth import ncu
    from cta.synth import nsys
    from cta.synth import tensorrt

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
        tensorrt.write_trtexec_timing(
            bundle_dir / "times.json", queries=40, compute_ms=compute_ms
        )
        tensorrt.write_trtexec_layers(
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


def inspect_command(
    source: Annotated[
        str, typer.Argument(help="Trace file, glob, or RunBundle directory.")
    ],
    cache_dir: Annotated[
        pathlib.Path, typer.Option("--cache-dir")
    ] = pathlib.Path(".cta_cache"),
) -> None:
    """Inspect detected trace formats, world size, and normalized Parquet table schemas."""  # noqa: E501
    from cta.core import engine

    info = engine.inspect_trace(source, cache_dir=cache_dir)
    output.console.print(pretty.Pretty(info))


def list_command() -> None:
    """List registered trace formats (adapters) and comparison aspects, including plugins."""  # noqa: E501
    from cta.core import registry

    registry.load_plugins()
    formats_table = table.Table(title="Trace formats")
    formats_table.add_column("format")
    formats_table.add_column("adapter version")
    for name, adapter in sorted(registry.ADAPTERS.items()):
        formats_table.add_row(name, adapter.version)
    output.console.print(formats_table)
    aspects_table = table.Table(title="Aspects")
    for column in ("name", "title", "requires"):
        aspects_table.add_column(column)
    for name, aspect in sorted(registry.ASPECTS.items()):
        mode = "any of" if aspect.any_of_formats else "all of"
        aspects_table.add_row(
            name, aspect.title, f"{mode}: {', '.join(aspect.required_formats)}"
        )
    output.console.print(aspects_table)
