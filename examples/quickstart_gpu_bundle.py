"""Quickstart: Multi-artifact GPU RunBundle comparison (Nsight Systems + TensorRT + Jtop)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cta
from cta.synth.generators import (
    write_jtop_trace,
    write_nsys_sqlite,
    write_trtexec_layers,
    write_trtexec_timing,
)


def _populate_bundle(bundle_dir: Path, *, compute_ms: float, power_mw: float, precision: str, fused: bool) -> Path:
    write_nsys_sqlite(bundle_dir / "profile.sqlite", gemm_us=int(compute_ms * 300), comm_us=240)
    write_trtexec_timing(bundle_dir / "times.json", queries=40, compute_ms=compute_ms)
    write_trtexec_layers(bundle_dir / "profile.json", bundle_dir / "engine.json", fused=fused, precision=precision)
    write_jtop_trace(bundle_dir / "jtop.json", duration_s=4, power_mw=power_mw)
    return bundle_dir


def main(work_dir: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = work_dir or Path(tmp)
        fp16_dir = _populate_bundle(root / "fp16", compute_ms=3.2, power_mw=22_000.0, precision="FP16", fused=False)
        int8_dir = _populate_bundle(root / "int8", compute_ms=1.3, power_mw=14_000.0, precision="INT8", fused=True)

        report = cta.compare(
            fp16_dir,
            int8_dir,
            labels=["FP16_Unfused", "INT8_Fused"],
            warmup_steps=2,
            cache_dir=root / ".cta_cache",
        )
        print(report.to_markdown())

        exported = report.export(root / "artifacts", formats=("markdown", "latex", "csv", "json", "pdf"))
        report.plot(kind="tornado", path=root / "artifacts" / "tornado.pdf")
        print(f"Exported {len(exported) + 1} report & figure files.")


if __name__ == "__main__":
    main()
