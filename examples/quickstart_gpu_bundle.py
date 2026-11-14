"""Quickstart: multi-artifact GPU RunBundle comparison.

Builds two synthetic RunBundles (Nsight Systems + TensorRT + Jtop), one FP16
and one fused INT8, compares them, and exports the report and a tornado plot.
"""

from __future__ import annotations

import pathlib
import tempfile

import cta
from cta.synth import jtop
from cta.synth import nsys
from cta.synth import tensorrt


def _populate_bundle(
    bundle_dir: pathlib.Path,
    *,
    compute_ms: float,
    power_mw: float,
    precision: str,
    is_fused: bool,
) -> pathlib.Path:
    """Writes the nsys, trtexec and jtop files of one synthetic RunBundle.

    Args:
        bundle_dir: Directory of the bundle; created if missing.
        compute_ms: Compute time per inference query in milliseconds.
        power_mw: Total board power in milliwatts.
        precision: TensorRT layer precision, for example ``"FP16"``.
        is_fused: Whether the TensorRT engine has fused layers.

    Returns:
        ``bundle_dir``.
    """
    nsys.write_nsys_sqlite(
        bundle_dir / "profile.sqlite",
        gemm_us=int(compute_ms * 300),
        comm_us=240,
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
    return bundle_dir


def main(work_dir: pathlib.Path | None = None) -> None:
    """Runs the quickstart and prints the comparison report.

    Args:
        work_dir: Directory for the bundles, cache and exported files; a
            temporary directory (removed afterwards) when ``None``.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_dir = work_dir or pathlib.Path(tmp_dir)
        fp16_dir = _populate_bundle(
            root_dir / "fp16",
            compute_ms=3.2,
            power_mw=22_000.0,
            precision="FP16",
            is_fused=False,
        )
        int8_dir = _populate_bundle(
            root_dir / "int8",
            compute_ms=1.3,
            power_mw=14_000.0,
            precision="INT8",
            is_fused=True,
        )

        report = cta.compare(
            fp16_dir,
            int8_dir,
            labels=["FP16_Unfused", "INT8_Fused"],
            warmup_steps=2,
            cache_dir=root_dir / ".cta_cache",
        )
        print(report.to_markdown())

        paths_by_format = report.export(
            root_dir / "artifacts",
            formats=("markdown", "latex", "csv", "json", "pdf"),
        )
        report.plot(kind="tornado", path=root_dir / "artifacts" / "tornado.pdf")
        print(f"Exported {len(paths_by_format) + 1} report & figure files.")


if __name__ == "__main__":
    main()
