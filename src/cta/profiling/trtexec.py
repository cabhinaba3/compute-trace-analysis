"""Runs TensorRT trtexec with the timing, profile and layer-info exports."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib

from cta.profiling import base


def build_argv(
    args: Sequence[str], bundle_dir: pathlib.Path, *, trtexec: str = "trtexec"
) -> list[str]:
    """Builds the ``trtexec`` command line with CTA's export options.

    Args:
        args: trtexec's own arguments, e.g. ``--onnx=model.onnx``.
        bundle_dir: Directory that receives ``times.json``, ``profile.json`` and
            ``layers.json``.
        trtexec: Path of the trtexec executable.

    Returns:
        The full command line.
    """
    exports = [
        f"--exportTimes={bundle_dir / 'times.json'}",
        f"--exportProfile={bundle_dir / 'profile.json'}",
        f"--exportLayerInfo={bundle_dir / 'layers.json'}",
        "--profilingVerbosity=detailed",
        "--dumpProfile",
    ]
    return [trtexec, *args, *exports]


def run(
    args: Sequence[str], out_dir: pathlib.Path, name: str
) -> base.ProfileResult:
    """Runs TensorRT's trtexec and keeps its timing, profile and layer files.

    Args:
        args: trtexec's own arguments (``--onnx``/``--loadEngine``,
            ``--fp16``, ...); the three export options are added.
        out_dir: Directory that receives the bundle directory.
        name: Bundle directory name.

    Returns:
        The run result.

    Raises:
        base.ProfilerError: If trtexec is not found.
    """
    trtexec = base.find_tool(
        "trtexec",
        "TRTEXEC_PATH",
        "Install TensorRT (on Jetson: /usr/src/tensorrt/bin/trtexec).",
    )
    bundle_dir = base.make_bundle_dir(out_dir, name)
    argv = build_argv(args, bundle_dir, trtexec=trtexec)
    returncode = base.run_echoed(argv)
    export_paths = (
        bundle_dir / "times.json",
        bundle_dir / "profile.json",
        bundle_dir / "layers.json",
    )
    artifacts = [path for path in export_paths if path.exists()]
    return base.ProfileResult(
        "trtexec", bundle_dir, argv, returncode, artifacts
    )
