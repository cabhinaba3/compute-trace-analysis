"""Hatch build hook: compiles the C++ extension into the cta package.

One build for everything: `pip install .` (or `-e .`) runs CMake on native/,
which writes `cta/_native*.so` next to the Python code, and the wheel is
tagged as a platform wheel.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class NativeBuildHook(BuildHookInterface):
    """Builds `cta._native` with CMake and nanobind."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        """Compiles the extension into ``cta/`` and tags the wheel as native.

        Args:
            version: Build version (``"standard"`` or ``"editable"``).
            build_data: Hatch's mutable build configuration.
        """
        if self.target_name != "wheel":
            return  # sdists ship the sources; only wheels compile them
        import nanobind

        root = pathlib.Path(self.root)
        build_dir = root / "build" / "native"
        subprocess.run(
            [
                "cmake",
                "-S", str(root / "native"),
                "-B", str(build_dir),
                "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-Dnanobind_DIR={nanobind.cmake_dir()}",
                f"-DCTA_NATIVE_OUTPUT={root / 'cta'}",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build_dir)], check=True)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        build_data["artifacts"].append("cta/_native*.so")
