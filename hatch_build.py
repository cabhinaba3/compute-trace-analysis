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

# Cache entries that hold a path no later configure overwrites. pip builds in
# a temporary environment that is deleted afterwards, so a cache from an
# earlier run points at a cmake, ninja and nanobind that are gone.
_PATH_ENTRIES = (
    "CMAKE_COMMAND",
    "CMAKE_ROOT",
    "CMAKE_MAKE_PROGRAM",
    "NB_DIR",
)


def cmake_supports_fresh() -> bool:
    """Returns whether the available cmake has the ``--fresh`` option (3.24)."""
    output = subprocess.run(
        ["cmake", "--version"], capture_output=True, text=True, check=True
    ).stdout
    major, minor = output.split()[2].split("-")[0].split(".")[:2]
    return (int(major), int(minor)) >= (3, 24)


def read_cache(cache_file: pathlib.Path) -> dict[str, str]:
    """Reads the ``NAME:TYPE=value`` entries of a CMakeCache.txt.

    Args:
        cache_file: Path of the cache file.

    Returns:
        Entry name to value; empty when the file does not exist.
    """
    entries: dict[str, str] = {}
    if not cache_file.exists():
        return entries
    for line in cache_file.read_text(errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line:
            continue
        name_type, _, value = line.partition("=")
        entries[name_type.partition(":")[0]] = value
    return entries


def cache_is_stale(
    build_dir: pathlib.Path, source_dir: pathlib.Path, generator: str
) -> bool:
    """Tells whether an existing CMake cache would break the next configure.

    The cache is stale when a recorded tool path no longer exists, when it was
    made for another generator, or when the build or source directory moved.

    Args:
        build_dir: The build directory.
        source_dir: The directory holding CMakeLists.txt.
        generator: The generator the next configure uses.

    Returns:
        False when there is no cache or it is still usable.
    """
    entries = read_cache(build_dir / "CMakeCache.txt")
    if not entries:
        return False
    if entries.get("CMAKE_GENERATOR", generator) != generator:
        return True
    if entries.get("CMAKE_CACHEFILE_DIR", str(build_dir)) != str(build_dir):
        return True
    if entries.get("CMAKE_HOME_DIRECTORY", str(source_dir)) != str(source_dir):
        return True
    return any(
        name in entries and not pathlib.Path(entries[name]).exists()
        for name in _PATH_ENTRIES
    )


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
        source_dir = root / "native"
        build_dir = root / "build" / "native"
        fresh = cache_is_stale(build_dir, source_dir, "Ninja")
        if fresh:
            self.app.display_info(
                "cta: starting from an empty CMake cache, which is stale"
            )
            if not cmake_supports_fresh():
                # Before cmake 3.24: remove the cache by hand.
                (build_dir / "CMakeCache.txt").unlink(missing_ok=True)
                fresh = False
        subprocess.run(
            [
                "cmake",
                *(["--fresh"] if fresh else []),
                "-S",
                str(source_dir),
                "-B",
                str(build_dir),
                "-G",
                "Ninja",
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
