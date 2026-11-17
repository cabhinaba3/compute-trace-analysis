"""Hatch build hook: compiles the C++ extension into the cta package.

One build for everything: `pip install .` (or `-e .`) runs CMake on native/,
which writes `src/cta/_native*.so` next to the Python code, and the wheel is
tagged as a platform wheel.

The build can be adjusted with environment variables (all optional):

    CTA_BUILD_DIR     CMake build directory (default: build/native).
    CTA_GENERATOR     CMake generator (default: Ninja if installed, else
                      "Unix Makefiles").
    CTA_BUILD_TYPE    CMake build type (default: Release).
    CTA_BUILD_JOBS    Parallel compile jobs (default: the generator's choice).
    CTA_PREFIX_PATH   Extra prefixes to search for libraries, separated by
                      ":" (becomes CMAKE_PREFIX_PATH).
    CTA_CMAKE_ARGS    Extra arguments for the CMake configure step.
    CTA_FRESH         "auto" (default) starts from an empty CMake cache only
                      when the cache is stale, "always" always does, "never"
                      never does.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import pathlib
import shlex
import shutil
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
_FRESH_MODES = ("auto", "always", "never")


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


def choose_generator(env: Mapping[str, str], ninja_found: bool) -> str:
    """Picks the CMake generator.

    Args:
        env: The environment variables.
        ninja_found: Whether ``ninja`` is on the PATH.

    Returns:
        ``CTA_GENERATOR`` if set, else Ninja when installed, else Makefiles.
    """
    if env.get("CTA_GENERATOR"):
        return env["CTA_GENERATOR"]
    return "Ninja" if ninja_found else "Unix Makefiles"


def use_fresh(env: Mapping[str, str], stale: bool) -> bool:
    """Decides whether to configure from an empty cache.

    Args:
        env: The environment variables.
        stale: Whether the existing cache is stale.

    Returns:
        True when ``CTA_FRESH`` is ``always``, or ``auto`` (the default) and
        the cache is stale.

    Raises:
        ValueError: ``CTA_FRESH`` is not one of auto, always, never.
    """
    mode = env.get("CTA_FRESH", "auto").lower()
    if mode not in _FRESH_MODES:
        raise ValueError(
            f"CTA_FRESH is '{mode}'; use one of {', '.join(_FRESH_MODES)}"
        )
    return mode == "always" or (mode == "auto" and stale)


def configure_command(
    env: Mapping[str, str],
    *,
    source_dir: pathlib.Path,
    build_dir: pathlib.Path,
    output_dir: pathlib.Path,
    python: str,
    nanobind_dir: str,
    generator: str,
    fresh: bool,
) -> list[str]:
    """Builds the cmake configure command.

    Args:
        env: The environment variables.
        source_dir: The directory holding CMakeLists.txt.
        build_dir: The build directory.
        output_dir: Where the extension is written (the cta package).
        python: The Python interpreter the extension is built for.
        nanobind_dir: The nanobind CMake directory.
        generator: The CMake generator.
        fresh: Whether to pass ``--fresh``.

    Returns:
        The argument list; ``CTA_CMAKE_ARGS`` comes last, so it can override.
    """
    command = ["cmake"]
    if fresh:
        command.append("--fresh")
    command += [
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        generator,
        f"-DCMAKE_BUILD_TYPE={env.get('CTA_BUILD_TYPE', 'Release')}",
        f"-DPython_EXECUTABLE={python}",
        f"-Dnanobind_DIR={nanobind_dir}",
        f"-DCTA_NATIVE_OUTPUT={output_dir}",
    ]
    if env.get("CTA_PREFIX_PATH"):
        command.append(f"-DCMAKE_PREFIX_PATH={env['CTA_PREFIX_PATH']}")
    command += shlex.split(env.get("CTA_CMAKE_ARGS", ""))
    return command


def build_command(env: Mapping[str, str], build_dir: pathlib.Path) -> list[str]:
    """Builds the ``cmake --build`` command.

    Args:
        env: The environment variables.
        build_dir: The build directory.

    Returns:
        The argument list, with ``-j`` when ``CTA_BUILD_JOBS`` is set.

    Raises:
        ValueError: ``CTA_BUILD_JOBS`` is not a positive integer.
    """
    command = ["cmake", "--build", str(build_dir)]
    jobs = env.get("CTA_BUILD_JOBS")
    if jobs:
        if not jobs.isdigit() or int(jobs) < 1:
            raise ValueError(
                f"CTA_BUILD_JOBS is '{jobs}'; use a positive integer"
            )
        command += ["-j", jobs]
    return command


class NativeBuildHook(BuildHookInterface):
    """Builds `cta._native` with CMake and nanobind."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        """Compiles the extension into ``src/cta/`` and tags the wheel as native.

        Args:
            version: Build version (``"standard"`` or ``"editable"``).
            build_data: Hatch's mutable build configuration.
        """
        if self.target_name != "wheel":
            return  # sdists ship the sources; only wheels compile them
        import nanobind

        env = os.environ
        root = pathlib.Path(self.root)
        source_dir = root / "native"
        build_dir = pathlib.Path(
            env.get("CTA_BUILD_DIR") or root / "build" / "native"
        ).resolve()
        generator = choose_generator(env, shutil.which("ninja") is not None)
        stale = cache_is_stale(build_dir, source_dir, generator)
        fresh = use_fresh(env, stale)
        if fresh:
            reason = "is stale" if stale else "was requested (CTA_FRESH)"
            self.app.display_info(
                f"cta: starting from an empty CMake cache, which {reason}"
            )
        if fresh and not cmake_supports_fresh():
            # Before cmake 3.24: remove the cache by hand.
            (build_dir / "CMakeCache.txt").unlink(missing_ok=True)
            fresh = False
        subprocess.run(
            configure_command(
                env,
                source_dir=source_dir,
                build_dir=build_dir,
                output_dir=root / "src" / "cta",
                python=sys.executable,
                nanobind_dir=nanobind.cmake_dir(),
                generator=generator,
                fresh=fresh,
            ),
            check=True,
        )
        subprocess.run(build_command(env, build_dir), check=True)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        build_data["artifacts"].append("src/cta/_native*.so")
