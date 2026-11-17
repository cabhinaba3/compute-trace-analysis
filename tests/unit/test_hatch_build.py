"""Tests of the build options in hatch_build.py.

The build hook is read from the repository root, so the functions that turn
environment variables into CMake commands run without compiling anything.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import types

import pytest

pytest.importorskip("hatchling")

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cta_hatch_build", _ROOT / "hatch_build.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hatch_build = _load()
_PATHS = {
    "source_dir": pathlib.Path("/src/native"),
    "build_dir": pathlib.Path("/src/build/native"),
    "output_dir": pathlib.Path("/src/cta"),
    "python": "/usr/bin/python3",
    "nanobind_dir": "/nb/cmake",
    "generator": "Ninja",
}


def _configure(env: dict[str, str], fresh: bool = False) -> list[str]:
    return hatch_build.configure_command(env, fresh=fresh, **_PATHS)


def test_default_configure_command() -> None:
    assert _configure({}) == [
        "cmake",
        "-S",
        "/src/native",
        "-B",
        "/src/build/native",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPython_EXECUTABLE=/usr/bin/python3",
        "-Dnanobind_DIR=/nb/cmake",
        "-DCTA_NATIVE_OUTPUT=/src/cta",
    ]


def test_environment_variables_change_the_configure_command() -> None:
    command = _configure(
        {
            "CTA_BUILD_TYPE": "RelWithDebInfo",
            "CTA_PREFIX_PATH": "/opt/a:/opt/b",
            "CTA_CMAKE_ARGS": "-DFOO=1 '-DBAR=two words'",
        },
        fresh=True,
    )
    assert command[:2] == ["cmake", "--fresh"]
    assert "-DCMAKE_BUILD_TYPE=RelWithDebInfo" in command
    assert "-DCMAKE_PREFIX_PATH=/opt/a:/opt/b" in command
    # User arguments come last, so they can override the defaults.
    assert command[-2:] == ["-DFOO=1", "-DBAR=two words"]


def test_build_jobs() -> None:
    build_dir = pathlib.Path("/b")
    assert hatch_build.build_command({}, build_dir) == [
        "cmake",
        "--build",
        "/b",
    ]
    assert hatch_build.build_command({"CTA_BUILD_JOBS": "8"}, build_dir)[
        -2:
    ] == ["-j", "8"]
    for bad in ("0", "-2", "many"):
        with pytest.raises(ValueError, match="CTA_BUILD_JOBS"):
            hatch_build.build_command({"CTA_BUILD_JOBS": bad}, build_dir)


def test_generator_falls_back_to_makefiles_without_ninja() -> None:
    assert hatch_build.choose_generator({}, ninja_found=True) == "Ninja"
    assert (
        hatch_build.choose_generator({}, ninja_found=False) == "Unix Makefiles"
    )
    env = {"CTA_GENERATOR": "Unix Makefiles"}
    assert hatch_build.choose_generator(env, ninja_found=True) == (
        "Unix Makefiles"
    )


@pytest.mark.parametrize(
    ("mode", "stale", "expected"),
    [
        (None, True, True),
        (None, False, False),
        ("auto", True, True),
        ("always", False, True),
        ("ALWAYS", False, True),
        ("never", True, False),
    ],
)
def test_fresh_modes(mode: str | None, stale: bool, expected: bool) -> None:
    env = {} if mode is None else {"CTA_FRESH": mode}
    assert hatch_build.use_fresh(env, stale) is expected


def test_fresh_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="CTA_FRESH"):
        hatch_build.use_fresh({"CTA_FRESH": "sometimes"}, True)


def _write_cache(
    build_dir: pathlib.Path, source_dir: pathlib.Path, **entries: str
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "CMAKE_GENERATOR": "Ninja",
        "CMAKE_CACHEFILE_DIR": str(build_dir),
        "CMAKE_HOME_DIRECTORY": str(source_dir),
        **entries,
    }
    lines = ["# comment", "// help text"] + [
        f"{name}:INTERNAL={value}" for name, value in values.items()
    ]
    (build_dir / "CMakeCache.txt").write_text("\n".join(lines))


def test_cache_is_stale_when_a_recorded_tool_is_gone(
    tmp_path: pathlib.Path,
) -> None:
    source_dir, build_dir = tmp_path / "native", tmp_path / "build"
    tool = tmp_path / "ninja"
    tool.write_text("")
    _write_cache(build_dir, source_dir, CMAKE_MAKE_PROGRAM=str(tool))
    assert not hatch_build.cache_is_stale(build_dir, source_dir, "Ninja")
    tool.unlink()
    assert hatch_build.cache_is_stale(build_dir, source_dir, "Ninja")


def test_cache_is_stale_for_another_generator_or_a_moved_tree(
    tmp_path: pathlib.Path,
) -> None:
    source_dir, build_dir = tmp_path / "native", tmp_path / "build"
    _write_cache(build_dir, source_dir)
    assert not hatch_build.cache_is_stale(build_dir, source_dir, "Ninja")
    assert hatch_build.cache_is_stale(build_dir, source_dir, "Unix Makefiles")
    assert hatch_build.cache_is_stale(build_dir, tmp_path / "other", "Ninja")
    moved = tmp_path / "moved"
    build_dir.rename(moved)
    assert hatch_build.cache_is_stale(moved, source_dir, "Ninja")


def test_no_cache_is_not_stale(tmp_path: pathlib.Path) -> None:
    assert not hatch_build.cache_is_stale(
        tmp_path / "none", tmp_path / "src", "Ninja"
    )
    assert hatch_build.read_cache(tmp_path / "none" / "CMakeCache.txt") == {}


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("cmake version 3.18.4", False),
        ("cmake version 3.23.5", False),
        ("cmake version 3.24.0", True),
        ("cmake version 3.28.3", True),
        ("cmake version 4.0.0-rc1", True),
    ],
)
def test_fresh_needs_cmake_3_24(
    monkeypatch: pytest.MonkeyPatch, version: str, expected: bool
) -> None:
    def fake_run(*args: object, **kwargs: object) -> object:
        return types.SimpleNamespace(stdout=f"{version}\n\nCMake suite")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert hatch_build.cmake_supports_fresh() is expected
