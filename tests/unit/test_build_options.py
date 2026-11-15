"""Tests of the stale-cache check in hatch_build.py.

The build hook is read from the repository root, so the cache logic runs
without compiling anything.
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
