"""Tests of the plugin registries: lazy loading, overriding and lookups."""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

from cta import plugins


def test_registry_imports_its_builtins_on_first_lookup(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry: plugins.Registry[str] = plugins.Registry(
        "thing", ("cta_test_builtin",)
    )
    holder = types.ModuleType("cta_test_holder")
    holder.__dict__["REGISTRY"] = registry
    monkeypatch.setitem(sys.modules, "cta_test_holder", holder)
    (tmp_path / "cta_test_builtin.py").write_text(
        "import cta_test_holder\n"
        "cta_test_holder.REGISTRY.register('builtin', 'v', aliases=['b'])\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "cta_test_builtin", raising=False)

    assert "cta_test_builtin" not in sys.modules
    assert registry["b"] == "v"
    assert "cta_test_builtin" in sys.modules
    assert list(registry) == ["builtin"]


def test_registering_after_the_builtins_overrides_them() -> None:
    registry: plugins.Registry[str] = plugins.Registry("thing", ())
    registry.register("name", "built-in", aliases=("alias",))
    registry.register("name", "plugin")
    assert registry["name"] == "plugin"
    assert registry["alias"] == "plugin"
    assert registry.resolve("alias") == "name"


def test_unknown_names_raise_with_the_closest_match() -> None:
    registry: plugins.Registry[str] = plugins.Registry("trace format", ())
    registry.register("kineto", "adapter")
    with pytest.raises(KeyError, match="Did you mean 'kineto'"):
        registry["kinetto"]
    assert "kinetto" not in registry
    assert "kineto" in registry


def test_builtin_registries_list_every_builtin_plugin() -> None:
    assert {"chakra", "goal", "kineto", "nsys"} <= set(plugins.ADAPTERS)
    assert {"latency", "kernels", "critical_path"} <= set(plugins.ASPECTS)
