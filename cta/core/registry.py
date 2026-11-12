from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from cta.core.models import AspectResult, CompareConfig, TraceArtifact, TrialGroup


class TraceAdapter(Protocol):
    format_name: str
    version: str

    def detect(self, path: Path) -> bool: ...
    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]: ...


class ComparisonAspect(Protocol):
    name: str
    title: str
    required_formats: tuple[str, ...]
    any_of_formats: bool  # If True, requires at least one of required_formats rather than all

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult: ...


ADAPTERS: dict[str, TraceAdapter] = {}
ASPECTS: dict[str, ComparisonAspect] = {}
_PLUGINS_LOADED = False

TAdapter = TypeVar("TAdapter", bound=type)
TAspect = TypeVar("TAspect", bound=type)


def register_adapter(name: str, *, version: str = "1.0") -> Callable[[TAdapter], TAdapter]:
    def decorator(cls: TAdapter) -> TAdapter:
        instance = cls()
        instance.format_name = name
        instance.version = getattr(instance, "version", version)
        ADAPTERS[name] = instance
        return cls

    return decorator


def register_aspect(
    name: str,
    *,
    title: str,
    required_formats: tuple[str, ...],
    any_of_formats: bool = False,
) -> Callable[[TAspect], TAspect]:
    def decorator(cls: TAspect) -> TAspect:
        instance = cls()
        instance.name = name
        instance.title = title
        instance.required_formats = required_formats
        instance.any_of_formats = any_of_formats
        ASPECTS[name] = instance
        return cls

    return decorator


def load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    import cta.adapters  # noqa: F401
    import cta.aspects  # noqa: F401

    for ep in entry_points(group="cta.plugins"):
        ep.load()
