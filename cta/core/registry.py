"""Plugin registry for trace adapters and comparison aspects.

Adapters and aspects register themselves with the ``register_adapter`` and
``register_aspect`` class decorators when their module is imported.
``load_plugins`` imports the built-in ones and every ``cta.plugins`` entry
point.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
import pathlib
from typing import Protocol, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from cta.core import models


class TraceAdapter(Protocol):
    """Reads one trace format into Parquet tables.

    Attributes:
        format_name: Name of the format, set at registration.
        version: Adapter version; part of the cache key, so bumping it
            invalidates cached tables.
    """

    format_name: str
    version: str

    def detect(self, path: pathlib.Path) -> bool:
        """Returns whether a file is a trace of this adapter's format.

        Args:
            path: The file to check.

        Returns:
            True if this adapter can ingest the file.
        """

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts an artifact's trace files into Parquet tables.

        Args:
            artifact: The trace files to read.
            out_dir: Directory to write the tables to.

        Returns:
            Table name -> path of the written Parquet file.
        """


class ComparisonAspect(Protocol):
    """Computes one family of metrics across trial groups.

    Attributes:
        name: Registered name of the aspect.
        title: Human-readable title.
        required_formats: Trace formats the aspect needs.
        any_of_formats: If True, the aspect needs at least one of
            ``required_formats`` rather than all of them.
        default_enabled: If False, the aspect only runs when requested
            explicitly via ``aspects=`` / ``--aspects``.
    """

    name: str
    title: str
    required_formats: tuple[str, ...]
    any_of_formats: bool
    default_enabled: bool

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Computes the aspect's metrics for all groups.

        Args:
            groups: The groups to compare; the first one is the baseline.
            tables: Group id -> one map of table name -> Parquet path per
                trial.
            config: The run's options.

        Returns:
            The aspect's result.
        """


ADAPTERS: dict[str, TraceAdapter] = {}
ASPECTS: dict[str, ComparisonAspect] = {}
_plugins_loaded = False

TAdapter = TypeVar("TAdapter", bound=type)
TAspect = TypeVar("TAspect", bound=type)


def register_adapter(
    name: str, *, version: str = "1.0"
) -> Callable[[TAdapter], TAdapter]:
    """Returns a class decorator that registers a trace adapter.

    The decorator instantiates the class once, sets the instance's
    ``format_name`` and stores it in ``ADAPTERS`` under ``name``.

    Args:
        name: Format name to register the adapter under.
        version: Adapter version, used only when the class does not define
            a ``version`` attribute itself.

    Returns:
        The decorator; it returns the class unchanged.
    """

    def decorator(cls: TAdapter) -> TAdapter:
        """Instantiates and registers ``cls``; returns it unchanged."""
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
    default_enabled: bool = True,
) -> Callable[[TAspect], TAspect]:
    """Returns a class decorator that registers a comparison aspect.

    The decorator instantiates the class once, sets the given attributes on
    the instance and stores it in ``ASPECTS`` under ``name``.

    Args:
        name: Name to register the aspect under.
        title: Human-readable title.
        required_formats: Trace formats the aspect needs.
        any_of_formats: If True, the aspect needs at least one of
            ``required_formats`` rather than all of them.
        default_enabled: If False, the aspect only runs when requested.

    Returns:
        The decorator; it returns the class unchanged.
    """

    def decorator(cls: TAspect) -> TAspect:
        """Instantiates and registers ``cls``; returns it unchanged."""
        instance = cls()
        instance.name = name
        instance.title = title
        instance.required_formats = required_formats
        instance.any_of_formats = any_of_formats
        instance.default_enabled = default_enabled
        ASPECTS[name] = instance
        return cls

    return decorator


def load_plugins() -> None:
    """Imports the built-in and entry-point plugins, once per process.

    Importing ``cta.adapters`` and ``cta.aspects`` registers the built-in
    adapters and aspects; then every ``cta.plugins`` entry point is loaded.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    import cta.adapters
    import cta.aspects  # noqa: F401

    for entry_point in metadata.entry_points(group="cta.plugins"):
        entry_point.load()
