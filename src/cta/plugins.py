"""Extension points: registries of trace adapters and comparison aspects.

A :class:`Registry` holds the plugins of one kind under their names. It
imports its built-in modules, and then every module of the ``cta.plugins``
entry-point group, the first time it is used, so callers never load anything
by hand. Built-in and third-party plugins register the same way, with the
``register_adapter`` and ``register_aspect`` class decorators (and
``cta.viz.registry.register_plot`` for plot kinds) run at import time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import difflib
import importlib
from importlib import metadata
import pathlib
from typing import Protocol, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from cta import models

ENTRY_POINT_GROUP = "cta.plugins"

T = TypeVar("T")
TAdapter = TypeVar("TAdapter", bound=type)
TAspect = TypeVar("TAspect", bound=type)

_entry_points_loaded = False


def load_entry_points() -> None:
    """Imports every ``cta.plugins`` entry point, once per process."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
        entry_point.load()


class Registry(Mapping[str, T]):
    """Plugins of one kind, keyed by name, in registration order.

    Reading the registry (lookup, iteration, ``len``) or registering into it
    first loads the built-in modules and the entry points. Registering a name
    again replaces the earlier plugin, so a plugin can override a built-in.

    Attributes:
        kind: What the registry holds, for error messages (``"adapter"``).
    """

    def __init__(self, kind: str, builtins: Sequence[str]) -> None:
        """Initialises an empty registry.

        Args:
            kind: What the registry holds, for error messages.
            builtins: Modules that register the built-in plugins when
                imported, in the order their plugins should be listed.
        """
        self.kind = kind
        self._builtins = tuple(builtins)
        self._items: dict[str, T] = {}
        self._aliases: dict[str, str] = {}
        self._loaded = False

    def load(self) -> None:
        """Imports the built-in modules, then the entry points, once."""
        if self._loaded:
            return
        # Set first: the imported modules register into this registry.
        self._loaded = True
        for module in self._builtins:
            importlib.import_module(module)
        load_entry_points()

    def register(
        self, name: str, item: T, *, aliases: Iterable[str] = ()
    ) -> None:
        """Adds a plugin under a name and optional aliases.

        Args:
            name: Name of the plugin.
            item: The plugin.
            aliases: Other names that resolve to ``name``.
        """
        self.load()
        self._items[name] = item
        for alias in aliases:
            self._aliases[alias] = name

    def resolve(self, name: str) -> str:
        """Returns the plugin name an alias stands for (or ``name`` itself).

        Args:
            name: A plugin name or alias.

        Returns:
            The registered name; unknown names are returned unchanged.
        """
        self.load()
        return self._aliases.get(name, name)

    def suggest(self, name: str) -> str | None:
        """Returns the registered name or alias closest to ``name``.

        Args:
            name: An unknown name.

        Returns:
            The closest known name or alias, or None if none is close.
        """
        self.load()
        matches = difflib.get_close_matches(
            name, [*self._items, *self._aliases], n=1
        )
        return matches[0] if matches else None

    def __getitem__(self, name: str) -> T:
        """Returns the plugin registered under a name or alias.

        Raises:
            KeyError: Nothing is registered under ``name``; the message names
                the closest match.
        """
        key = self.resolve(name)
        if key not in self._items:
            hint = self.suggest(name)
            raise KeyError(
                f"Unknown {self.kind} {name!r}."
                + (f" Did you mean {hint!r}?" if hint else "")
            )
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        """Iterates over the registered names in registration order."""
        self.load()
        return iter(list(self._items))

    def __len__(self) -> int:
        """Returns the number of registered plugins."""
        self.load()
        return len(self._items)


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


class Aspect(Protocol):
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
        config: models.AnalysisConfig,
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


# Built-in adapters, in detection order: the first adapter whose ``detect``
# accepts a file decides its format.
ADAPTERS: Registry[TraceAdapter] = Registry(
    "trace format",
    (
        "cta.adapters.chakra.reader",
        "cta.adapters.goal.reader",
        "cta.adapters.jax.reader",
        "cta.adapters.jtop.reader",
        "cta.adapters.kineto.reader",
        "cta.adapters.ncu.reader",
        "cta.adapters.nsys.reader",
        "cta.adapters.trtexec.reader",
    ),
)

# Built-in aspects, in report order.
ASPECTS: Registry[Aspect] = Registry(
    "aspect",
    (
        "cta.aspects.device.latency",
        "cta.aspects.device.kernels",
        "cta.aspects.device.overlap",
        "cta.aspects.device.os_runtime",
        "cta.aspects.device.roofline",
        "cta.aspects.distributed.critical_path",
        "cta.aspects.distributed.collectives",
        "cta.aspects.distributed.rank_skew",
        "cta.aspects.distributed.traffic",
        "cta.aspects.distributed.equivalence",
        "cta.aspects.edge.layer_fusion",
        "cta.aspects.edge.energy",
        "cta.aspects.hta",
        "cta.aspects.network",
        "cta.aspects.whatif",
    ),
)


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
        ADAPTERS.register(name, instance)
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
        ASPECTS.register(name, instance)
        return cls

    return decorator
