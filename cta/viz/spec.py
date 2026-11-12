"""Plot kind registry, typed options and declarative plot specs.

A kind is a function ``build(ctx, opts) -> Chart`` registered with :func:`register_plot`. Its options are declared
(type, default, help), which is what powers ``cta plot --list-kinds``, option coercion from CLI strings and the
rejection of unknown spec keys (with a suggestion), the same way an unknown gate metric is an error.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from cta.viz.data import DataContext
    from cta.viz.ir import Chart


class NotApplicable(ValueError):
    """The plot cannot be drawn for this report (missing aspect/source). Subclasses ValueError for the legacy API."""


@dataclass(frozen=True, slots=True)
class Opt:
    """A declared option. ``type`` is one of str, int, float, bool, list (of str), range ("a:b" or [a, b])."""

    type: str
    default: Any = None
    help: str = ""
    choices: tuple[str, ...] | None = None


COMMON_OPTIONS: dict[str, Opt] = {
    "title": Opt("str", None, "Figure title (default: the kind's own)."),
    "theme": Opt("str", "light", "light or dark.", ("light", "dark")),
    "palette": Opt("str", None, "okabe_ito, tableau, pastel, or a comma-separated list of colours."),
    "width": Opt("float", None, "Figure width in inches (static output)."),
    "height": Opt("float", None, "Figure height in inches (static output)."),
    "labels": Opt("list", None, "Only these group labels."),
    "rank": Opt("list", None, "Only these ranks (raw-table plots)."),
    "device": Opt("list", None, "Only these GPU devices (raw-table plots)."),
    "steps": Opt("range", None, "Only steps in [a:b) after warm-up slicing, e.g. 2:10 (raw-table plots)."),
    "where": Opt("str", None, "SQL predicate applied to the raw table, e.g. \"category = 'comm' AND duration_ms > 0.1\"."),
}

Builder = Callable[["DataContext", dict[str, Any]], "Chart"]


@dataclass(slots=True)
class PlotKind:
    name: str
    group: str
    title: str
    build: Builder
    options: dict[str, Opt] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    needs: str = ""  # human-readable data requirement, shown by --list-kinds
    legacy: bool = False  # one of the original six kinds (exported by `--export pdf|svg|png`)

    @property
    def doc(self) -> str:
        return (self.build.__doc__ or "").strip().splitlines()[0] if self.build.__doc__ else ""

    def all_options(self) -> dict[str, Opt]:
        return {**COMMON_OPTIONS, **self.options}


PLOT_REGISTRY: dict[str, PlotKind] = {}
_ALIASES: dict[str, str] = {}
GROUPS = ("overview", "timeline", "steps", "compare", "sweep", "distributed", "kernels", "edge", "drift", "custom")


def register_plot(
    name: str,
    *,
    group: str,
    title: str,
    options: Mapping[str, Opt] | None = None,
    aliases: Iterable[str] = (),
    needs: str = "",
    legacy: bool = False,
) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        if group not in GROUPS:
            raise ValueError(f"plot group must be one of {GROUPS}, got {group!r}")
        PLOT_REGISTRY[name] = PlotKind(name, group, title, fn, dict(options or {}), tuple(aliases), needs, legacy)
        for a in aliases:
            _ALIASES[a] = name
        return fn

    return deco


def load_kinds() -> None:
    import cta.viz.kinds  # noqa: F401  (registers the built-in kinds)


def get_kind(name: str) -> PlotKind:
    load_kinds()
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in PLOT_REGISTRY:
        known = sorted({*PLOT_REGISTRY, *_ALIASES})
        hint = difflib.get_close_matches(name.lower(), known, n=1)
        raise ValueError(
            f"Unknown plot kind '{name}'." + (f" Did you mean '{hint[0]}'?" if hint else "") + f" Supported: {', '.join(sorted(PLOT_REGISTRY))}"
        )
    return PLOT_REGISTRY[key]


def _coerce(name: str, opt: Opt, value: Any) -> Any:
    if value is None:
        return None
    t = opt.type
    try:
        if t == "int":
            return int(value)
        if t == "float":
            return float(value)
        if t == "bool":
            if isinstance(value, str):
                if value.lower() in ("1", "true", "yes", "on"):
                    return True
                if value.lower() in ("0", "false", "no", "off"):
                    return False
                raise ValueError(value)
            return bool(value)
        if t == "list":
            if isinstance(value, str):
                return [v.strip() for v in value.split(",") if v.strip()]
            return list(value) if isinstance(value, (list, tuple, set)) else [value]
        if t == "range":
            if isinstance(value, str):
                a, _, b = value.partition(":")
                return (int(a) if a else None, int(b) if b else None)
            lo, hi = value
            return (None if lo is None else int(lo), None if hi is None else int(hi))
        out = str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"option '{name}': cannot read {value!r} as {t}") from exc
    if opt.choices and out not in opt.choices:
        raise ValueError(f"option '{name}': {out!r} is not one of {', '.join(opt.choices)}")
    return out


def resolve_options(kind: PlotKind, given: Mapping[str, Any]) -> dict[str, Any]:
    """Defaults + coerced values; unknown keys are an error naming the closest valid option."""
    declared = kind.all_options()
    unknown = [k for k in given if k not in declared]
    if unknown:
        msgs = []
        for k in unknown:
            hint = difflib.get_close_matches(k, declared, n=1)
            msgs.append(f"'{k}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
        raise ValueError(f"Unknown option(s) for plot kind '{kind.name}': {', '.join(msgs)}. Valid: {', '.join(sorted(declared))}")
    return {k: _coerce(k, o, given[k]) if k in given else o.default for k, o in declared.items()}


@dataclass(slots=True)
class PlotSpec:
    """A declarative plot: ``kind`` plus its options (flat, as in YAML), and an optional output ``name``."""

    kind: str
    options: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PlotSpec:
        if not isinstance(raw, Mapping):
            raise ValueError(f"A plot spec must be a mapping, got {type(raw).__name__}")
        body = dict(raw)
        kind = str(body.pop("kind", "custom"))
        name = body.pop("name", None)
        spec = cls(kind=kind, options=body, name=str(name) if name else None)
        resolve_options(get_kind(kind), spec.options)  # validate now, not at render time
        return spec

    @property
    def stem(self) -> str:
        return self.name or self.kind  # as typed, so `--kind tornado` still writes tornado.<ext>


def load_specs(path: str | Path) -> list[PlotSpec]:
    """Specs from a YAML file: a list, or a mapping with a ``plots:`` list."""
    data = yaml.safe_load(Path(path).read_text())
    items = data.get("plots") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a non-empty list of plot specs (or a 'plots:' list)")
    return [PlotSpec.from_dict(it) for it in items]


def kind_catalog() -> list[PlotKind]:
    load_kinds()
    return sorted(PLOT_REGISTRY.values(), key=lambda k: (GROUPS.index(k.group), k.name))
