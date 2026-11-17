"""Plot kind registry and typed plot options.

A kind is a function ``build(ctx, opts) -> Chart`` registered with
:func:`register_plot`. Its options are declared (type, default, help), which
is what powers ``cta plot --list-kinds``, option coercion from CLI strings and
the rejection of unknown spec keys (with a suggestion), the same way an
unknown gate metric is an error. Plugins register kinds the same way; see
``cta.plugins``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import dataclasses
import difflib
from typing import Any, TYPE_CHECKING

from cta import plugins

if TYPE_CHECKING:
    from cta.viz import context
    from cta.viz import ir


class NotApplicableError(ValueError):
    """The plot cannot be drawn for this report (missing aspect or source).

    Subclasses ValueError, so callers that catch ValueError catch it too.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class Opt:
    """A declared plot option.

    Attributes:
        value_type: One of ``str``, ``int``, ``float``, ``bool``, ``list``
            (of str) or ``range`` (``"a:b"`` or ``[a, b]``).
        default: Value used when the option is not given.
        help_text: One-line description of the option.
        choices: Allowed values; None allows any value.
    """

    value_type: str
    default: Any = None
    help_text: str = ""
    choices: tuple[str, ...] | None = None


COMMON_OPTIONS: dict[str, Opt] = {
    "title": Opt("str", None, "Figure title (default: the kind's own)."),
    "theme": Opt("str", "light", "light or dark.", ("light", "dark")),
    "palette": Opt(
        "str",
        None,
        "okabe_ito, tableau, pastel, or a comma-separated list of colours.",
    ),
    "width": Opt("float", None, "Figure width in inches (static output)."),
    "height": Opt("float", None, "Figure height in inches (static output)."),
    "labels": Opt("list", None, "Only these group labels."),
    "rank": Opt("list", None, "Only these ranks (raw-table plots)."),
    "device": Opt("list", None, "Only these GPU devices (raw-table plots)."),
    "steps": Opt(
        "range",
        None,
        "Only steps in [a:b) after warm-up slicing, e.g. 2:10 "
        "(raw-table plots).",
    ),
    "where": Opt(
        "str",
        None,
        "SQL predicate applied to the raw table, e.g. "
        "\"category = 'comm' AND duration_ms > 0.1\".",
    ),
}

Builder = Callable[["context.DataContext", dict[str, Any]], "ir.Chart"]


@dataclasses.dataclass(slots=True)
class PlotKind:
    """A registered plot kind.

    Attributes:
        name: Kind name, as given to ``cta plot --kind``.
        group: Dashboard tab the kind belongs to, one of ``GROUPS``.
        title: Human-readable title.
        build: Function that builds the chart from a data context and the
            resolved options.
        options: The kind's own declared options (``COMMON_OPTIONS`` come on
            top).
        aliases: Other names that resolve to this kind.
        needs: Human-readable data requirement, shown by ``--list-kinds``.
        exported: True if ``--export pdf|svg|png`` draws the kind.
    """

    name: str
    group: str
    title: str
    build: Builder
    options: dict[str, Opt] = dataclasses.field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    needs: str = ""
    exported: bool = False

    @property
    def doc(self) -> str:
        """The first line of the builder's docstring (``""`` if it has none)."""
        docstring = self.build.__doc__
        if not docstring:
            return ""
        return docstring.strip().splitlines()[0]

    def all_options(self) -> dict[str, Opt]:
        """Returns the common options merged with the kind's own options.

        Returns:
            Option name -> declaration; the kind's own declarations override
            common ones of the same name.
        """
        return {**COMMON_OPTIONS, **self.options}


GROUPS = (
    "overview",
    "timeline",
    "steps",
    "compare",
    "sweep",
    "distributed",
    "kernels",
    "edge",
    "drift",
    "custom",
)

# Built-in plot kinds; ``kind_catalog`` sorts them by group and name.
PLOT_KINDS: plugins.Registry[PlotKind] = plugins.Registry(
    "plot kind",
    tuple(
        f"cta.viz.kinds.{module}"
        for module in (
            "comparisons",
            "concurrency",
            "custom",
            "distributed",
            "distributions",
            "effects",
            "facet",
            "heatmaps",
            "history",
            "host_api",
            "kernels",
            "memory",
            "ncu",
            "network",
            "scopes",
            "series",
            "steps",
            "summary",
            "sweeps",
            "telemetry",
            "timeline",
        )
    ),
)


def register_plot(
    name: str,
    *,
    group: str,
    title: str,
    options: Mapping[str, Opt] | None = None,
    aliases: Iterable[str] = (),
    needs: str = "",
    exported: bool = False,
) -> Callable[[Builder], Builder]:
    """Returns a decorator that registers a chart builder as a plot kind.

    Args:
        name: Kind name; registering a name again replaces the earlier kind.
        group: Dashboard tab of the kind, one of ``GROUPS``.
        title: Human-readable title.
        options: The kind's own declared options.
        aliases: Other names that resolve to this kind.
        needs: Human-readable data requirement, shown by ``--list-kinds``.
        exported: True if ``--export pdf|svg|png`` draws the kind.

    Returns:
        A decorator that registers the builder and returns it unchanged. It
        raises ValueError if ``group`` is not one of ``GROUPS``.
    """

    def decorator(builder: Builder) -> Builder:
        if group not in GROUPS:
            raise ValueError(
                f"plot group must be one of {GROUPS}, got {group!r}"
            )
        PLOT_KINDS.register(
            name,
            PlotKind(
                name,
                group,
                title,
                builder,
                dict(options or {}),
                tuple(aliases),
                needs,
                exported,
            ),
            aliases=aliases,
        )
        return builder

    return decorator


def get_kind(name: str) -> PlotKind:
    """Looks up a plot kind by name or alias, ignoring case.

    Args:
        name: Kind name or alias.

    Returns:
        The registered plot kind.

    Raises:
        ValueError: If no kind has that name or alias; the message suggests
            the closest match.
    """
    try:
        return PLOT_KINDS[name.lower()]
    except KeyError:
        hint = PLOT_KINDS.suggest(name.lower())
        raise ValueError(
            f"Unknown plot kind '{name}'."
            + (f" Did you mean '{hint}'?" if hint else "")
            + f" Supported: {', '.join(sorted(PLOT_KINDS))}"
        ) from None


def _coerce(name: str, opt: Opt, value: Any) -> Any:
    """Converts a given option value (often a CLI string) to its declared type.

    Args:
        name: Option name, for error messages.
        opt: The option declaration.
        value: The value as given; None stays None.

    Returns:
        The converted value.

    Raises:
        ValueError: If the value cannot be read as the declared type or is
            not one of the declared choices.
    """
    if value is None:
        return None
    option_type = opt.value_type
    try:
        if option_type == "int":
            return int(value)
        if option_type == "float":
            return float(value)
        if option_type == "bool":
            if isinstance(value, str):
                if value.lower() in ("1", "true", "yes", "on"):
                    return True
                if value.lower() in ("0", "false", "no", "off"):
                    return False
                raise ValueError(value)
            return bool(value)
        if option_type == "list":
            if isinstance(value, str):
                return [
                    item.strip() for item in value.split(",") if item.strip()
                ]
            return (
                list(value)
                if isinstance(value, (list, tuple, set))
                else [value]
            )
        if option_type == "range":
            if isinstance(value, str):
                start_text, _, end_text = value.partition(":")
                return (
                    int(start_text) if start_text else None,
                    int(end_text) if end_text else None,
                )
            start, end = value
            return (
                None if start is None else int(start),
                None if end is None else int(end),
            )
        text = str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"option '{name}': cannot read {value!r} as {option_type}"
        ) from exc
    if opt.choices and text not in opt.choices:
        raise ValueError(
            f"option '{name}': {text!r} is not one of {', '.join(opt.choices)}"
        )
    return text


def resolve_options(kind: PlotKind, given: Mapping[str, Any]) -> dict[str, Any]:
    """Applies defaults and type coercion to the options given for a kind.

    Args:
        kind: The plot kind.
        given: Option values as given (from a spec file or the CLI).

    Returns:
        Every declared option (common and kind-specific) -> its coerced value,
        or its default when not given.

    Raises:
        ValueError: If an option is not declared (the message names the
            closest valid option) or a value cannot be coerced.
    """
    declared = kind.all_options()
    unknown = [key for key in given if key not in declared]
    if unknown:
        problems = []
        for key in unknown:
            hint = difflib.get_close_matches(key, declared, n=1)
            problems.append(
                f"'{key}'" + (f" (did you mean '{hint[0]}'?)" if hint else "")
            )
        raise ValueError(
            f"Unknown option(s) for plot kind '{kind.name}': "
            f"{', '.join(problems)}. Valid: {', '.join(sorted(declared))}"
        )
    return {
        key: _coerce(key, option, given[key])
        if key in given
        else option.default
        for key, option in declared.items()
    }


def kind_catalog() -> list[PlotKind]:
    """Returns every registered plot kind, ordered by group, then name.

    Returns:
        The plot kinds in ``GROUPS`` order, sorted by name within a group.
    """
    return sorted(
        PLOT_KINDS.values(),
        key=lambda kind: (GROUPS.index(kind.group), kind.name),
    )


def exported_kinds() -> list[str]:
    """Returns the names of the kinds ``--export pdf|svg|png`` draws.

    Returns:
        The exported kinds, in ``kind_catalog`` order.
    """
    return [kind.name for kind in kind_catalog() if kind.exported]
