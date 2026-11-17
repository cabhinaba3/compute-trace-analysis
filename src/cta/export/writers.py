"""Export formats: writers that put a report into a directory.

A writer takes a report and an output directory and returns the files it
wrote. Each format module registers its writers with ``register_writer``;
``write_report`` runs the ones named by ``--export`` or ``Report.export``.
Plugins add formats the same way (see ``cta.plugins``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import pathlib
from typing import TYPE_CHECKING

from cta import plugins

if TYPE_CHECKING:
    from cta import models

Writer = Callable[["models.Report", pathlib.Path], dict[str, pathlib.Path]]

# Built-in formats.
WRITERS: plugins.Registry[Writer] = plugins.Registry(
    "export format",
    (
        "cta.export.markdown",
        "cta.export.latex",
        "cta.export.serialize",
        "cta.export.figures",
    ),
)


def register_writer(
    name: str, *, aliases: Iterable[str] = ()
) -> Callable[[Writer], Writer]:
    """Returns a decorator that registers a writer as an export format.

    Args:
        name: Format name, as given to ``--export``.
        aliases: Other names for the format.

    Returns:
        A decorator that registers the writer and returns it unchanged.
    """

    def decorator(writer: Writer) -> Writer:
        WRITERS.register(name, writer, aliases=aliases)
        return writer

    return decorator


def resolve_formats(formats: Iterable[str]) -> list[str]:
    """Resolves format names and aliases, keeping order and dropping repeats.

    Args:
        formats: Format names or aliases, case-insensitive.

    Returns:
        The registered format names, in the order given.

    Raises:
        ValueError: A format is not registered; the message names the
            closest one.
    """
    resolved = list(dict.fromkeys(WRITERS.resolve(f.lower()) for f in formats))
    unknown = [name for name in resolved if name not in WRITERS]
    if unknown:
        hints = [hint for name in unknown if (hint := WRITERS.suggest(name))]
        raise ValueError(
            f"Unknown export format(s): {', '.join(unknown)}."
            + (f" Did you mean {', '.join(hints)}?" if hints else "")
            + f" Available: {', '.join(sorted(WRITERS))}"
        )
    return resolved


def write_report(
    report: models.Report,
    out_dir: pathlib.Path,
    formats: Iterable[str] = ("markdown", "csv", "json"),
) -> dict[str, pathlib.Path]:
    """Writes a report in several formats into one directory.

    Args:
        report: The report to write.
        out_dir: Output directory; created if missing.
        formats: Format names or aliases, case-insensitive, written in the
            order given.

    Returns:
        The written files, keyed by what each writer reports (``"json"``,
        ``"csv:<aspect>"``, ``"pdf:<plot kind>"``, ...).

    Raises:
        ValueError: A format is not registered; the message names the
            closest one.
    """
    names = resolve_formats(formats)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}
    for name in names:
        written.update(WRITERS[name](report, out_dir))
    return written
