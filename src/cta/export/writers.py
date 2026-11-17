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

# Built-in formats, in the order ``write_report`` writes them.
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


def write_report(
    report: models.Report,
    out_dir: pathlib.Path,
    formats: Iterable[str] = ("markdown", "csv", "json"),
) -> dict[str, pathlib.Path]:
    """Writes a report in several formats into one directory.

    Args:
        report: The report to write.
        out_dir: Output directory; created if missing.
        formats: Format names or aliases, case-insensitive.

    Returns:
        The written files, keyed by what each writer reports (``"json"``,
        ``"csv:<aspect>"``, ``"pdf:<plot kind>"``, ...).

    Raises:
        ValueError: A format is not registered; the message names the
            closest one.
    """
    requested = {WRITERS.resolve(name.lower()) for name in formats}
    unknown = sorted(name for name in requested if name not in WRITERS)
    if unknown:
        hints = [WRITERS.suggest(name) for name in unknown]
        raise ValueError(
            f"Unknown export format(s) {unknown}"
            + (
                f" (did you mean {', '.join(h for h in hints if h)}?)"
                if any(hints)
                else ""
            )
            + f". Available: {', '.join(WRITERS)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}
    for name, writer in WRITERS.items():
        if name in requested:
            written.update(writer(report, out_dir))
    return written
