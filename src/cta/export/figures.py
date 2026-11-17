"""Export formats that draw plots: ``html``, ``pdf``, ``svg`` and ``png``.

``html`` writes the interactive dashboard, which needs the ``viz`` extra and
is skipped without it. The static formats draw every plot kind marked
``exported`` that applies to the report.
"""

from __future__ import annotations

import contextlib
import functools
import pathlib

from cta import models
from cta.export import writers
from cta.viz import registry


@writers.register_writer("html")
def write_dashboard(
    report: models.Report, out_dir: pathlib.Path
) -> dict[str, pathlib.Path]:
    """Writes ``dashboard.html``, unless plotly is missing or no plot fits."""
    # plotly may be missing (ImportError) or nothing drawable (ValueError).
    with contextlib.suppress(ImportError, ValueError):
        return {"html": report.dashboard(out_dir / "dashboard.html")}
    return {}


def _write_figures(
    report: models.Report, out_dir: pathlib.Path, extension: str
) -> dict[str, pathlib.Path]:
    """Writes one ``<kind>.<extension>`` per exported kind that applies."""
    written: dict[str, pathlib.Path] = {}
    for kind in registry.exported_kinds():
        figure_path = out_dir / f"{kind}.{extension}"
        try:
            report.plot(kind, path=figure_path)
        except ValueError:
            continue  # the kind does not apply to this report
        written[f"{extension}:{kind}"] = figure_path
    return written


for _extension in ("pdf", "svg", "png"):
    writers.register_writer(_extension)(
        functools.partial(_write_figures, extension=_extension)
    )
