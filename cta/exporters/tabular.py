"""Exports a report as JSON and CSV, or as a bundle of several formats."""

from __future__ import annotations

from collections.abc import Iterable
import contextlib
import datetime as datetime_module
import json
import pathlib
import subprocess
from typing import Any

from cta.core import models
from cta.exporters import latex
from cta.exporters import markdown

# "tornado" is an alias of "topk", so it is not exported separately.
PLOT_KINDS = (
    "breakdown",
    "topk",
    "traffic_delta",
    "skew",
    "roofline",
    "scaling",
)


def _version() -> str:
    """Returns the installed CTA version (imported lazily)."""
    import cta

    return cta.__version__


def _git_commit() -> str | None:
    """Returns the short commit hash of the working directory, if any."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def report_to_dict(report: models.ComparisonReport) -> dict[str, Any]:
    """Returns a JSON-serialisable view of a report.

    ``aspects`` keeps the legacy summary-only shape; ``details`` and
    ``scalars`` hold the rest, for the aspects that have any.

    Args:
        report: The report to convert.

    Returns:
        The report as plain dicts and lists, with the creation time, CTA
        version and git commit.
    """
    # The creation time lets `cta plot --kind history` order saved runs.
    created_at = datetime_module.datetime.now(datetime_module.UTC).isoformat(
        timespec="seconds"
    )
    return {
        "created_at": created_at,
        "cta_version": _version(),
        "git_commit": _git_commit(),
        "mode": report.mode,
        "groups": [group.label for group in report.groups],
        "diagnoses": [shift.summary for shift in report.diagnoses],
        "violations": [violation.message for violation in report.violations],
        "aspects": {
            name: result.summary.to_dicts()
            for name, result in report.aspects.items()
        },
        "details": {
            name: {
                table_name: table.to_dicts()
                for table_name, table in result.details.items()
            }
            for name, result in report.aspects.items()
            if result.details
        },
        "scalars": {
            name: result.scalars
            for name, result in report.aspects.items()
            if result.scalars
        },
    }


def report_to_json(report: models.ComparisonReport) -> str:
    """Returns a report as indented JSON (see ``report_to_dict``).

    Args:
        report: The report to convert.

    Returns:
        The JSON text; values JSON cannot represent are converted with
        ``str``.
    """
    return json.dumps(report_to_dict(report), indent=2, default=str)


def export_bundle(
    report: models.ComparisonReport,
    out_dir: pathlib.Path,
    *,
    formats: Iterable[str] = ("markdown", "csv", "json"),
) -> dict[str, pathlib.Path]:
    """Writes a report in several formats into one directory.

    Supported formats (case-insensitive): ``markdown``/``md``
    (``report.md``), ``latex``/``tex`` (``report.tex``), ``csv`` (one
    ``<aspect>_summary.csv`` per non-empty summary), ``json``
    (``report.json``), ``html`` (``dashboard.html``; skipped when plotly is
    missing or nothing can be drawn) and ``pdf``/``svg``/``png`` (one
    figure per plot kind in ``PLOT_KINDS`` that applies). Unknown formats
    are ignored.

    Args:
        report: The report to export.
        out_dir: Output directory; created if missing.
        formats: The formats to write.

    Returns:
        The written files, keyed by ``markdown``, ``latex``, ``json``,
        ``html``, ``csv:<aspect>`` or ``<extension>:<plot kind>``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}
    requested = {output_format.lower() for output_format in formats}

    if "markdown" in requested or "md" in requested:
        markdown_path = out_dir / "report.md"
        markdown.export_markdown(report, markdown_path)
        written["markdown"] = markdown_path

    if "latex" in requested or "tex" in requested:
        latex_path = out_dir / "report.tex"
        latex.export_latex(report, latex_path)
        written["latex"] = latex_path

    if "csv" in requested:
        for name, result in report.aspects.items():
            if not result.summary.is_empty():
                csv_path = out_dir / f"{name}_summary.csv"
                result.summary.write_csv(csv_path)
                written[f"csv:{name}"] = csv_path

    if "json" in requested:
        json_path = out_dir / "report.json"
        json_path.write_text(report_to_json(report))
        written["json"] = json_path

    if "html" in requested:
        # The dashboard is optional: plotly may be missing (ImportError) or
        # nothing may be drawable (ValueError).
        with contextlib.suppress(ImportError, ValueError):
            written["html"] = report.dashboard(out_dir / "dashboard.html")

    for extension in ("pdf", "svg", "png"):
        if extension in requested:
            for kind in PLOT_KINDS:
                figure_path = out_dir / f"{kind}.{extension}"
                try:
                    report.plot(kind, path=figure_path)
                except ValueError:
                    # The plot does not apply to this report's aspects.
                    continue
                written[f"{extension}:{kind}"] = figure_path

    return written
