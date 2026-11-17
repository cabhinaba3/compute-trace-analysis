"""Serialises a report as JSON or as one CSV per aspect summary.

Registers the ``json`` and ``csv`` export formats.
"""

from __future__ import annotations

import datetime as datetime_module
import json
import pathlib
import subprocess
from typing import Any

from cta import models
from cta.export import writers


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


def report_to_dict(report: models.Report) -> dict[str, Any]:
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


def report_to_json(report: models.Report) -> str:
    """Returns a report as indented JSON (see ``report_to_dict``).

    Args:
        report: The report to convert.

    Returns:
        The JSON text; values JSON cannot represent are converted with
        ``str``.
    """
    return json.dumps(report_to_dict(report), indent=2, default=str)


@writers.register_writer("json")
def write_json(
    report: models.Report, out_dir: pathlib.Path
) -> dict[str, pathlib.Path]:
    """Writes ``report.json``."""
    json_path = out_dir / "report.json"
    json_path.write_text(report_to_json(report))
    return {"json": json_path}


@writers.register_writer("csv")
def write_csv(
    report: models.Report, out_dir: pathlib.Path
) -> dict[str, pathlib.Path]:
    """Writes one ``<aspect>_summary.csv`` per non-empty aspect summary."""
    written: dict[str, pathlib.Path] = {}
    for name, result in report.aspects.items():
        if not result.summary.is_empty():
            csv_path = out_dir / f"{name}_summary.csv"
            result.summary.write_csv(csv_path)
            written[f"csv:{name}"] = csv_path
    return written
