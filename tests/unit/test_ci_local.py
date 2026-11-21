"""`ci-local.sh` runs what `.github/workflows/ci.yml` runs.

A local check that has drifted from CI says "passed" for something CI rejects,
so the commands of the script's `test` job must still be run lines of the
workflow.
"""

from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _workflow_commands(job: str) -> set[str]:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    return {
        step["run"].strip()
        for step in workflow["jobs"][job]["steps"]
        if "run" in step
    }


def test_the_test_job_commands_are_the_ones_ci_runs() -> None:
    script = (ROOT / "ci-local.sh").read_text()
    local = set(re.findall(r'"\$PY" -m (\S+(?: [^\n]+)?)', script))
    # `ruff check ...` in CI is `"$PY" -m ruff check ...` here.
    local = {c.strip() for c in local}

    ci = _workflow_commands("test")
    for command in ci:
        if command.startswith(("ruff", "mypy", "pytest")):
            assert command in local, f"ci-local.sh does not run: {command}"


def test_the_core_install_checks_are_the_ones_ci_runs() -> None:
    script = (ROOT / "ci-local.sh").read_text()
    ci = _workflow_commands("core-install")

    assert "./configure.sh" in ci and "./configure.sh" in script
    assert "pytest tests/unit/test_install_extras.py" in ci
    assert "tests/unit/test_install_extras.py" in script


def test_the_pre_push_hook_calls_the_script() -> None:
    hook = (ROOT / ".githooks/pre-push").read_text()

    assert "./ci-local.sh" in hook
    assert (ROOT / "ci-local.sh").stat().st_mode & 0o111
