"""Google Python Style Guide checks that ruff cannot express.

Section 2.2 of the guide: import packages and modules only, never individual
classes or functions. ``typing``, ``collections.abc`` and
``typing_extensions`` are the documented exceptions. Package ``__init__``
files that exist to re-export a public API are exempt.
"""

import ast
import importlib.util
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CHECKED_DIRECTORIES = ("cta", "tests", "examples")
_NAME_IMPORT_ALLOWED = {
    "__future__",
    "collections.abc",
    "typing",
    "typing_extensions",
}
_REEXPORT_FILES = {"cta/__init__.py", "cta/viz/__init__.py"}


def _python_files() -> list[pathlib.Path]:
    """Returns the Python files whose imports are checked, sorted per directory.

    Real-data fixtures and the re-export ``__init__`` files are left out.
    """
    files = []
    for directory in _CHECKED_DIRECTORIES:
        for path in sorted((_REPO_ROOT / directory).rglob("*.py")):
            relative = path.relative_to(_REPO_ROOT).as_posix()
            if "data_real" in path.parts or relative in _REEXPORT_FILES:
                continue
            files.append(path)
    return files


def _is_module(dotted_name: str) -> bool:
    """Returns whether a dotted name is an importable module or package.

    Args:
        dotted_name: Name such as ``cta.core.models``.

    Returns:
        True if the import system finds a module or package of that name;
        False if it finds none or its parent package cannot be imported.
    """
    try:
        return importlib.util.find_spec(dotted_name) is not None
    except (ImportError, ValueError):
        return False


@pytest.mark.parametrize(
    "path",
    _python_files(),
    ids=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
)
def test_imports_name_modules_only(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        if node.module is None:
            continue
        if node.module in _NAME_IMPORT_ALLOWED:
            continue
        top_level_package = node.module.split(".")[0]
        if not _is_module(top_level_package):
            continue  # optional dependency not installed: cannot verify
        for alias in node.names:
            if not _is_module(f"{node.module}.{alias.name}"):
                offenders.append(
                    f"line {node.lineno}: from {node.module} import "
                    f"{alias.name}"
                )
    assert not offenders, "\n".join(offenders)
