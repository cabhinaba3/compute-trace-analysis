"""Imports of optional dependencies that name the extra providing them.

Every optional package belongs to one extra of the ``compute-trace-analysis``
distribution, so a missing one is fixed with a single ``pip install``.
"""

from __future__ import annotations

import importlib
import importlib.util
import types

DISTRIBUTION = "compute-trace-analysis"


def install_hint(extra: str) -> str:
    """Returns the pip command that installs an extra.

    Args:
        extra: Name of the extra, e.g. ``"viz"``.

    Returns:
        ``pip install 'compute-trace-analysis[<extra>]'``.
    """
    return f"pip install '{DISTRIBUTION}[{extra}]'"


def missing(module: str, *, extra: str, purpose: str) -> ImportError:
    """Returns the error to raise when an optional module is not installed.

    Args:
        module: The missing module, e.g. ``"plotly.graph_objects"``.
        extra: The extra that installs it.
        purpose: What needs it, e.g. ``"interactive HTML plots"``.

    Returns:
        An ImportError whose message names the package and the pip command.
    """
    package = module.split(".")[0]
    return ImportError(
        f"{package} is needed for {purpose}: {install_hint(extra)}"
    )


def require(module: str, *, extra: str, purpose: str) -> types.ModuleType:
    """Imports an optional module, or says which extra installs it.

    Args:
        module: Dotted module name.
        extra: The extra that installs it.
        purpose: What needs it, for the error message.

    Returns:
        The imported module.

    Raises:
        ImportError: The module is not installed; the message names
            ``extra``.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise missing(module, extra=extra, purpose=purpose) from exc


def available(module: str) -> bool:
    """Returns whether a module could be imported, without importing it.

    Args:
        module: Dotted module name.

    Returns:
        True if the import system finds the module.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False
