"""CTA v1.0: Cross-platform multi-trace and multi-aspect comparison framework.

Public names are resolved lazily (PEP 562) so ``import cta`` and
``cta --version`` do not pay for polars, scipy, matplotlib or the profiler
libraries until a feature that needs them is used.
"""

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

# Single source of truth: read by hatch (see [tool.hatch.version] in
# pyproject.toml).
__version__ = "1.0.0"

# Public name -> (module, attribute in that module).
_LAZY: dict[str, tuple[str, str]] = {
    "EquivalenceVerdict": ("cta.aspects.distributed", "EquivalenceVerdict"),
    "verify_chakra_goal_equivalence": (
        "cta.aspects.distributed",
        "verify_chakra_goal_equivalence",
    ),
    "detect_format": ("cta.ingest.discovery", "detect_format"),
    "resolve_bundle": ("cta.ingest.discovery", "resolve_bundle"),
    "resolve_trial_group": ("cta.ingest.discovery", "resolve_trial_group"),
    "analyze": ("cta.engine", "analyze"),
    "compare": ("cta.engine", "compare"),
    "inspect_trace": ("cta.engine", "inspect_trace"),
    "run_comparison": ("cta.engine", "run_comparison"),
    "load_manifest": ("cta.ingest.manifest", "load_manifest"),
    "AspectResult": ("cta.models", "AspectResult"),
    "BottleneckShift": ("cta.models", "BottleneckShift"),
    "CompareConfig": ("cta.models", "CompareConfig"),
    "ComparisonReport": ("cta.models", "ComparisonReport"),
    "RegressionViolation": ("cta.models", "RegressionViolation"),
    "RunBundle": ("cta.models", "RunBundle"),
    "TraceArtifact": ("cta.models", "TraceArtifact"),
    "TrialGroup": ("cta.models", "TrialGroup"),
    "register_adapter": ("cta.plugins", "register_adapter"),
    "register_aspect": ("cta.plugins", "register_aspect"),
}

__all__ = [
    "__version__",
    "AspectResult",
    "BottleneckShift",
    "CompareConfig",
    "ComparisonReport",
    "EquivalenceVerdict",
    "RegressionViolation",
    "RunBundle",
    "TraceArtifact",
    "TrialGroup",
    "analyze",
    "compare",
    "detect_format",
    "verify_chakra_goal_equivalence",
    "inspect_trace",
    "load_manifest",
    "register_adapter",
    "register_aspect",
    "resolve_bundle",
    "resolve_trial_group",
    "run_comparison",
]

if TYPE_CHECKING:
    from cta.aspects.distributed import EquivalenceVerdict
    from cta.aspects.distributed import verify_chakra_goal_equivalence
    from cta.engine import analyze
    from cta.engine import compare
    from cta.engine import inspect_trace
    from cta.engine import run_comparison
    from cta.ingest.discovery import detect_format
    from cta.ingest.discovery import resolve_bundle
    from cta.ingest.discovery import resolve_trial_group
    from cta.ingest.manifest import load_manifest
    from cta.models import AspectResult
    from cta.models import BottleneckShift
    from cta.models import CompareConfig
    from cta.models import ComparisonReport
    from cta.models import RegressionViolation
    from cta.models import RunBundle
    from cta.models import TraceArtifact
    from cta.models import TrialGroup
    from cta.plugins import register_adapter
    from cta.plugins import register_aspect


def __getattr__(name: str) -> Any:
    """Imports and returns a public name on first access (PEP 562).

    Args:
        name: The attribute looked up on the ``cta`` module.

    Returns:
        The object that ``_LAZY`` maps ``name`` to.

    Raises:
        AttributeError: ``name`` is not a public name of ``cta``.
    """
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f"module 'cta' has no attribute {name!r}"
        ) from None
    value = getattr(importlib.import_module(module), attr)
    # Cache the value so __getattr__ runs once per name.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Returns the module's names, including lazy ones not yet imported."""
    return sorted([*globals(), *_LAZY])
