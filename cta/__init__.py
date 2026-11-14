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
    "detect_format": ("cta.core.bundle", "detect_format"),
    "resolve_bundle": ("cta.core.bundle", "resolve_bundle"),
    "resolve_trial_group": ("cta.core.bundle", "resolve_trial_group"),
    "analyze": ("cta.core.engine", "analyze"),
    "compare": ("cta.core.engine", "compare"),
    "inspect_trace": ("cta.core.engine", "inspect_trace"),
    "run_comparison": ("cta.core.engine", "run_comparison"),
    "load_manifest": ("cta.core.manifest", "load_manifest"),
    "AspectResult": ("cta.core.models", "AspectResult"),
    "BottleneckShift": ("cta.core.models", "BottleneckShift"),
    "CompareConfig": ("cta.core.models", "CompareConfig"),
    "ComparisonReport": ("cta.core.models", "ComparisonReport"),
    "RegressionViolation": ("cta.core.models", "RegressionViolation"),
    "RunBundle": ("cta.core.models", "RunBundle"),
    "TraceArtifact": ("cta.core.models", "TraceArtifact"),
    "TrialGroup": ("cta.core.models", "TrialGroup"),
    "register_adapter": ("cta.core.registry", "register_adapter"),
    "register_aspect": ("cta.core.registry", "register_aspect"),
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
    from cta.core.bundle import detect_format
    from cta.core.bundle import resolve_bundle
    from cta.core.bundle import resolve_trial_group
    from cta.core.engine import analyze
    from cta.core.engine import compare
    from cta.core.engine import inspect_trace
    from cta.core.engine import run_comparison
    from cta.core.manifest import load_manifest
    from cta.core.models import AspectResult
    from cta.core.models import BottleneckShift
    from cta.core.models import CompareConfig
    from cta.core.models import ComparisonReport
    from cta.core.models import RegressionViolation
    from cta.core.models import RunBundle
    from cta.core.models import TraceArtifact
    from cta.core.models import TrialGroup
    from cta.core.registry import register_adapter
    from cta.core.registry import register_aspect


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
