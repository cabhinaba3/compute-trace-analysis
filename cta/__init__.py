"""CTA v1.0: Cross-platform multi-trace and multi-aspect comparison framework.

Public names are resolved lazily (PEP 562) so ``import cta`` and ``cta --version`` do not pay for polars,
scipy, matplotlib or the profiler libraries until a feature that needs them is used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"  # single source of truth: read by hatch (see [tool.hatch.version] in pyproject.toml)

# public name -> (module, attribute in that module)
_LAZY: dict[str, tuple[str, str]] = {
    "EquivalenceVerdict": ("cta.aspects.distributed", "EquivalenceVerdict"),
    "equiv": ("cta.aspects.distributed", "verify_chakra_goal_equivalence"),
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
    "equiv",
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
    from cta.aspects.distributed import verify_chakra_goal_equivalence as equiv
    from cta.core.bundle import detect_format, resolve_bundle, resolve_trial_group
    from cta.core.engine import analyze, compare, inspect_trace, run_comparison
    from cta.core.manifest import load_manifest
    from cta.core.models import (
        AspectResult,
        BottleneckShift,
        CompareConfig,
        ComparisonReport,
        RegressionViolation,
        RunBundle,
        TraceArtifact,
        TrialGroup,
    )
    from cta.core.registry import register_adapter, register_aspect


def __getattr__(name: str) -> Any:
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'cta' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), attr)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
