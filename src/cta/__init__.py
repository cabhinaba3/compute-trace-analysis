"""CTA: multi-trace, multi-aspect performance comparison for AI/HPC runs.

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
    # Entry points.
    "analyze": ("cta.api", "analyze"),
    "compare": ("cta.api", "compare"),
    "inspect_trace": ("cta.api", "inspect_trace"),
    "run_comparison": ("cta.engine", "run_comparison"),
    "verify_chakra_goal_equivalence": (
        "cta.aspects.distributed.equivalence",
        "verify_chakra_goal_equivalence",
    ),
    # Simulation (see cta.sim).
    "simulate": ("cta.sim.engine", "simulate"),
    "compare_simulations": ("cta.sim.compare", "compare"),
    "load_sim_config": ("cta.sim.config", "load_config"),
    # Inputs.
    "detect_format": ("cta.ingest.discovery", "detect_format"),
    "resolve_bundle": ("cta.ingest.discovery", "resolve_bundle"),
    "resolve_trial_group": ("cta.ingest.discovery", "resolve_trial_group"),
    "load_manifest": ("cta.ingest.manifest", "load_manifest"),
    # Data model.
    "AnalysisConfig": ("cta.models", "AnalysisConfig"),
    "AspectResult": ("cta.models", "AspectResult"),
    "BottleneckShift": ("cta.models", "BottleneckShift"),
    "EquivalenceVerdict": (
        "cta.aspects.distributed.equivalence",
        "EquivalenceVerdict",
    ),
    "GateViolation": ("cta.models", "GateViolation"),
    "Report": ("cta.models", "Report"),
    "RunBundle": ("cta.models", "RunBundle"),
    "SimComparison": ("cta.sim.compare", "SimComparison"),
    "SimConfig": ("cta.sim.config", "SimConfig"),
    "SimResult": ("cta.sim.result", "SimResult"),
    "TraceArtifact": ("cta.models", "TraceArtifact"),
    "TrialGroup": ("cta.models", "TrialGroup"),
    # Extension points (see cta.plugins).
    "register_adapter": ("cta.plugins", "register_adapter"),
    "register_aspect": ("cta.plugins", "register_aspect"),
    "register_plot": ("cta.viz.registry", "register_plot"),
    "register_writer": ("cta.export.writers", "register_writer"),
}

__all__ = [
    "__version__",
    "analyze",
    "compare",
    "inspect_trace",
    "run_comparison",
    "verify_chakra_goal_equivalence",
    "simulate",
    "compare_simulations",
    "load_sim_config",
    "detect_format",
    "resolve_bundle",
    "resolve_trial_group",
    "load_manifest",
    "AnalysisConfig",
    "AspectResult",
    "BottleneckShift",
    "EquivalenceVerdict",
    "GateViolation",
    "Report",
    "RunBundle",
    "SimComparison",
    "SimConfig",
    "SimResult",
    "TraceArtifact",
    "TrialGroup",
    "register_adapter",
    "register_aspect",
    "register_plot",
    "register_writer",
]

if TYPE_CHECKING:
    from cta.api import analyze
    from cta.api import compare
    from cta.api import inspect_trace
    from cta.aspects.distributed.equivalence import EquivalenceVerdict
    from cta.aspects.distributed.equivalence import (
        verify_chakra_goal_equivalence,
    )
    from cta.engine import run_comparison
    from cta.export.writers import register_writer
    from cta.ingest.discovery import detect_format
    from cta.ingest.discovery import resolve_bundle
    from cta.ingest.discovery import resolve_trial_group
    from cta.ingest.manifest import load_manifest
    from cta.models import AnalysisConfig
    from cta.models import AspectResult
    from cta.models import BottleneckShift
    from cta.models import GateViolation
    from cta.models import Report
    from cta.models import RunBundle
    from cta.models import TraceArtifact
    from cta.models import TrialGroup
    from cta.plugins import register_adapter
    from cta.plugins import register_aspect
    from cta.sim.compare import compare as compare_simulations
    from cta.sim.compare import SimComparison
    from cta.sim.config import load_config as load_sim_config
    from cta.sim.config import SimConfig
    from cta.sim.engine import simulate
    from cta.sim.result import SimResult
    from cta.viz.registry import register_plot


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
