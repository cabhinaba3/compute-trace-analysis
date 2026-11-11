"""CTA v1.0: Cross-platform multi-trace and multi-aspect comparison framework."""

from cta.core.bundle import detect_format, resolve_bundle, resolve_trial_group
from cta.core.engine import compare, inspect_trace, run_comparison
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

__version__ = "1.0.0"

__all__ = [
    "AspectResult",
    "BottleneckShift",
    "CompareConfig",
    "ComparisonReport",
    "RegressionViolation",
    "RunBundle",
    "TraceArtifact",
    "TrialGroup",
    "__version__",
    "compare",
    "detect_format",
    "inspect_trace",
    "load_manifest",
    "register_adapter",
    "register_aspect",
    "resolve_bundle",
    "resolve_trial_group",
    "run_comparison",
]
