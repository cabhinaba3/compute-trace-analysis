"""Publication visualization suite."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cta.core.models import ComparisonReport


def render_plot(
    report: ComparisonReport,
    *,
    kind: str = "breakdown",
    path: Path | None = None,
    **kwargs: Any,
) -> Any:
    raise NotImplementedError(f"Plot '{kind}' will be registered in Wave 4")
