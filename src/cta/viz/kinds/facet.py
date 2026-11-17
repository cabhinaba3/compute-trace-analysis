"""Plot kind facet: any raw-table kind drawn once per rank, device or group."""

from __future__ import annotations

from typing import Any

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz.kinds import chart_helpers

_FACETABLE = (
    "metric_trace",
    "metric_dist",
    "violin",
    "boxplot",
    "category_compare",
    "queue_depth",
    "memcpy_bandwidth",
    "memcpy_sizes",
    "idle_gaps",
    "ridgeline",
    "sync_waits",
    "runtime_top",
    "network_bandwidth",
    "kernel_scatter",
)


def _parse_inner(text: str) -> dict[str, str]:
    """Reads ``key=value,key=value`` into a dict of strings."""
    pairs = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, separator, value = part.partition("=")
        if not separator:
            raise ValueError(f"inner option '{part}' needs key=value")
        pairs[key.strip()] = value.strip()
    return pairs


@registry.register_plot(
    "facet",
    group="overview",
    title="Any raw-table plot, one panel per rank, device or group",
    needs="a raw-table plot kind and a report",
    aliases=("small_multiples", "facets"),
    options={
        "kind": registry.Opt(
            "str",
            "metric_dist",
            "Plot kind to repeat.",
            _FACETABLE,
        ),
        "by": registry.Opt(
            "str",
            "rank",
            "What to split on.",
            ("rank", "device", "label"),
        ),
        "inner": registry.Opt(
            "str",
            "",
            "Options of the repeated kind as key=value,key=value.",
        ),
        "max_panels": registry.Opt("int", 8, "Most panels to draw."),
        "columns": registry.Opt("int", 4, "Panels per row."),
    },
)
def facet(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Repeats a plot kind once per rank, device or group.

    Each panel is the repeated kind drawn from only that slice of the data,
    so the panels are directly comparable. Only the first panel of a kind
    that draws several is kept.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A grid with one panel per value.

    Raises:
        NotApplicableError: The split column has no value, or the
            repeated kind cannot be drawn from any slice.
        ValueError: ``inner`` is malformed or names an option the repeated
            kind lacks.
    """
    inner_kind = registry.get_kind(options["kind"])
    kernels = ctx.frame("kernels", options)
    by = options["by"]
    column = {"rank": "rank", "device": "device", "label": "label"}[by]
    values = sorted(kernels[column].unique().to_list())[
        : max(1, options["max_panels"])
    ]
    if not values:
        raise registry.NotApplicableError(f"the trace has no {by} values")
    given = _parse_inner(options["inner"])
    shared = {
        key: options[key] for key in ("labels", "rank", "device", "steps")
    }
    panels, skipped = [], []
    for value in values:
        narrowed = {
            **{k: v for k, v in shared.items() if v is not None},
            **given,
            {"rank": "rank", "device": "device", "label": "labels"}[by]: (
                [value]
            ),
        }
        resolved = registry.resolve_options(inner_kind, narrowed)
        try:
            chart = inner_kind.build(ctx, resolved)
        except registry.NotApplicableError as error:
            skipped.append(f"{by} {value}: {error}")
            continue
        panel = chart.panels[0]
        panel.title = f"{by} {value}"
        panels.append(panel)
    if not panels:
        raise registry.NotApplicableError(
            f"{inner_kind.name} cannot be drawn from any {by}: "
            + "; ".join(skipped)
        )
    return chart_helpers.grid(
        f"{inner_kind.name} by {by}",
        panels,
        max_cols=max(1, options["columns"]),
        notes="; ".join(skipped),
        meta=chart_helpers.SKIP_SUITABILITY,
    )
