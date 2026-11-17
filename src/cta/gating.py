"""Regression gates: thresholds on how much a metric may regress.

A gate maps a summary metric to the regression it may show, in percent of
the baseline. ``cta compare --gate`` fails (exit code 2) when one is crossed.
"""

from __future__ import annotations

from cta import models


def parse_gate_spec(spec: str | None) -> dict[str, float]:
    """Parses a regression gate spec such as ``"p99_latency_ms:+5%"``.

    Args:
        spec: Comma-separated ``<metric>:<pct>`` items; the ``%`` is
            optional and empty items are skipped.

    Returns:
        Metric name -> threshold in percent; empty for an empty spec.

    Raises:
        ValueError: An item has no ``:`` or its threshold is not a number.
    """
    if not spec:
        return {}
    gates: dict[str, float] = {}
    for raw_gate in spec.split(","):
        gate = raw_gate.strip()
        if not gate:
            continue
        if ":" not in gate:
            raise ValueError(
                f"Invalid gate '{gate}'. Expected '<metric>:<pct>%', "
                "e.g. 'p99_latency_ms:+5%'"
            )
        metric, threshold_text = gate.split(":", 1)
        try:
            gates[metric.strip()] = float(threshold_text.strip().rstrip("%"))
        except ValueError:
            raise ValueError(
                f"Invalid gate '{gate}': '{threshold_text.strip()}' "
                "is not a number"
            ) from None
    return gates


def missing_gate_metrics(
    aspects: dict[str, models.AspectResult], gates: dict[str, float]
) -> list[str]:
    """Returns the gate metrics that no aspect reports.

    Such a metric is usually a typo, or its aspect did not run.

    Args:
        aspects: Aspect results keyed by aspect name.
        gates: Regression gates keyed by metric name.

    Returns:
        The unreported gate metrics, sorted.
    """
    known = {
        str(metric)
        for result in aspects.values()
        if "metric" in result.summary.columns
        for metric in result.summary["metric"]
    }
    return sorted(set(gates) - known)


# Substrings that mark a metric where a higher value is better. They are
# matched against metric names, so changing them changes gate behaviour.
_HIGHER_IS_BETTER_TERMS = (
    "throughput",
    "tflops",
    "per_s",
    "per_joule",
    "qps",
    "flops",
    "bandwidth",
    "util",
)


def _higher_is_better(metric: str) -> bool:
    """Returns whether a metric name names a higher-is-better metric."""
    lowered = metric.lower()
    return any(term in lowered for term in _HIGHER_IS_BETTER_TERMS)


def evaluate_gates(
    aspects: dict[str, models.AspectResult], gates: dict[str, float]
) -> list[models.GateViolation]:
    """Checks every gated metric's ``delta_pct`` against its threshold.

    Only summaries with ``metric`` and ``delta_pct`` columns are checked; a
    missing ``delta_pct`` value counts as 0. For higher-is-better metrics
    (throughput, bandwidth, utilisation, ...) the sign of the change is
    flipped, so the observed value is always the regression in percent. A
    threshold >= 0 is violated when the observed value is above it, a
    negative threshold when the observed value is below it.

    Args:
        aspects: Aspect results keyed by aspect name.
        gates: Thresholds in percent keyed by metric name.

    Returns:
        One violation per failing (aspect, metric) row, in aspect and row
        order.
    """
    if not gates:
        return []
    violations: list[models.GateViolation] = []
    for aspect_name, result in aspects.items():
        summary = result.summary
        has_gate_columns = (
            "metric" in summary.columns and "delta_pct" in summary.columns
        )
        if not has_gate_columns:
            continue
        for row in summary.iter_rows(named=True):
            metric = str(row["metric"])
            if metric not in gates:
                continue
            threshold_pct = gates[metric]
            observed_pct = float(row["delta_pct"] or 0.0)
            if _higher_is_better(metric):
                # A drop in a throughput/efficiency metric is the
                # regression. Flip the sign so the threshold always means
                # "allowed regression in %".
                observed_pct = -observed_pct
            is_violated = (
                threshold_pct >= 0 and observed_pct > threshold_pct
            ) or (threshold_pct < 0 and observed_pct < threshold_pct)
            if is_violated:
                violations.append(
                    models.GateViolation(
                        metric=metric,
                        aspect=aspect_name,
                        observed_pct=observed_pct,
                        threshold_pct=threshold_pct,
                        message=(
                            f"Gate failed on '{metric}' ({aspect_name}): "
                            f"observed {observed_pct:+.2f}% exceeds "
                            f"threshold {threshold_pct:+.2f}%"
                        ),
                    )
                )
    return violations
