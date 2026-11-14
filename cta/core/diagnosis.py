"""Regression gates and bottleneck-regime diagnosis of comparison results."""

from __future__ import annotations

from cta.core import models


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
        for metric in result.summary["metric"].to_list()
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
) -> list[models.RegressionViolation]:
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
    violations: list[models.RegressionViolation] = []
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
                    models.RegressionViolation(
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


def _extract_shares(
    group_label: str, aspects: dict[str, models.AspectResult]
) -> dict[str, float]:
    """Collects a group's ``share_<category>`` scalars from all aspects.

    Returns:
        Category -> time share. A later aspect overrides an earlier one.
    """
    shares: dict[str, float] = {}
    for result in aspects.values():
        scalars = result.scalars.get(group_label, {})
        for name, value in scalars.items():
            if name.startswith("share_"):
                shares[name.removeprefix("share_")] = float(value)
    return shares


def _classify_regime(shares: dict[str, float]) -> str:
    """Names the bottleneck regime that a set of time shares indicates.

    Rules, first match wins: thermal throttling above 15%; rank skew above
    25% and at least the exposed-comm share; else the largest remaining
    share, "Dominated" when it is at least 35%, "Mixed" otherwise.
    """
    if not shares:
        return "Unclassified"
    if shares.get("Thermal-Throttle", 0.0) > 0.15:
        return "Thermal/Power-Throttled"
    rank_skew_share = shares.get("Rank-Skew", 0.0)
    if rank_skew_share > 0.25 and rank_skew_share >= shares.get(
        "Exposed-Comm", 0.0
    ):
        return "Rank-Imbalanced"
    core_shares = {
        category: share
        for category, share in shares.items()
        if category not in ("Thermal-Throttle", "Rank-Skew")
    }
    if not core_shares:
        return max(shares.items(), key=lambda item: item[1])[0]
    dominant, dominant_share = max(
        core_shares.items(), key=lambda item: item[1]
    )
    return (
        f"{dominant}-Dominated"
        if dominant_share >= 0.35
        else "Mixed-Bottleneck"
    )


def _top_shares_text(shares: dict[str, float]) -> str:
    """Formats the two largest shares as ``"<category> <pct>%, ..."``."""
    top_two = sorted(shares.items(), key=lambda item: -item[1])[:2]
    return ", ".join(
        f"{category} {share * 100:.1f}%" for category, share in top_two
    )


def diagnose_bottleneck_shifts(
    groups: tuple[models.TrialGroup, ...],
    aspects: dict[str, models.AspectResult],
) -> list[models.BottleneckShift]:
    """Classifies each group's bottleneck regime and compares it to baseline.

    The regime is derived from the ``share_<category>`` scalars that the
    aspects report per group. The first group is the baseline.

    Args:
        groups: The analysed groups; the first one is the baseline.
        aspects: Aspect results keyed by aspect name.

    Returns:
        For a single group, one entry describing its regime. Otherwise one
        entry per candidate group with shares. Empty when there are no
        groups or the baseline has no shares.
    """
    if not groups:
        return []
    baseline_label = groups[0].label
    baseline_shares = _extract_shares(baseline_label, aspects)
    if not baseline_shares:
        return []
    baseline_regime = _classify_regime(baseline_shares)
    baseline_top = _top_shares_text(baseline_shares)

    if len(groups) == 1:
        return [
            models.BottleneckShift(
                baseline_label=baseline_label,
                candidate_label=baseline_label,
                baseline_regime=baseline_regime,
                candidate_regime=baseline_regime,
                shares_baseline=baseline_shares,
                shares_candidate=baseline_shares,
                summary=(
                    f"Bottleneck Regime: {baseline_label} "
                    f"[{baseline_regime}: {baseline_top}]"
                ),
            )
        ]

    shifts: list[models.BottleneckShift] = []
    for candidate_group in groups[1:]:
        candidate_label = candidate_group.label
        candidate_shares = _extract_shares(candidate_label, aspects)
        if not candidate_shares:
            continue
        candidate_regime = _classify_regime(candidate_shares)
        candidate_top = _top_shares_text(candidate_shares)
        prefix = (
            "Regime Shift"
            if baseline_regime != candidate_regime
            else "Regime Unchanged"
        )
        summary = (
            f"{prefix}: {baseline_label} [{baseline_regime}: {baseline_top}] "
            f"-> {candidate_label} [{candidate_regime}: {candidate_top}]"
        )
        shifts.append(
            models.BottleneckShift(
                baseline_label=baseline_label,
                candidate_label=candidate_label,
                baseline_regime=baseline_regime,
                candidate_regime=candidate_regime,
                shares_baseline=baseline_shares,
                shares_candidate=candidate_shares,
                summary=summary,
            )
        )
    return shifts
