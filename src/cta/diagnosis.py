"""Bottleneck-regime diagnosis: what dominates each group's time, and shifts."""

from __future__ import annotations

from cta import models


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
