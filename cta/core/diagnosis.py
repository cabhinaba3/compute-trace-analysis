from __future__ import annotations

from cta.core.models import AspectResult, BottleneckShift, RegressionViolation, TrialGroup


def parse_gate_spec(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    gates: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid gate '{item}'. Expected '<metric>:<pct>%', e.g. 'p99_latency_ms:+5%'")
        k, v = item.split(":", 1)
        gates[k.strip()] = float(v.strip().rstrip("%"))
    return gates


def evaluate_gates(aspects: dict[str, AspectResult], gates: dict[str, float]) -> list[RegressionViolation]:
    if not gates:
        return []
    violations: list[RegressionViolation] = []
    for aspect_name, res in aspects.items():
        df = res.summary
        if "metric" not in df.columns or "delta_pct" not in df.columns:
            continue
        for row in df.iter_rows(named=True):
            metric = str(row["metric"])
            if metric not in gates:
                continue
            thresh = gates[metric]
            obs = float(row["delta_pct"] or 0.0)
            violated = (thresh >= 0 and obs > thresh) or (thresh < 0 and obs < thresh)
            if violated:
                violations.append(
                    RegressionViolation(
                        metric=metric,
                        aspect=aspect_name,
                        observed_pct=obs,
                        threshold_pct=thresh,
                        message=(
                            f"Gate failed on '{metric}' ({aspect_name}): "
                            f"observed {obs:+.2f}% exceeds threshold {thresh:+.2f}%"
                        ),
                    )
                )
    return violations


def _extract_shares(group_label: str, aspects: dict[str, AspectResult]) -> dict[str, float]:
    shares: dict[str, float] = {}
    for aspect in aspects.values():
        sc = aspect.scalars.get(group_label, {})
        for k, v in sc.items():
            if k.startswith("share_"):
                shares[k.removeprefix("share_")] = float(v)
    return shares


def _classify_regime(shares: dict[str, float]) -> str:
    if not shares:
        return "Unclassified"
    if shares.get("Thermal-Throttle", 0.0) > 0.15:
        return "Thermal/Power-Throttled"
    if shares.get("Rank-Skew", 0.0) > 0.25 and shares.get("Rank-Skew", 0.0) >= shares.get("Exposed-Comm", 0.0):
        return "Rank-Imbalanced"
    core = {k: v for k, v in shares.items() if k not in ("Thermal-Throttle", "Rank-Skew")}
    if not core:
        return max(shares.items(), key=lambda kv: kv[1])[0]
    dominant, val = max(core.items(), key=lambda kv: kv[1])
    if dominant == "Compute" and shares.get("Memory-BW", 0.0) > val:
        return "Memory-BW-Bound"
    return f"{dominant}-Dominated" if val >= 0.35 else "Mixed-Bottleneck"


def diagnose_bottleneck_shifts(
    groups: tuple[TrialGroup, ...],
    aspects: dict[str, AspectResult],
) -> list[BottleneckShift]:
    if not groups:
        return []
    base_lbl = groups[0].label
    base_shares = _extract_shares(base_lbl, aspects)
    if not base_shares:
        return []
    base_regime = _classify_regime(base_shares)
    top_b = ", ".join(f"{k} {v * 100:.1f}%" for k, v in sorted(base_shares.items(), key=lambda x: -x[1])[:2])

    if len(groups) == 1:
        return [
            BottleneckShift(
                baseline_label=base_lbl,
                candidate_label=base_lbl,
                baseline_regime=base_regime,
                candidate_regime=base_regime,
                shares_baseline=base_shares,
                shares_candidate=base_shares,
                summary=f"Bottleneck Regime: {base_lbl} [{base_regime}: {top_b}]",
            )
        ]

    shifts: list[BottleneckShift] = []
    for cand_grp in groups[1:]:
        cand_lbl = cand_grp.label
        cand_shares = _extract_shares(cand_lbl, aspects)
        if not cand_shares:
            continue
        cand_regime = _classify_regime(cand_shares)
        top_c = ", ".join(f"{k} {v * 100:.1f}%" for k, v in sorted(cand_shares.items(), key=lambda x: -x[1])[:2])
        prefix = "Regime Shift" if base_regime != cand_regime else "Regime Unchanged"
        msg = f"{prefix}: {base_lbl} [{base_regime}: {top_b}] -> {cand_lbl} [{cand_regime}: {top_c}]"
        shifts.append(
            BottleneckShift(
                baseline_label=base_lbl,
                candidate_label=cand_lbl,
                baseline_regime=base_regime,
                candidate_regime=cand_regime,
                shares_baseline=base_shares,
                shares_candidate=cand_shares,
                summary=msg,
            )
        )
    return shifts
