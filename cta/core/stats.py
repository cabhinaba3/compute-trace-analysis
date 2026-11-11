from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import stats as sp_stats


def _trial_summary(vals: list[float]) -> tuple[float, float, float, float, float]:
    if not vals:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(sp_stats.t.ppf(0.975, df=n - 1) * std / math.sqrt(n)) if n > 1 else 0.0
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return (mean, std, ci95, float(q50), float(q75 - q25))


def _hypothesis_stats(base: list[float], cand: list[float], alpha: float) -> tuple[float | None, float | None, float | None, bool]:
    if len(base) < 2 or len(cand) < 2:
        return (None, None, None, False)
    b = np.asarray(base, dtype=np.float64)
    c = np.asarray(cand, dtype=np.float64)
    b_var, c_var = np.var(b, ddof=1), np.var(c, ddof=1)
    if b_var == 0.0 and c_var == 0.0:
        same = float(np.mean(b)) == float(np.mean(c))
        p_val = 1.0 if same else 0.0
        return (p_val, p_val, 0.0 if same else float("inf"), not same)
    _, p_welch = sp_stats.ttest_ind(c, b, equal_var=False)
    try:
        _, p_mwu = sp_stats.mannwhitneyu(c, b, alternative="two-sided")
    except ValueError:
        p_mwu = 1.0
    pooled_sd = math.sqrt(((len(b) - 1) * b_var + (len(c) - 1) * c_var) / (len(b) + len(c) - 2))
    cohens_d = float((np.mean(c) - np.mean(b)) / pooled_sd) if pooled_sd > 0 else 0.0
    pw = float(p_welch) if not math.isnan(p_welch) else 1.0
    pm = float(p_mwu) if not math.isnan(p_mwu) else 1.0
    return (pw, pm, cohens_d, pw < alpha)


def compare_scalar_metrics(
    metrics: list[tuple[str, str, bool]],
    group_labels: list[str],
    group_trials: list[dict[str, list[float]]],
    *,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Build a crisp comparison DataFrame for scalar metrics across 2 or N TrialGroups.

    metrics: [(metric_key, unit, lower_is_better), ...]
    """
    rows: list[dict[str, object]] = []
    is_pair = len(group_labels) == 2

    for key, unit, lower_is_better in metrics:
        base_vals = group_trials[0].get(key, [])
        if not any(gt.get(key) for gt in group_trials):
            continue
        b_mean, b_std, b_ci, b_med, _ = _trial_summary(base_vals)
        if is_pair:
            cand_vals = group_trials[1].get(key, [])
            c_mean, c_std, c_ci, c_med, _ = _trial_summary(cand_vals)
            delta = c_mean - b_mean
            delta_pct = (delta / b_mean * 100.0) if abs(b_mean) > 1e-12 else 0.0
            if lower_is_better:
                speedup = (b_mean / c_mean) if abs(c_mean) > 1e-12 else 1.0
            else:
                speedup = (c_mean / b_mean) if abs(b_mean) > 1e-12 else 1.0
            pw, pm, d, sig = _hypothesis_stats(base_vals, cand_vals, alpha)
            rows.append(
                {
                    "metric": key,
                    "unit": unit,
                    "baseline": b_mean,
                    "baseline_std": b_std,
                    "baseline_median": b_med,
                    "candidate": c_mean,
                    "candidate_std": c_std,
                    "candidate_median": c_med,
                    "delta": delta,
                    "delta_pct": delta_pct,
                    "speedup": speedup,
                    "ci95_baseline": b_ci,
                    "ci95_candidate": c_ci,
                    "p_welch": pw,
                    "p_mannwhitney": pm,
                    "cohens_d": d,
                    "significant": sig,
                }
            )
        else:
            row: dict[str, object] = {"metric": key, "unit": unit}
            for lbl, gt in zip(group_labels, group_trials, strict=True):
                vals = gt.get(key, [])
                m, s, _, _, _ = _trial_summary(vals)
                row[lbl] = m
                row[f"{lbl}_std"] = s
                row[f"{lbl}_delta_pct"] = ((m - b_mean) / b_mean * 100.0) if abs(b_mean) > 1e-12 else 0.0
            rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame({"metric": [], "unit": []})


def compare_entity_tables(
    group_labels: list[str],
    group_dfs: list[pl.DataFrame],
    *,
    key_col: str,
    val_col: str,
    count_col: str | None = None,
) -> pl.DataFrame:
    """Vectorized outer join comparing per-entity (kernel/family/layer/collective) totals across groups."""
    merged: pl.DataFrame | None = None
    for idx, (lbl, df) in enumerate(zip(group_labels, group_dfs, strict=True)):
        cols = [pl.col(key_col), pl.col(val_col).alias(lbl)]
        if count_col and count_col in df.columns:
            cols.append(pl.col(count_col).alias(f"{lbl}_count"))
        sub = df.select(cols)
        merged = sub if idx == 0 else merged.join(sub, on=key_col, how="full", coalesce=True)  # type: ignore[union-attr]

    if merged is None or merged.is_empty():
        return pl.DataFrame({key_col: []})

    fill_exprs = [pl.col(lbl).fill_null(0.0) for lbl in group_labels]
    if count_col:
        fill_exprs.extend(
            pl.col(f"{lbl}_count").fill_null(0)
            for lbl in group_labels
            if f"{lbl}_count" in merged.columns
        )
    merged = merged.with_columns(fill_exprs)

    if len(group_labels) == 2:
        b_col, c_col = pl.col(group_labels[0]), pl.col(group_labels[1])
        merged = merged.with_columns(
            (c_col - b_col).alias("delta"),
            pl.when(b_col.abs() > 1e-12)
            .then((c_col - b_col) / b_col * 100.0)
            .otherwise(pl.when(c_col > 0).then(100.0).otherwise(0.0))
            .alias("delta_pct"),
            pl.when(c_col.abs() > 1e-12).then(b_col / c_col).otherwise(1.0).alias("speedup"),
        ).sort(pl.col("delta").abs(), descending=True)
    else:
        merged = merged.sort(group_labels[0], descending=True)

    return merged
