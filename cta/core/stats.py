"""Trial statistics and comparison tables for scalar and per-entity metrics."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

# Means closer to zero than this count as zero when dividing by them.
_NEAR_ZERO = 1e-12


def _trial_summary(
    values: list[float],
) -> tuple[float, float, float, float, float]:
    """Summarises one group's trial values.

    Returns:
        ``(mean, sample std, 95% CI half-width, median, IQR)``. All zero
        without values; std and CI are zero for a single value.
    """
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    samples = np.asarray(values, dtype=np.float64)
    n = len(samples)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1)) if n > 1 else 0.0
    ci95 = 0.0
    if n > 1:
        # Lazy: importing scipy.stats costs more than 1 s.
        from scipy import stats as sp_stats

        ci95 = float(sp_stats.t.ppf(0.975, df=n - 1) * std / math.sqrt(n))
    q25, q50, q75 = np.percentile(samples, [25.0, 50.0, 75.0])
    return (mean, std, ci95, float(q50), float(q75 - q25))


def _hypothesis_stats(
    baseline_values: list[float], candidate_values: list[float], alpha: float
) -> tuple[float | None, float | None, float | None, bool]:
    """Tests whether candidate trials differ from baseline trials.

    Returns:
        ``(Welch t-test p, Mann-Whitney U p, Cohen's d, significant)``,
        where significant means the Welch p is below ``alpha``. All None
        and False with fewer than two trials per side. When neither side
        varies, both p-values are 1 (equal means, d = 0) or 0 (d = inf).
    """
    if len(baseline_values) < 2 or len(candidate_values) < 2:
        return (None, None, None, False)
    # Lazy: importing scipy.stats costs more than 1 s.
    from scipy import stats as sp_stats

    baseline = np.asarray(baseline_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    baseline_var = np.var(baseline, ddof=1)
    candidate_var = np.var(candidate, ddof=1)
    if baseline_var == 0.0 and candidate_var == 0.0:
        is_same = float(np.mean(baseline)) == float(np.mean(candidate))
        p_value = 1.0 if is_same else 0.0
        cohens_d = 0.0 if is_same else float("inf")
        return (p_value, p_value, cohens_d, not is_same)
    _, welch_p = sp_stats.ttest_ind(candidate, baseline, equal_var=False)
    try:
        _, mannwhitney_p = sp_stats.mannwhitneyu(
            candidate, baseline, alternative="two-sided"
        )
    except ValueError:
        mannwhitney_p = 1.0
    pooled_sd = math.sqrt(
        (
            (len(baseline) - 1) * baseline_var
            + (len(candidate) - 1) * candidate_var
        )
        / (len(baseline) + len(candidate) - 2)
    )
    cohens_d = (
        float((np.mean(candidate) - np.mean(baseline)) / pooled_sd)
        if pooled_sd > 0
        else 0.0
    )
    p_welch = float(welch_p) if not math.isnan(welch_p) else 1.0
    p_mannwhitney = (
        float(mannwhitney_p) if not math.isnan(mannwhitney_p) else 1.0
    )
    return (p_welch, p_mannwhitney, cohens_d, p_welch < alpha)


def _single_group_row(
    metric: str, unit: str, label: str, values: list[float]
) -> dict[str, object]:
    """Builds the summary row of one metric for a single group."""
    mean, std, ci95, median, _ = _trial_summary(values)
    return {
        "metric": metric,
        "unit": unit,
        label: mean,
        "std": std,
        "median": median,
        "ci95_single": ci95,
    }


def _pair_row(
    metric: str,
    unit: str,
    lower_is_better: bool,
    baseline_values: list[float],
    candidate_values: list[float],
    alpha: float,
) -> dict[str, object]:
    """Builds the baseline-vs-candidate row of one metric."""
    baseline_mean, baseline_std, baseline_ci95, baseline_median, _ = (
        _trial_summary(baseline_values)
    )
    candidate_mean, candidate_std, candidate_ci95, candidate_median, _ = (
        _trial_summary(candidate_values)
    )
    delta = candidate_mean - baseline_mean
    delta_pct = (
        (delta / baseline_mean * 100.0)
        if abs(baseline_mean) > _NEAR_ZERO
        else 0.0
    )
    # Direction-aware: a speedup above 1 means the candidate is better.
    if lower_is_better:
        speedup = (
            (baseline_mean / candidate_mean)
            if abs(candidate_mean) > _NEAR_ZERO
            else 1.0
        )
    else:
        speedup = (
            (candidate_mean / baseline_mean)
            if abs(baseline_mean) > _NEAR_ZERO
            else 1.0
        )
    p_welch, p_mannwhitney, cohens_d, is_significant = _hypothesis_stats(
        baseline_values, candidate_values, alpha
    )
    return {
        "metric": metric,
        "unit": unit,
        "baseline": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_median": baseline_median,
        "candidate": candidate_mean,
        "candidate_std": candidate_std,
        "candidate_median": candidate_median,
        "delta": delta,
        "delta_pct": delta_pct,
        "speedup": speedup,
        "ci95_baseline": baseline_ci95,
        "ci95_candidate": candidate_ci95,
        "p_welch": p_welch,
        "p_mannwhitney": p_mannwhitney,
        "cohens_d": cohens_d,
        "significant": is_significant,
    }


def _multi_group_row(
    metric: str,
    unit: str,
    baseline_mean: float,
    group_labels: list[str],
    group_trials: list[dict[str, list[float]]],
) -> dict[str, object]:
    """Builds the row of one metric across three or more groups."""
    row: dict[str, object] = {"metric": metric, "unit": unit}
    for label, trials in zip(group_labels, group_trials, strict=True):
        mean, std, _, _, _ = _trial_summary(trials.get(metric, []))
        row[label] = mean
        row[f"{label}_std"] = std
        row[f"{label}_delta_pct"] = (
            ((mean - baseline_mean) / baseline_mean * 100.0)
            if abs(baseline_mean) > _NEAR_ZERO
            else 0.0
        )
    return row


def compare_scalar_metrics(
    metrics: list[tuple[str, str, bool]],
    group_labels: list[str],
    group_trials: list[dict[str, list[float]]],
    *,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Builds a comparison table of scalar metrics across 1, 2 or N groups.

    Metrics without values in any group are skipped.

    Args:
        metrics: ``(metric name, unit, lower_is_better)`` per metric, in
            output order.
        group_labels: Group labels; the first group is the baseline.
        group_trials: Per group, metric name -> one value per trial.
        alpha: Significance level of the Welch t-test.

    Returns:
        One row per metric with ``metric`` and ``unit``, plus:

        - one group: ``<label>`` (the mean), ``std``, ``median`` and
          ``ci95_single``;
        - two groups: mean, std and median of ``baseline`` and
          ``candidate``, ``delta``, ``delta_pct``, a direction-aware
          ``speedup`` (above 1 means the candidate is better), 95% CIs,
          Welch and Mann-Whitney p-values, ``cohens_d`` and
          ``significant``;
        - more groups: ``<label>``, ``<label>_std`` and
          ``<label>_delta_pct`` (versus the first group) per group.

        Without rows, an empty frame with ``metric`` and ``unit`` columns.
    """
    rows: list[dict[str, object]] = []
    is_pair = len(group_labels) == 2

    for metric, unit, lower_is_better in metrics:
        baseline_values = group_trials[0].get(metric, [])
        if not any(trials.get(metric) for trials in group_trials):
            continue
        if is_pair:
            candidate_values = group_trials[1].get(metric, [])
            rows.append(
                _pair_row(
                    metric,
                    unit,
                    lower_is_better,
                    baseline_values,
                    candidate_values,
                    alpha,
                )
            )
        elif len(group_labels) == 1:
            rows.append(
                _single_group_row(
                    metric, unit, group_labels[0], baseline_values
                )
            )
        else:
            baseline_mean = _trial_summary(baseline_values)[0]
            rows.append(
                _multi_group_row(
                    metric, unit, baseline_mean, group_labels, group_trials
                )
            )

    return (
        pl.DataFrame(rows) if rows else pl.DataFrame({"metric": [], "unit": []})
    )


def compare_entity_tables(
    group_labels: list[str],
    group_tables: list[pl.DataFrame],
    *,
    key_column: str,
    value_column: str,
    count_column: str | None = None,
) -> pl.DataFrame:
    """Outer-joins per-entity totals across groups.

    Entities are kernels, op families, layers, collectives and the like.
    Values missing in a group become 0. With two groups the result gets
    ``delta``, ``delta_pct`` (100 for an entity new in the candidate) and
    ``speedup`` = baseline / candidate, sorted by absolute delta; with one
    group a ``share_pct`` of the total, sorted by value; with more groups
    it is sorted by the first group's value. All sorts are descending.

    Args:
        group_labels: Group labels; each names a value column.
        group_tables: One table per group, in label order.
        key_column: Column that identifies the entity.
        value_column: Column with the value to compare.
        count_column: Optional count column, copied as ``<label>_count`` from
            the groups that have it.

    Returns:
        One row per entity: ``key_column``, one value column per group label
        and the columns above. Only a ``key_column`` column when there is
        nothing to compare.
    """
    merged: pl.DataFrame | None = None
    for label, entities in zip(group_labels, group_tables, strict=True):
        columns = [pl.col(key_column), pl.col(value_column).alias(label)]
        if count_column and count_column in entities.columns:
            columns.append(pl.col(count_column).alias(f"{label}_count"))
        group_table = entities.select(columns)
        # An empty group built as pl.DataFrame({key: []}) has no dtype to
        # join on.
        if group_table.schema[key_column] == pl.Null:
            group_table = group_table.with_columns(
                pl.col(key_column).cast(pl.String)
            )
        merged = (
            group_table
            if merged is None
            else merged.join(
                group_table, on=key_column, how="full", coalesce=True
            )
        )

    if merged is None or merged.is_empty():
        return pl.DataFrame({key_column: []})

    fill_nulls = [pl.col(label).fill_null(0.0) for label in group_labels]
    if count_column:
        fill_nulls.extend(
            pl.col(f"{label}_count").fill_null(0)
            for label in group_labels
            if f"{label}_count" in merged.columns
        )
    merged = merged.with_columns(fill_nulls)

    if len(group_labels) == 2:
        baseline = pl.col(group_labels[0])
        candidate = pl.col(group_labels[1])
        merged = merged.with_columns(
            (candidate - baseline).alias("delta"),
            pl.when(baseline.abs() > _NEAR_ZERO)
            .then((candidate - baseline) / baseline * 100.0)
            .otherwise(pl.when(candidate > 0).then(100.0).otherwise(0.0))
            .alias("delta_pct"),
            pl.when(candidate.abs() > _NEAR_ZERO)
            .then(baseline / candidate)
            .otherwise(1.0)
            .alias("speedup"),
        ).sort(pl.col("delta").abs(), descending=True)
    elif len(group_labels) == 1:
        values = pl.col(group_labels[0])
        total = float(merged[group_labels[0]].sum())
        share_pct = (
            (values / total * 100.0) if abs(total) > _NEAR_ZERO else pl.lit(0.0)
        )
        merged = merged.with_columns(share_pct.alias("share_pct")).sort(
            group_labels[0], descending=True
        )
    else:
        merged = merged.sort(group_labels[0], descending=True)

    return merged
