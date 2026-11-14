"""The ``roofline_diff`` aspect: Nsight Compute roofline and utilisation."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl

from cta.core import models
from cta.core import registry
from cta.core import stats

# Summary metric and the ``ncu_kernels`` column it duration-weights.
_WEIGHTED_METRIC_COLUMNS = (
    ("weighted_sm_pct", "sm_throughput_pct"),
    ("weighted_dram_pct", "dram_throughput_pct"),
    ("weighted_tensor_pct", "tensor_pipe_pct"),
    ("weighted_occupancy_pct", "occupancy_pct"),
    ("stall_mem_pct", "stall_mem_pct"),
    ("stall_compute_pct", "stall_compute_pct"),
)


def _duration_weighted_mean(df: pl.DataFrame, column: str) -> float | None:
    """Returns the duration-weighted mean of ``column`` over non-null rows.

    Falls back to the plain mean when the durations sum to zero.

    Args:
        df: Table with ``column`` and ``duration_ns`` columns.
        column: Column to average.

    Returns:
        The mean, or None when no row has a value in ``column``.
    """
    with_value = df.filter(pl.col(column).is_not_null())
    if with_value.is_empty():
        return None
    total_ns = float(with_value["duration_ns"].sum() or 0.0)
    if total_ns <= 0:
        return float(with_value[column].mean())  # type: ignore[arg-type]
    weighted = with_value[column] * with_value["duration_ns"]
    return float(weighted.sum()) / total_ns


def _roofline_table(kernels: pl.DataFrame) -> pl.DataFrame:
    """Aggregates profiled kernels into one roofline point per kernel name.

    Arithmetic intensity and TFLOP/s are ratios of per-kernel sums
    (sum(FLOPs) / sum(bytes) and sum(FLOPs) / sum(time)), not means of
    per-launch ratios; throughput percentages are duration-weighted.

    Args:
        kernels: Concatenated ``ncu_kernels`` tables of one group.

    Returns:
        One row per (``canonical_name``, ``op_family``).
    """
    return (
        kernels.group_by(["canonical_name", "op_family"])
        .agg(
            (pl.col("duration_ns").sum() / 1e3).alias("duration_us"),
            pl.col("flops").sum().alias("_flops"),
            pl.col("dram_bytes").sum().alias("_bytes"),
            pl.col("flops").is_not_null().any().alias("_has_flops"),
            pl.col("duration_ns").sum().alias("_dur_ns"),
            (pl.col("sm_throughput_pct") * pl.col("duration_ns"))
            .sum()
            .alias("_sm_w"),
            (pl.col("dram_throughput_pct") * pl.col("duration_ns"))
            .sum()
            .alias("_dr_w"),
        )
        .with_columns(
            pl.when(pl.col("_has_flops") & (pl.col("_bytes") > 0))
            .then(pl.col("_flops") / pl.col("_bytes"))
            .alias("arithmetic_intensity"),
            pl.when(pl.col("_has_flops") & (pl.col("_dur_ns") > 0))
            .then(pl.col("_flops") / pl.col("_dur_ns") / 1e3)
            .alias("achieved_tflops"),
            (pl.col("_sm_w") / pl.col("_dur_ns")).alias("sm_throughput_pct"),
            (pl.col("_dr_w") / pl.col("_dur_ns")).alias("dram_throughput_pct"),
        )
        .select(
            "canonical_name",
            "op_family",
            "duration_us",
            "arithmetic_intensity",
            "achieved_tflops",
            "sm_throughput_pct",
            "dram_throughput_pct",
        )
    )


def _findings_table(findings: list[pl.DataFrame]) -> pl.DataFrame:
    """Counts the kernels each warning/optimisation/error NCU rule flagged."""
    return (
        pl.concat(findings)
        .filter(pl.col("level").is_in(["warning", "optimization", "error"]))
        .group_by(["rule", "title", "level"])
        .agg(
            pl.len().alias("kernels"),
            pl.col("speedup_pct").max().alias("max_speedup_pct"),
        )
        .sort("kernels", descending=True)
    )


@registry.register_aspect(
    "roofline_diff",
    title="Nsight Compute Roofline, Utilization & Warp Stalls",
    required_formats=("ncu",),
)
class RooflineDiffAspect:
    """Compares Nsight Compute utilisation, roofline position and stalls.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Nsight Compute, which provides per-kernel counters.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "roofline_diff"
    title = "Nsight Compute Roofline, Utilization & Warp Stalls"
    required_formats = ("ncu",)
    any_of_formats = False

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Compares Nsight Compute kernel metrics across the trial groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings; only the significance level is
                used.

        Returns:
            Summary metrics, a ``by_kernel`` detail table, one
            ``roofline_<label>`` table per group with kernels and one
            ``findings_<label>`` table per group with NCU rule findings,
            plus compute/memory-bandwidth shares as scalars.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}
        kernel_totals: list[pl.DataFrame] = []
        roofline_tables: dict[str, pl.DataFrame] = {}
        findings_tables: dict[str, pl.DataFrame] = {}

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "total_kernel_us": [],
                "weighted_sm_pct": [],
                "weighted_dram_pct": [],
                "weighted_tensor_pct": [],
                "weighted_occupancy_pct": [],
                "overall_arithmetic_intensity": [],
                "achieved_tflops": [],
                "stall_mem_pct": [],
                "stall_compute_pct": [],
            }
            group_kernels: list[pl.DataFrame] = []
            group_findings: list[pl.DataFrame] = []
            for table_paths in tables[group.group_id]:
                kernels = pl.read_parquet(table_paths["ncu.ncu_kernels"])
                if kernels.is_empty():
                    continue
                total_ns = float(kernels["duration_ns"].sum() or 0.0)
                metric_values["total_kernel_us"].append(total_ns / 1e3)
                # Missing metrics are null (not 0): average only over kernels
                # that report the metric.
                for metric, column in _WEIGHTED_METRIC_COLUMNS:
                    weighted_mean = _duration_weighted_mean(kernels, column)
                    if weighted_mean is not None:
                        metric_values[metric].append(weighted_mean)
                with_flops = kernels.filter(pl.col("flops").is_not_null())
                if with_flops.height:
                    flops = float(with_flops["flops"].sum())
                    flops_duration_ns = float(
                        with_flops["duration_ns"].sum() or 0.0
                    )
                    with_bytes = with_flops.filter(
                        pl.col("dram_bytes").is_not_null()
                        & (pl.col("dram_bytes") > 0)
                    )
                    # Arithmetic intensity = total FLOPs / total DRAM bytes of
                    # the kernels that report both.
                    if with_bytes.height:
                        metric_values["overall_arithmetic_intensity"].append(
                            float(with_bytes["flops"].sum())
                            / float(with_bytes["dram_bytes"].sum())
                        )
                    if flops_duration_ns > 0:
                        metric_values["achieved_tflops"].append(
                            flops / flops_duration_ns / 1e3
                        )
                group_kernels.append(kernels)
                if "ncu.ncu_findings" in table_paths:
                    findings = pl.read_parquet(table_paths["ncu.ncu_findings"])
                    if findings.height:
                        group_findings.append(findings)

            # NCU's own guided-analysis rules, summarised per rule
            # (report-based inputs only).
            if group_findings:
                findings_tables[f"findings_{group.label}"] = _findings_table(
                    group_findings
                )
            trials.append(metric_values)
            sm_pct = (
                float(np.mean(metric_values["weighted_sm_pct"]))
                if metric_values["weighted_sm_pct"]
                else 0.0
            )
            dram_pct = (
                float(np.mean(metric_values["weighted_dram_pct"]))
                if metric_values["weighted_dram_pct"]
                else 0.0
            )
            total_pct = (sm_pct + dram_pct) or 1.0
            scalars[group.label] = {
                "share_Compute": sm_pct / total_pct,
                "share_Memory-BW": dram_pct / total_pct,
            }

            if group_kernels:
                roofline = _roofline_table(pl.concat(group_kernels))
                roofline_tables[f"roofline_{group.label}"] = roofline
                kernel_totals.append(
                    roofline.select("canonical_name", "duration_us")
                )
            else:
                kernel_totals.append(
                    pl.DataFrame({"canonical_name": [], "duration_us": []})
                )

        summary = stats.compare_scalar_metrics(
            [
                ("total_kernel_us", "us", True),
                ("achieved_tflops", "TFLOP/s", False),
                ("overall_arithmetic_intensity", "FLOP/B", False),
                ("weighted_sm_pct", "%", False),
                ("weighted_dram_pct", "%", False),
                ("weighted_tensor_pct", "%", False),
                ("weighted_occupancy_pct", "%", False),
                ("stall_mem_pct", "%", True),
                ("stall_compute_pct", "%", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        details = {
            "by_kernel": stats.compare_entity_tables(
                labels,
                kernel_totals,
                key_column="canonical_name",
                value_column="duration_us",
            ),
            **roofline_tables,
            **findings_tables,
        }
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary,
            details=details,
            scalars=scalars,
        )
