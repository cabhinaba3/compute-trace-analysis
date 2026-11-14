"""TensorRT layer-fusion & Jetson hardware/energy comparison aspects."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl

from cta.core import models
from cta.core import registry
from cta.core import stats

# Engine-layer precisions counted as low precision.
_LOW_PRECISIONS = ["INT8", "FP8", "FP16", "BF16"]


@registry.register_aspect(
    "layer_fusion",
    title="TensorRT Layer Timing, Fusion & Precision Mix",
    required_formats=("trtexec_profile", "trtexec_engine"),
    any_of_formats=True,
)
class LayerFusionAspect:
    """Compares TensorRT layer time, layer fusion and precision mix.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: trtexec layer profile and engine layer info.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "layer_fusion"
    title = "TensorRT Layer Timing, Fusion & Precision Mix"
    required_formats = ("trtexec_profile", "trtexec_engine")
    any_of_formats = True

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Compares TensorRT layers across the trial groups.

        Layer counts and fusion come from the engine layer info when present,
        else from the layer profile; precision needs the engine layer info.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings; only the significance level is
                used.

        Returns:
            Summary metrics plus ``by_layer`` and ``by_precision`` detail
            tables when any group has the data for them.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        layer_totals: list[pl.DataFrame] = []
        precision_totals: list[pl.DataFrame] = []

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "total_layer_ms": [],
                "layer_count": [],
                "fused_layer_count": [],
                "fusion_ratio_pct": [],
                "low_precision_layer_pct": [],
            }
            trial_layers: list[pl.DataFrame] = []
            trial_precisions: list[pl.DataFrame] = []
            for table_paths in tables[group.group_id]:
                has_profile = "trtexec_profile.trt_profile" in table_paths
                has_engine = "trtexec_engine.trt_engine" in table_paths
                profile_layers = (
                    pl.read_parquet(table_paths["trtexec_profile.trt_profile"])
                    if has_profile
                    else pl.DataFrame()
                )
                engine_layers = (
                    pl.read_parquet(table_paths["trtexec_engine.trt_engine"])
                    if has_engine
                    else pl.DataFrame()
                )

                if profile_layers.height:
                    metric_values["total_layer_ms"].append(
                        float(profile_layers["avg_ms"].to_numpy().sum())
                    )
                    trial_layers.append(profile_layers.select("name", "avg_ms"))
                if engine_layers.height:
                    n_layers = engine_layers.height
                    n_fused = int(
                        engine_layers.filter(pl.col("is_fused")).height
                    )
                    n_low_precision = int(
                        engine_layers.filter(
                            pl.col("precision").is_in(_LOW_PRECISIONS)
                        ).height
                    )
                    metric_values["layer_count"].append(float(n_layers))
                    metric_values["fused_layer_count"].append(float(n_fused))
                    metric_values["fusion_ratio_pct"].append(
                        n_fused / n_layers * 100.0
                    )
                    metric_values["low_precision_layer_pct"].append(
                        n_low_precision / n_layers * 100.0
                    )
                    trial_precisions.append(
                        engine_layers.group_by("precision").agg(
                            pl.len().cast(pl.Float64).alias("layers")
                        )
                    )
                elif profile_layers.height:
                    n_layers = profile_layers.height
                    n_fused = int(
                        profile_layers.filter(pl.col("is_fused")).height
                    )
                    metric_values["layer_count"].append(float(n_layers))
                    metric_values["fused_layer_count"].append(float(n_fused))
                    metric_values["fusion_ratio_pct"].append(
                        n_fused / n_layers * 100.0
                    )

            trials.append(metric_values)
            layer_totals.append(
                pl.concat(trial_layers)
                .group_by("name")
                .agg(pl.col("avg_ms").mean())
                if trial_layers
                else pl.DataFrame({"name": [], "avg_ms": []})
            )
            precision_totals.append(
                pl.concat(trial_precisions)
                .group_by("precision")
                .agg(pl.col("layers").mean())
                if trial_precisions
                else pl.DataFrame({"precision": [], "layers": []})
            )

        summary = stats.compare_scalar_metrics(
            [
                ("total_layer_ms", "ms", True),
                ("layer_count", "layers", True),
                ("fused_layer_count", "layers", False),
                ("fusion_ratio_pct", "%", False),
                ("low_precision_layer_pct", "%", False),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        details: dict[str, pl.DataFrame] = {}
        if any(totals.height for totals in layer_totals):
            details["by_layer"] = stats.compare_entity_tables(
                labels, layer_totals, key_column="name", value_column="avg_ms"
            )
        if any(totals.height for totals in precision_totals):
            details["by_precision"] = stats.compare_entity_tables(
                labels,
                precision_totals,
                key_column="precision",
                value_column="layers",
            )
        return models.AspectResult(
            name=self.name, title=self.title, summary=summary, details=details
        )


def _count_bundle_queries(
    table_paths: dict[str, pathlib.Path],
    warmup_steps: int,
    max_steps: int | None,
) -> int:
    """Returns the number of queries (steps) a trial ran after slicing.

    Uses the first steps table among trtexec timing, nsys, Kineto and JAX.

    Args:
        table_paths: Mapping of table key to Parquet path for one trial.
        warmup_steps: Steps to skip at the start.
        max_steps: Most steps to count, or None for no limit.

    Returns:
        The query count, or 0 when the trial has no steps table.
    """
    for trace_format in ("trtexec_timing", "nsys", "kineto", "jax"):
        steps_key = f"{trace_format}.steps"
        if steps_key in table_paths:
            steps = pl.read_parquet(table_paths[steps_key])
            # One query per step; with several ranks/reports count the steps
            # of a single rank, not their sum.
            total_steps = (
                int(steps.group_by("rank").len()["len"].to_numpy().max())
                if "rank" in steps.columns and steps.height
                else steps.height
            )
            n_queries = max(0, total_steps - warmup_steps)
            return (
                min(n_queries, max_steps)
                if max_steps is not None
                else n_queries
            )
    return 0


@registry.register_aspect(
    "system_energy",
    title="System Utilization, Thermals & Energy Efficiency",
    required_formats=("jtop",),
)
class SystemEnergyAspect:
    """Compares Jetson power, energy per query, utilisation and thermals.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: jtop telemetry.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "system_energy"
    title = "System Utilization, Thermals & Energy Efficiency"
    required_formats = ("jtop",)
    any_of_formats = False

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Compares jtop telemetry across the trial groups.

        Energy per query uses the trial's steps table (see
        ``_count_bundle_queries``) when the bundle has one.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings: warm-up and step limit for the
                query count, and the significance level.

        Returns:
            Summary metrics plus a ``share_Thermal-Throttle`` scalar for
            groups whose GPU reached 83 C.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "total_energy_j": [],
                "avg_power_w": [],
                "peak_power_w": [],
                "energy_per_query_mj": [],
                "queries_per_joule": [],
                "mean_gpu_util_pct": [],
                "mean_cpu_util_pct": [],
                "peak_gpu_mem_mb": [],
                "peak_ram_mb": [],
                "max_gpu_temp_c": [],
            }
            for table_paths in tables[group.group_id]:
                samples = pl.read_parquet(table_paths["jtop.samples"])
                if samples.is_empty():
                    continue
                # One telemetry file per device: energies add, the run lasts
                # as long as the longest device.
                per_device = samples.group_by(
                    "device", maintain_order=True
                ).agg(
                    pl.col("cumulative_energy_j").last().alias("e"),
                    (
                        pl.col("elapsed_s").last() - pl.col("elapsed_s").first()
                    ).alias("span"),
                )
                total_energy_j = float(per_device["e"].sum())
                span_s = float(np.max(per_device["span"].to_numpy()))
                power_w = samples["power_total_w"].to_numpy()
                avg_power_w = (
                    (total_energy_j / span_s)
                    if span_s > 0
                    else float(np.mean(power_w))
                )

                metric_values["total_energy_j"].append(total_energy_j)
                metric_values["avg_power_w"].append(avg_power_w)
                metric_values["peak_power_w"].append(float(np.max(power_w)))
                metric_values["mean_gpu_util_pct"].append(
                    float(np.mean(samples["gpu_util_pct"].to_numpy()))
                )
                metric_values["mean_cpu_util_pct"].append(
                    float(np.mean(samples["cpu_util_pct"].to_numpy()))
                )
                metric_values["peak_gpu_mem_mb"].append(
                    float(np.max(samples["gpu_mem_mb"].to_numpy()))
                )
                metric_values["peak_ram_mb"].append(
                    float(np.max(samples["ram_mb"].to_numpy()))
                )
                metric_values["max_gpu_temp_c"].append(
                    float(np.max(samples["gpu_temp_c"].to_numpy()))
                )

                n_queries = _count_bundle_queries(
                    table_paths, config.warmup_steps, config.max_steps
                )
                if n_queries > 0 and total_energy_j > 0:
                    metric_values["energy_per_query_mj"].append(
                        (total_energy_j * 1e3) / n_queries
                    )
                    metric_values["queries_per_joule"].append(
                        n_queries / total_energy_j
                    )

            trials.append(metric_values)
            max_temp_c = (
                float(np.max(metric_values["max_gpu_temp_c"]))
                if metric_values["max_gpu_temp_c"]
                else 0.0
            )
            if max_temp_c >= 83.0:
                scalars[group.label] = {
                    "share_Thermal-Throttle": min(
                        1.0, (max_temp_c - 80.0) / 15.0
                    )
                }

        summary = stats.compare_scalar_metrics(
            [
                ("total_energy_j", "J", True),
                ("avg_power_w", "W", True),
                ("peak_power_w", "W", True),
                ("energy_per_query_mj", "mJ/query", True),
                ("queries_per_joule", "queries/J", False),
                ("mean_gpu_util_pct", "%", False),
                ("mean_cpu_util_pct", "%", False),
                ("peak_gpu_mem_mb", "MB", True),
                ("peak_ram_mb", "MB", True),
                ("max_gpu_temp_c", "C", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return models.AspectResult(
            name=self.name, title=self.title, summary=summary, scalars=scalars
        )
