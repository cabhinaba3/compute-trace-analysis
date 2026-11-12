"""TensorRT layer-fusion & Jetson hardware/energy comparison aspects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cta.core.models import AspectResult, CompareConfig, TrialGroup
from cta.core.registry import register_aspect
from cta.core.stats import compare_entity_tables, compare_scalar_metrics


@register_aspect(
    "layer_fusion",
    title="TensorRT Layer Timing, Fusion & Precision Mix",
    required_formats=("trtexec_profile", "trtexec_engine"),
    any_of_formats=True,
)
class LayerFusionAspect:
    name = "layer_fusion"
    title = "TensorRT Layer Timing, Fusion & Precision Mix"
    required_formats = ("trtexec_profile", "trtexec_engine")
    any_of_formats = True

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        layer_dfs: list[pl.DataFrame] = []
        prec_dfs: list[pl.DataFrame] = []

        for g in groups:
            acc: dict[str, list[float]] = {
                "total_layer_ms": [],
                "layer_count": [],
                "fused_layer_count": [],
                "fusion_ratio_pct": [],
                "low_precision_layer_pct": [],
            }
            t_layers, t_precs = [], []
            for tmap in tables[g.group_id]:
                has_prof = "trtexec_profile.trt_profile" in tmap
                has_eng = "trtexec_engine.trt_engine" in tmap
                pdf = pl.read_parquet(tmap["trtexec_profile.trt_profile"]) if has_prof else pl.DataFrame()
                edf = pl.read_parquet(tmap["trtexec_engine.trt_engine"]) if has_eng else pl.DataFrame()

                if pdf.height:
                    acc["total_layer_ms"].append(float(pdf["avg_ms"].to_numpy().sum()))
                    t_layers.append(pdf.select("name", "avg_ms"))
                if edf.height:
                    n_l = edf.height
                    n_f = int(edf.filter(pl.col("is_fused")).height)
                    n_low = int(edf.filter(pl.col("precision").is_in(["INT8", "FP8", "FP16", "BF16"])).height)
                    acc["layer_count"].append(float(n_l))
                    acc["fused_layer_count"].append(float(n_f))
                    acc["fusion_ratio_pct"].append(n_f / n_l * 100.0)
                    acc["low_precision_layer_pct"].append(n_low / n_l * 100.0)
                    t_precs.append(edf.group_by("precision").agg(pl.len().cast(pl.Float64).alias("layers")))
                elif pdf.height:
                    n_l = pdf.height
                    n_f = int(pdf.filter(pl.col("is_fused")).height)
                    acc["layer_count"].append(float(n_l))
                    acc["fused_layer_count"].append(float(n_f))
                    acc["fusion_ratio_pct"].append(n_f / n_l * 100.0)

            trials.append(acc)
            layer_dfs.append(
                pl.concat(t_layers).group_by("name").agg(pl.col("avg_ms").mean())
                if t_layers
                else pl.DataFrame({"name": [], "avg_ms": []})
            )
            prec_dfs.append(
                pl.concat(t_precs).group_by("precision").agg(pl.col("layers").mean())
                if t_precs
                else pl.DataFrame({"precision": [], "layers": []})
            )

        summary = compare_scalar_metrics(
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
        if any(d.height for d in layer_dfs):
            details["by_layer"] = compare_entity_tables(labels, layer_dfs, key_col="name", val_col="avg_ms")
        if any(d.height for d in prec_dfs):
            details["by_precision"] = compare_entity_tables(labels, prec_dfs, key_col="precision", val_col="layers")
        return AspectResult(name=self.name, title=self.title, summary=summary, details=details)


def _count_bundle_queries(tmap: dict[str, Path], warmup_steps: int, max_steps: int | None) -> int:
    for fmt in ("trtexec_timing", "nsys", "kineto"):
        key = f"{fmt}.steps"
        if key in tmap:
            sdf = pl.read_parquet(tmap[key])
            # one query per step; with several ranks/reports count the steps of a single rank, not their sum
            total = int(sdf.group_by("rank").len()["len"].to_numpy().max()) if "rank" in sdf.columns and sdf.height else sdf.height
            n = max(0, total - warmup_steps) if total > warmup_steps else total
            return min(n, max_steps) if max_steps is not None else n
    return 0


@register_aspect(
    "system_energy",
    title="System Utilization, Thermals & Energy Efficiency",
    required_formats=("jtop",),
)
class SystemEnergyAspect:
    name = "system_energy"
    title = "System Utilization, Thermals & Energy Efficiency"
    required_formats = ("jtop",)
    any_of_formats = False

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}

        for g in groups:
            acc: dict[str, list[float]] = {
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
            for tmap in tables[g.group_id]:
                sdf = pl.read_parquet(tmap["jtop.samples"])
                if sdf.is_empty():
                    continue
                tot_e = float(sdf["cum_energy_j"].to_numpy()[-1])
                span_s = float(sdf["elapsed_s"].to_numpy()[-1] - sdf["elapsed_s"].to_numpy()[0])
                p_arr = sdf["power_tot_w"].to_numpy()
                avg_p = (tot_e / span_s) if span_s > 0 else float(np.mean(p_arr))

                acc["total_energy_j"].append(tot_e)
                acc["avg_power_w"].append(avg_p)
                acc["peak_power_w"].append(float(np.max(p_arr)))
                acc["mean_gpu_util_pct"].append(float(np.mean(sdf["gpu_util_pct"].to_numpy())))
                acc["mean_cpu_util_pct"].append(float(np.mean(sdf["cpu_util_pct"].to_numpy())))
                acc["peak_gpu_mem_mb"].append(float(np.max(sdf["gpu_mem_mb"].to_numpy())))
                acc["peak_ram_mb"].append(float(np.max(sdf["ram_mb"].to_numpy())))
                acc["max_gpu_temp_c"].append(float(np.max(sdf["gpu_temp_c"].to_numpy())))

                n_queries = _count_bundle_queries(tmap, config.warmup_steps, config.max_steps)
                if n_queries > 0 and tot_e > 0:
                    acc["energy_per_query_mj"].append((tot_e * 1e3) / n_queries)
                    acc["queries_per_joule"].append(n_queries / tot_e)

            trials.append(acc)
            max_t = float(np.max(acc["max_gpu_temp_c"])) if acc["max_gpu_temp_c"] else 0.0
            if max_t >= 83.0:
                scalars[g.label] = {"share_Thermal-Throttle": min(1.0, (max_t - 80.0) / 15.0)}

        summary = compare_scalar_metrics(
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
        return AspectResult(name=self.name, title=self.title, summary=summary, scalars=scalars)
