"""Aspect ``layer_fusion``: TensorRT layer time, fusion and precision mix."""

from __future__ import annotations

import pathlib

import polars as pl

from cta import models
from cta import plugins
from cta.analysis import stats

# Engine-layer precisions counted as low precision.
_LOW_PRECISIONS = ["INT8", "FP8", "FP16", "BF16"]


@plugins.register_aspect(
    "layer_fusion",
    title="TensorRT Layer Timing, Fusion & Precision Mix",
    required_formats=("trtexec_profile", "trtexec_engine"),
    any_of_formats=True,
)
class LayerFusionAspect(plugins.Aspect):
    """Compares TensorRT layer time, layer fusion and precision mix.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: trtexec layer profile and engine layer info.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.AnalysisConfig,
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
                base_layers = (
                    engine_layers if engine_layers.height else profile_layers
                )
                if base_layers.height:
                    n_layers = base_layers.height
                    n_fused = int(base_layers.filter(pl.col("is_fused")).height)
                    metric_values["layer_count"].append(float(n_layers))
                    metric_values["fused_layer_count"].append(float(n_fused))
                    metric_values["fusion_ratio_pct"].append(
                        n_fused / n_layers * 100.0
                    )
                if engine_layers.height:
                    n_low_precision = int(
                        engine_layers.filter(
                            pl.col("precision").is_in(_LOW_PRECISIONS)
                        ).height
                    )
                    metric_values["low_precision_layer_pct"].append(
                        n_low_precision / engine_layers.height * 100.0
                    )
                    trial_precisions.append(
                        engine_layers.group_by("precision").agg(
                            pl.len().cast(pl.Float64).alias("layers")
                        )
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
