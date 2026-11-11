"""Device, kernel, timeline, OS runtime, and roofline comparison aspects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cta.core.intervals import compute_timeline_overlap
from cta.core.models import AspectResult, CompareConfig, TrialGroup
from cta.core.registry import register_aspect
from cta.core.stats import compare_entity_tables, compare_scalar_metrics

_TIMELINE_FORMATS = ("nsys", "kineto", "trtexec_timing")
_KERNEL_FORMATS = ("nsys", "kineto")


def _slice_steps_and_kernels(
    steps_df: pl.DataFrame,
    kernels_df: pl.DataFrame,
    config: CompareConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if steps_df.is_empty():
        return steps_df, kernels_df
    ordered = steps_df.sort("start_ns")
    if config.warmup_steps > 0 and ordered.height > config.warmup_steps:
        ordered = ordered.slice(config.warmup_steps)
    if config.max_steps is not None and config.max_steps > 0:
        ordered = ordered.head(config.max_steps)

    if kernels_df.is_empty() or ordered.height == steps_df.height:
        return ordered, kernels_df

    valid_steps = ordered["step"].to_list()
    w_min = int(ordered["start_ns"].to_numpy().min())
    w_max = int(ordered["end_ns"].to_numpy().max())
    filtered_k = kernels_df.filter(
        pl.col("step").is_in(valid_steps) | ((pl.col("start_ns") >= w_min) & (pl.col("end_ns") <= w_max))
    )
    return ordered, filtered_k


def _pick_timeline_tables(tmap: dict[str, Path]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    for fmt in _TIMELINE_FORMATS:
        if f"{fmt}.steps" in tmap:
            steps = pl.read_parquet(tmap[f"{fmt}.steps"])
            kernels = pl.read_parquet(tmap[f"{fmt}.kernels"]) if f"{fmt}.kernels" in tmap else pl.DataFrame()
            runtime = pl.read_parquet(tmap[f"{fmt}.runtime"]) if f"{fmt}.runtime" in tmap else pl.DataFrame()
            return steps, kernels, runtime
    raise KeyError("No timeline steps table found in bundle")


@register_aspect(
    "latency_throughput",
    title="End-to-End Latency, Tail Percentiles & Throughput",
    required_formats=_TIMELINE_FORMATS,
    any_of_formats=True,
)
class LatencyThroughputAspect:
    name = "latency_throughput"
    title = "End-to-End Latency, Tail Percentiles & Throughput"
    required_formats = _TIMELINE_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []

        for g in groups:
            acc: dict[str, list[float]] = {
                "mean_step_ms": [],
                "median_step_ms": [],
                "p90_step_ms": [],
                "p95_step_ms": [],
                "p99_step_ms": [],
                "throughput_per_s": [],
                "total_steps": [],
                "cuda_runtime_ms": [],
            }
            for tmap in tables[g.group_id]:
                steps_raw, kernels_raw, runtime_df = _pick_timeline_tables(tmap)
                steps_df, _ = _slice_steps_and_kernels(steps_raw, kernels_raw, config)
                if steps_df.is_empty():
                    continue
                durs_ms = steps_df["duration_ns"].to_numpy().astype(np.float64) / 1e6
                s_min = float(steps_df["start_ns"].to_numpy().min())
                s_max = float(steps_df["end_ns"].to_numpy().max())
                wall_s = max(1e-9, (s_max - s_min) / 1e9)

                acc["mean_step_ms"].append(float(np.mean(durs_ms)))
                acc["median_step_ms"].append(float(np.median(durs_ms)))
                acc["p90_step_ms"].append(float(np.percentile(durs_ms, 90)))
                acc["p95_step_ms"].append(float(np.percentile(durs_ms, 95)))
                acc["p99_step_ms"].append(float(np.percentile(durs_ms, 99)))
                acc["throughput_per_s"].append(float(len(durs_ms) / wall_s))
                acc["total_steps"].append(float(len(durs_ms)))
                rt_ms = float(runtime_df["duration_ns"].to_numpy().sum() / 1e6) if runtime_df.height else 0.0
                acc["cuda_runtime_ms"].append(rt_ms)
            trials.append(acc)

        summary = compare_scalar_metrics(
            [
                ("mean_step_ms", "ms", True),
                ("median_step_ms", "ms", True),
                ("p90_step_ms", "ms", True),
                ("p95_step_ms", "ms", True),
                ("p99_step_ms", "ms", True),
                ("throughput_per_s", "steps/s", False),
                ("total_steps", "count", False),
                ("cuda_runtime_ms", "ms", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return AspectResult(name=self.name, title=self.title, summary=summary)


@register_aspect(
    "kernel_diff",
    title="Kernel & Operator Family Diff",
    required_formats=_KERNEL_FORMATS,
    any_of_formats=True,
)
class KernelDiffAspect:
    name = "kernel_diff"
    title = "Kernel & Operator Family Diff"
    required_formats = _KERNEL_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        fam_dfs: list[pl.DataFrame] = []
        kern_dfs: list[pl.DataFrame] = []
        scope_dfs: list[pl.DataFrame] = []

        for g in groups:
            acc: dict[str, list[float]] = {
                "total_kernel_ms": [],
                "compute_kernel_ms": [],
                "comm_kernel_ms": [],
                "memcpy_ms": [],
                "kernel_launches": [],
                "unique_kernels": [],
            }
            t_fam, t_kern, t_scope = [], [], []
            for tmap in tables[g.group_id]:
                fmt = "nsys" if "nsys.kernels" in tmap else "kineto"
                steps_raw = pl.read_parquet(tmap[f"{fmt}.steps"])
                kernels_raw = pl.read_parquet(tmap[f"{fmt}.kernels"])
                _, kdf = _slice_steps_and_kernels(steps_raw, kernels_raw, config)

                if kdf.is_empty():
                    for k in acc:
                        acc[k].append(0.0)
                    continue

                tot_ms = float(kdf["duration_ns"].to_numpy().sum() / 1e6)
                comp_ms = float(kdf.filter(pl.col("category") == "compute")["duration_ns"].to_numpy().sum() / 1e6)
                comm_ms = float(kdf.filter(pl.col("category") == "comm")["duration_ns"].to_numpy().sum() / 1e6)
                mcpy_ms = float(kdf.filter(pl.col("category") == "memcpy")["duration_ns"].to_numpy().sum() / 1e6)

                acc["total_kernel_ms"].append(tot_ms)
                acc["compute_kernel_ms"].append(comp_ms)
                acc["comm_kernel_ms"].append(comm_ms)
                acc["memcpy_ms"].append(mcpy_ms)
                acc["kernel_launches"].append(float(kdf.height))
                acc["unique_kernels"].append(float(kdf["canonical_name"].n_unique()))

                t_fam.append(
                    kdf.group_by("op_family").agg(
                        (pl.col("duration_ns").sum().cast(pl.Float64) / 1e6).alias("duration_ms"),
                        pl.len().alias("launches"),
                    )
                )
                t_kern.append(
                    kdf.group_by("canonical_name").agg(
                        (pl.col("duration_ns").sum().cast(pl.Float64) / 1e6).alias("duration_ms"),
                        pl.len().alias("launches"),
                    )
                )
                scoped = kdf.filter(pl.col("scope") != "")
                if scoped.height:
                    t_scope.append(
                        scoped.group_by("scope").agg(
                            (pl.col("duration_ns").sum().cast(pl.Float64) / 1e6).alias("duration_ms"),
                            pl.len().alias("launches"),
                        )
                    )

            trials.append(acc)
            fam_dfs.append(
                pl.concat(t_fam).group_by("op_family").agg(pl.col("duration_ms").mean(), pl.col("launches").mean().round(0).cast(pl.Int64))
                if t_fam
                else pl.DataFrame({"op_family": [], "duration_ms": [], "launches": []})
            )
            kern_dfs.append(
                pl.concat(t_kern).group_by("canonical_name").agg(pl.col("duration_ms").mean(), pl.col("launches").mean().round(0).cast(pl.Int64))
                if t_kern
                else pl.DataFrame({"canonical_name": [], "duration_ms": [], "launches": []})
            )
            scope_dfs.append(
                pl.concat(t_scope).group_by("scope").agg(pl.col("duration_ms").mean(), pl.col("launches").mean().round(0).cast(pl.Int64))
                if t_scope
                else pl.DataFrame({"scope": [], "duration_ms": [], "launches": []})
            )

        summary = compare_scalar_metrics(
            [
                ("total_kernel_ms", "ms", True),
                ("compute_kernel_ms", "ms", True),
                ("comm_kernel_ms", "ms", True),
                ("memcpy_ms", "ms", True),
                ("kernel_launches", "calls", True),
                ("unique_kernels", "count", False),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        details = {
            "by_family": compare_entity_tables(labels, fam_dfs, key_col="op_family", val_col="duration_ms", count_col="launches"),
            "by_kernel": compare_entity_tables(labels, kern_dfs, key_col="canonical_name", val_col="duration_ms", count_col="launches"),
        }
        if any(df.height > 0 for df in scope_dfs):
            details["by_scope"] = compare_entity_tables(labels, scope_dfs, key_col="scope", val_col="duration_ms", count_col="launches")

        return AspectResult(name=self.name, title=self.title, summary=summary, details=details)


@register_aspect(
    "timeline_overlap",
    title="Timeline Utilization, Overlap & GPU Metrics",
    required_formats=_TIMELINE_FORMATS,
    any_of_formats=True,
)
class TimelineOverlapAspect:
    name = "timeline_overlap"
    title = "Timeline Utilization, Overlap & GPU Metrics"
    required_formats = _TIMELINE_FORMATS
    any_of_formats = True

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
                "compute_only_ms": [],
                "exposed_comm_ms": [],
                "overlapped_comm_ms": [],
                "gpu_idle_ms": [],
                "comm_overlap_pct": [],
                "gpu_busy_pct": [],
                "sm_active_median_pct": [],
                "tensor_active_median_pct": [],
                "sm_issue_median_pct": [],
            }
            w_sums = {"comp": 0.0, "comm": 0.0, "idle": 0.0, "win": 0.0}
            for tmap in tables[g.group_id]:
                steps_raw, kernels_raw, _ = _pick_timeline_tables(tmap)
                steps_df, kdf = _slice_steps_and_kernels(steps_raw, kernels_raw, config)
                if kdf.height:
                    w_start = int(steps_df["start_ns"].to_numpy().min()) if steps_df.height else None
                    w_end = int(steps_df["end_ns"].to_numpy().max()) if steps_df.height else None
                    comp = kdf.filter(pl.col("category") == "compute")
                    comm = kdf.filter(pl.col("category").is_in(["comm", "memcpy"]))
                    ov = compute_timeline_overlap(
                        comp["start_ns"].to_numpy(),
                        comp["end_ns"].to_numpy(),
                        comm["start_ns"].to_numpy(),
                        comm["end_ns"].to_numpy(),
                        window_start=w_start,
                        window_end=w_end,
                    )
                    acc["compute_only_ms"].append(ov.compute_only_ns / 1e6)
                    acc["exposed_comm_ms"].append(ov.comm_only_ns / 1e6)
                    acc["overlapped_comm_ms"].append(ov.overlap_ns / 1e6)
                    acc["gpu_idle_ms"].append(ov.idle_ns / 1e6)
                    acc["comm_overlap_pct"].append(ov.comm_overlap_ratio * 100.0)
                    acc["gpu_busy_pct"].append(ov.gpu_busy_ratio * 100.0)
                    w_sums["comp"] += ov.compute_total_ns
                    w_sums["comm"] += ov.comm_only_ns
                    w_sums["idle"] += ov.idle_ns
                    w_sums["win"] += ov.window_ns

                if "nsys.gpu_metrics" in tmap:
                    gm = pl.read_parquet(tmap["nsys.gpu_metrics"])
                    if gm.height:
                        for m_key, needle in (
                            ("sm_active_median_pct", "sm"),
                            ("tensor_active_median_pct", "tensor"),
                            ("sm_issue_median_pct", "issue"),
                        ):
                            if needle == "sm":
                                sub = gm.filter(pl.col("metric_name").str.to_lowercase().str.contains(r"sms? active"))
                            elif needle == "tensor":
                                sub = gm.filter(pl.col("metric_name").str.to_lowercase().str.contains("tensor active"))
                            else:
                                sub = gm.filter(pl.col("metric_name").str.to_lowercase().str.contains("sm issue"))
                            if sub.height:
                                acc[m_key].append(float(np.median(sub["value"].to_numpy())))

            trials.append(acc)
            win = w_sums["win"] or 1.0
            scalars[g.label] = {
                "share_Compute": w_sums["comp"] / win,
                "share_Exposed-Comm": w_sums["comm"] / win,
                "share_Idle-Bubble": w_sums["idle"] / win,
            }

        summary = compare_scalar_metrics(
            [
                ("compute_only_ms", "ms", True),
                ("exposed_comm_ms", "ms", True),
                ("overlapped_comm_ms", "ms", False),
                ("gpu_idle_ms", "ms", True),
                ("comm_overlap_pct", "%", False),
                ("gpu_busy_pct", "%", False),
                ("sm_active_median_pct", "%", False),
                ("tensor_active_median_pct", "%", False),
                ("sm_issue_median_pct", "%", False),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return AspectResult(name=self.name, title=self.title, summary=summary, scalars=scalars)


@register_aspect(
    "os_runtime",
    title="OS Runtime Lock & Thread Contention",
    required_formats=("nsys",),
)
class OsRuntimeAspect:
    name = "os_runtime"
    title = "OS Runtime Lock & Thread Contention"
    required_formats = ("nsys",)
    any_of_formats = False

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        call_dfs: list[pl.DataFrame] = []

        for g in groups:
            acc: dict[str, list[float]] = {
                "total_osrt_ms": [],
                "lock_sync_ms": [],
                "sleep_poll_ms": [],
                "osrt_calls": [],
            }
            per_trial: list[pl.DataFrame] = []
            for tmap in tables[g.group_id]:
                odf = pl.read_parquet(tmap["nsys.osrt"])
                if odf.is_empty():
                    continue
                acc["total_osrt_ms"].append(float(odf["duration_ns"].to_numpy().sum() / 1e6))
                lock_df = odf.filter(pl.col("name").str.to_lowercase().str.contains("mutex|cond|rwlock|sem|futex|spin"))
                poll_df = odf.filter(pl.col("name").str.to_lowercase().str.contains("sleep|poll|epoll|select|wait"))
                acc["lock_sync_ms"].append(float(lock_df["duration_ns"].to_numpy().sum() / 1e6) if lock_df.height else 0.0)
                acc["sleep_poll_ms"].append(float(poll_df["duration_ns"].to_numpy().sum() / 1e6) if poll_df.height else 0.0)
                acc["osrt_calls"].append(float(odf["calls"].to_numpy().sum()))
                per_trial.append(
                    odf.with_columns((pl.col("duration_ns").cast(pl.Float64) / 1e6).alias("duration_ms")).select("name", "duration_ms", "calls")
                )
            trials.append(acc)
            call_dfs.append(
                pl.concat(per_trial).group_by("name").agg(pl.col("duration_ms").mean(), pl.col("calls").mean().round(0).cast(pl.Int64))
                if per_trial
                else pl.DataFrame({"name": [], "duration_ms": [], "calls": []})
            )

        summary = compare_scalar_metrics(
            [
                ("total_osrt_ms", "ms", True),
                ("lock_sync_ms", "ms", True),
                ("sleep_poll_ms", "ms", True),
                ("osrt_calls", "calls", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        by_call = compare_entity_tables(labels, call_dfs, key_col="name", val_col="duration_ms", count_col="calls")
        return AspectResult(name=self.name, title=self.title, summary=summary, details={"by_syscall": by_call})


@register_aspect(
    "roofline_diff",
    title="Nsight Compute Roofline, Utilization & Warp Stalls",
    required_formats=("ncu",),
)
class RooflineDiffAspect:
    name = "roofline_diff"
    title = "Nsight Compute Roofline, Utilization & Warp Stalls"
    required_formats = ("ncu",)
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
        kern_dfs: list[pl.DataFrame] = []
        roof_pts: dict[str, pl.DataFrame] = {}

        for g in groups:
            acc: dict[str, list[float]] = {
                "total_kernel_us": [],
                "weighted_sm_pct": [],
                "weighted_dram_pct": [],
                "weighted_tensor_pct": [],
                "weighted_occupancy_pct": [],
                "overall_arith_intensity": [],
                "achieved_tflops": [],
                "stall_mem_pct": [],
                "stall_compute_pct": [],
            }
            group_kdfs: list[pl.DataFrame] = []
            for tmap in tables[g.group_id]:
                ndf = pl.read_parquet(tmap["ncu.ncu_kernels"])
                if ndf.is_empty():
                    continue
                dur = ndf["duration_ns"].to_numpy()
                tot_ns = float(np.sum(dur)) or 1.0
                w = dur / tot_ns

                tot_flops = float(np.sum(ndf["flops"].to_numpy()))
                tot_bytes = float(np.sum(ndf["dram_bytes"].to_numpy()))
                acc["total_kernel_us"].append(tot_ns / 1e3)
                acc["weighted_sm_pct"].append(float(np.sum(w * ndf["sm_throughput_pct"].to_numpy())))
                acc["weighted_dram_pct"].append(float(np.sum(w * ndf["dram_throughput_pct"].to_numpy())))
                acc["weighted_tensor_pct"].append(float(np.sum(w * ndf["tensor_pipe_pct"].to_numpy())))
                acc["weighted_occupancy_pct"].append(float(np.sum(w * ndf["occupancy_pct"].to_numpy())))
                acc["overall_arith_intensity"].append((tot_flops / tot_bytes) if tot_bytes > 0 else 0.0)
                acc["achieved_tflops"].append(tot_flops / tot_ns / 1e3)
                acc["stall_mem_pct"].append(float(np.sum(w * ndf["stall_mem_pct"].to_numpy())))
                acc["stall_compute_pct"].append(float(np.sum(w * ndf["stall_compute_pct"].to_numpy())))
                group_kdfs.append(ndf)

            trials.append(acc)
            sm_m = float(np.mean(acc["weighted_sm_pct"])) if acc["weighted_sm_pct"] else 0.0
            dr_m = float(np.mean(acc["weighted_dram_pct"])) if acc["weighted_dram_pct"] else 0.0
            tot_util = (sm_m + dr_m) or 1.0
            scalars[g.label] = {
                "share_Compute": sm_m / tot_util,
                "share_Memory-BW": dr_m / tot_util,
            }

            if group_kdfs:
                cat_df = pl.concat(group_kdfs)
                roof_pts[f"roofline_{g.label}"] = cat_df.group_by(["canonical_name", "op_family"]).agg(
                    (pl.col("duration_ns").sum() / 1e3).alias("duration_us"),
                    pl.col("arith_intensity").mean(),
                    pl.col("achieved_tflops").mean(),
                    pl.col("sm_throughput_pct").mean(),
                    pl.col("dram_throughput_pct").mean(),
                )
                kern_dfs.append(roof_pts[f"roofline_{g.label}"].select("canonical_name", "duration_us"))
            else:
                kern_dfs.append(pl.DataFrame({"canonical_name": [], "duration_us": []}))

        summary = compare_scalar_metrics(
            [
                ("total_kernel_us", "us", True),
                ("achieved_tflops", "TFLOP/s", False),
                ("overall_arith_intensity", "FLOP/B", False),
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
            "by_kernel": compare_entity_tables(labels, kern_dfs, key_col="canonical_name", val_col="duration_us"),
            **roof_pts,
        }
        return AspectResult(name=self.name, title=self.title, summary=summary, details=details, scalars=scalars)
