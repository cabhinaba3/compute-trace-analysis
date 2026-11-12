"""Device, kernel, timeline, OS runtime, and roofline comparison aspects."""

from __future__ import annotations

import warnings
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
    """Apply --warmup-steps/--max-steps per rank (ranks have independent clocks) and keep the matching kernels."""
    if steps_df.is_empty():
        return steps_df, kernels_df
    ordered = steps_df.sort("rank", "start_ns").with_columns(pl.col("start_ns").cum_count().over("rank").alias("_i"))
    most = int(ordered["_i"].to_numpy().max())
    if config.warmup_steps > 0 and most <= config.warmup_steps:
        warnings.warn(
            f"--warmup-steps={config.warmup_steps} but a rank has at most {most} step(s) "
            "(no NVTX step markers?); nothing was skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif config.warmup_steps > 0:
        ordered = ordered.filter(pl.col("_i") > config.warmup_steps)
    if config.max_steps is not None and config.max_steps > 0:
        ordered = ordered.with_columns(pl.col("start_ns").cum_count().over("rank").alias("_j")).filter(
            pl.col("_j") <= config.max_steps
        ).drop("_j")
    ordered = ordered.drop("_i")

    if kernels_df.is_empty() or ordered.height == steps_df.height:
        return ordered, kernels_df

    windows = ordered.group_by("rank").agg(pl.col("start_ns").min().alias("_w0"), pl.col("end_ns").max().alias("_w1"))
    by_step = kernels_df.join(ordered.select("rank", "step").unique().with_columns(pl.lit(True).alias("_ok")), on=["rank", "step"], how="left")
    by_step = by_step.join(windows, on="rank", how="left")
    in_window = (pl.col("start_ns") >= pl.col("_w0")) & (pl.col("end_ns") <= pl.col("_w1"))
    filtered_k = by_step.filter(pl.col("_ok").fill_null(False) | in_window).drop("_ok", "_w0", "_w1")
    return ordered, filtered_k


def _gpu_metrics_in_steps(gm: pl.DataFrame, steps_df: pl.DataFrame) -> pl.DataFrame:
    """Keep only GPU-metric samples inside each rank's step window.

    The sampler runs for the whole capture, so medians over every sample mostly measure idle time before and
    after the profiled steps. Falls back to all samples when nothing falls inside the window (e.g. sources whose
    sample timestamps are not on the kernel clock).
    """
    if gm.is_empty() or steps_df.is_empty() or "rank" not in gm.columns:
        return gm
    wins = steps_df.group_by("rank").agg(pl.col("start_ns").min().alias("_w0"), pl.col("end_ns").max().alias("_w1"))
    inside = gm.join(wins, on="rank", how="inner").filter(
        pl.col("timestamp_ns").is_between(pl.col("_w0"), pl.col("_w1"))
    ).drop("_w0", "_w1")
    return inside if inside.height else gm


def _duration_weighted_mean(df: pl.DataFrame, col: str) -> float | None:
    """Duration-weighted mean of ``col`` over the rows where it is not null (None if no row has it)."""
    sub = df.filter(pl.col(col).is_not_null())
    if sub.is_empty():
        return None
    total = float(sub["duration_ns"].sum() or 0.0)
    if total <= 0:
        return float(sub[col].mean())  # type: ignore[arg-type]
    return float((sub[col] * sub["duration_ns"]).sum()) / total


def _overlap_per_device(steps_df: pl.DataFrame, kdf: pl.DataFrame) -> tuple[int, int, int, int, int]:
    """Compute/comm overlap summed over every (rank, device) timeline.

    Each device is its own timeline and each rank has its own clock, so overlap is computed per (rank, device)
    inside that rank's step window and then summed: (compute_only, comm_only, overlap, idle, window) in ns.
    """
    windows = {
        int(r["rank"]): (int(r["w0"]), int(r["w1"]))
        for r in steps_df.group_by("rank").agg(pl.col("start_ns").min().alias("w0"), pl.col("end_ns").max().alias("w1")).iter_rows(named=True)
    }
    tot = [0, 0, 0, 0, 0]
    for (rank, _device), sub in kdf.group_by(["rank", "device"], maintain_order=True):
        w0, w1 = windows.get(int(rank), (None, None))  # type: ignore[call-overload]
        comp = sub.filter(pl.col("category") == "compute")
        comm = sub.filter(pl.col("category").is_in(["comm", "memcpy"]))
        ov = compute_timeline_overlap(
            comp["start_ns"].to_numpy(),
            comp["end_ns"].to_numpy(),
            comm["start_ns"].to_numpy(),
            comm["end_ns"].to_numpy(),
            window_start=w0,
            window_end=w1,
        )
        for i, v in enumerate((ov.compute_only_ns, ov.comm_only_ns, ov.overlap_ns, ov.idle_ns, ov.window_ns)):
            tot[i] += v
    return tot[0], tot[1], tot[2], tot[3], tot[4]


def _pick_timeline_tables(tmap: dict[str, Path]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    for fmt in _TIMELINE_FORMATS:
        if f"{fmt}.steps" in tmap:
            steps = pl.read_parquet(tmap[f"{fmt}.steps"])
            kernels = pl.read_parquet(tmap[f"{fmt}.kernels"]) if f"{fmt}.kernels" in tmap else pl.DataFrame()
            runtime = pl.read_parquet(tmap[f"{fmt}.runtime"]) if f"{fmt}.runtime" in tmap else pl.DataFrame()
            return steps, kernels, runtime
    raise KeyError("No timeline steps table found in bundle")


def _with_scope_level(kdf: pl.DataFrame, depth: int | None) -> pl.DataFrame:
    """Re-pick ``scope`` at a coarser level: the ``depth``-th enclosing range from the outermost (clamped to the innermost)."""
    if depth is None or "scope_path" not in kdf.columns:
        return kdf
    if depth < 1:
        raise ValueError(f"scope_depth must be >= 1, got {depth}")
    parts = pl.col("scope_path").str.split(" > ")
    picked = parts.list.get(depth - 1, null_on_oob=True).fill_null(parts.list.last())
    return kdf.with_columns(pl.when(pl.col("scope_path") == "").then(pl.lit("")).otherwise(picked).alias("scope"))


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
                # Ranks have independent clocks: throughput is steps/wall per rank, averaged over ranks.
                per_rank = steps_df.group_by("rank").agg(
                    pl.len().alias("n"), (pl.col("end_ns").max() - pl.col("start_ns").min()).alias("wall_ns")
                )
                thr = float(np.mean(per_rank["n"].to_numpy() / np.maximum(1e-9, per_rank["wall_ns"].to_numpy() / 1e9)))

                acc["mean_step_ms"].append(float(np.mean(durs_ms)))
                acc["median_step_ms"].append(float(np.median(durs_ms)))
                acc["p90_step_ms"].append(float(np.percentile(durs_ms, 90)))
                acc["p95_step_ms"].append(float(np.percentile(durs_ms, 95)))
                acc["p99_step_ms"].append(float(np.percentile(durs_ms, 99)))
                acc["throughput_per_s"].append(thr)
                acc["total_steps"].append(float(per_rank["n"].to_numpy().mean()))  # steps per rank
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
                kdf = _with_scope_level(kdf, config.scope_depth)
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
                    comp_only, comm_only, overlap, idle, window = _overlap_per_device(steps_df, kdf)
                    comm_total = comm_only + overlap
                    acc["compute_only_ms"].append(comp_only / 1e6)
                    acc["exposed_comm_ms"].append(comm_only / 1e6)
                    acc["overlapped_comm_ms"].append(overlap / 1e6)
                    acc["gpu_idle_ms"].append(idle / 1e6)
                    acc["comm_overlap_pct"].append(overlap / comm_total * 100.0 if comm_total else 0.0)
                    acc["gpu_busy_pct"].append((window - idle) / window * 100.0 if window else 0.0)
                    w_sums["comp"] += comp_only + overlap
                    w_sums["comm"] += comm_only
                    w_sums["idle"] += idle
                    w_sums["win"] += window

                if "nsys.gpu_metrics" in tmap:
                    gm = _gpu_metrics_in_steps(pl.read_parquet(tmap["nsys.gpu_metrics"]), steps_df)
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
        find_pts: dict[str, pl.DataFrame] = {}

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
            group_findings: list[pl.DataFrame] = []
            for tmap in tables[g.group_id]:
                ndf = pl.read_parquet(tmap["ncu.ncu_kernels"])
                if ndf.is_empty():
                    continue
                tot_ns = float(ndf["duration_ns"].sum() or 0.0)
                acc["total_kernel_us"].append(tot_ns / 1e3)
                # Missing metrics are null (not 0): average only over kernels that report the metric.
                for key, col in (
                    ("weighted_sm_pct", "sm_throughput_pct"),
                    ("weighted_dram_pct", "dram_throughput_pct"),
                    ("weighted_tensor_pct", "tensor_pipe_pct"),
                    ("weighted_occupancy_pct", "occupancy_pct"),
                    ("stall_mem_pct", "stall_mem_pct"),
                    ("stall_compute_pct", "stall_compute_pct"),
                ):
                    wm = _duration_weighted_mean(ndf, col)
                    if wm is not None:
                        acc[key].append(wm)
                with_flops = ndf.filter(pl.col("flops").is_not_null())
                if with_flops.height:
                    flops = float(with_flops["flops"].sum())
                    dur = float(with_flops["duration_ns"].sum() or 0.0)
                    both = with_flops.filter(pl.col("dram_bytes").is_not_null() & (pl.col("dram_bytes") > 0))
                    if both.height:  # AI = total FLOPs / total DRAM bytes of the kernels that have both
                        acc["overall_arith_intensity"].append(float(both["flops"].sum()) / float(both["dram_bytes"].sum()))
                    if dur > 0:
                        acc["achieved_tflops"].append(flops / dur / 1e3)
                group_kdfs.append(ndf)
                if "ncu.ncu_findings" in tmap:
                    fdf = pl.read_parquet(tmap["ncu.ncu_findings"])
                    if fdf.height:
                        group_findings.append(fdf)

            if group_findings:  # NCU's own guided-analysis rules, summarised per rule (report-based inputs only)
                find_pts[f"findings_{g.label}"] = (
                    pl.concat(group_findings)
                    .filter(pl.col("level").is_in(["warning", "optimization", "error"]))
                    .group_by(["rule", "title", "level"])
                    .agg(pl.len().alias("kernels"), pl.col("speedup_pct").max().alias("max_speedup_pct"))
                    .sort("kernels", descending=True)
                )
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
                roof_pts[f"roofline_{g.label}"] = (
                    cat_df.group_by(["canonical_name", "op_family"])
                    .agg(
                        (pl.col("duration_ns").sum() / 1e3).alias("duration_us"),
                        pl.col("flops").sum().alias("_flops"),
                        pl.col("dram_bytes").sum().alias("_bytes"),
                        pl.col("flops").is_not_null().any().alias("_has_flops"),
                        pl.col("duration_ns").sum().alias("_dur_ns"),
                        (pl.col("sm_throughput_pct") * pl.col("duration_ns")).sum().alias("_sm_w"),
                        (pl.col("dram_throughput_pct") * pl.col("duration_ns")).sum().alias("_dr_w"),
                    )
                    .with_columns(
                        # per-entity aggregates: sum(FLOPs)/sum(bytes) and sum(FLOPs)/sum(time), not means of ratios
                        pl.when(pl.col("_has_flops") & (pl.col("_bytes") > 0)).then(pl.col("_flops") / pl.col("_bytes")).alias("arith_intensity"),
                        pl.when(pl.col("_has_flops") & (pl.col("_dur_ns") > 0)).then(pl.col("_flops") / pl.col("_dur_ns") / 1e3).alias("achieved_tflops"),
                        (pl.col("_sm_w") / pl.col("_dur_ns")).alias("sm_throughput_pct"),
                        (pl.col("_dr_w") / pl.col("_dur_ns")).alias("dram_throughput_pct"),
                    )
                    .select("canonical_name", "op_family", "duration_us", "arith_intensity", "achieved_tflops", "sm_throughput_pct", "dram_throughput_pct")
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
            **find_pts,
        }
        return AspectResult(name=self.name, title=self.title, summary=summary, details=details, scalars=scalars)
