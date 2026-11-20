"""Optimization advice: what to try, and the most each change could save.

``diagnosis`` says what dominates a run; this module says what to do about
it and puts a number on it. Every recommendation carries a ceiling, the most
the change could save, found by taking the loss the trace shows out as a
whole. Three rules read a GPU timeline (nsys, rocprof-style or Kineto
tables), each against the compute that could have hidden the loss:

* ``launch_bound``: the GPU waits in short gaps between many tiny kernels
  (CUDA or HIP graphs, kernel fusion).
* ``exposed_comm``: communication that no compute runs beside (bucketing,
  overlap).
* ``exposed_copies``: host-device copies that no compute runs beside (pinned
  memory, prefetching).

Their ceilings are measured on the timeline, not solved on a graph, so they
say "measured-removal". For a Chakra or GOAL run the ``whatif`` aspect has
solved the critical path again with each op family removed, and each family
whose removal shortens it becomes a recommendation with ``"whatif-removal"``
as its basis; that needs the aspect to have run (``--aspects whatif``).

A timeline ceiling is the largest value over the (rank, device) timelines
(the slowest device decides the step), per step of that rank, and its
percentage is of that rank's mean step. Several trials of a group are
averaged. Gaps and copies are counted over the steps ``--warmup-steps``,
``--max-steps``, ``--ranks`` and ``--devices`` keep.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib
import re

import numpy as np
import polars as pl

from cta import models
from cta.analysis import copies as copy_names
from cta.analysis import intervals
from cta.analysis import step_slicing

# A kernel shorter than this is "tiny": its launch costs more than its work.
TINY_KERNEL_NS = 10_000
# A gap between kernels shorter than this is a launch gap; longer ones are
# the host doing something else (input, sync) and not what graphs fix.
LAUNCH_GAP_NS = 100_000
# launch_bound needs at least this share of tiny kernels, or a median kernel
# shorter than ``SHORT_MEDIAN_NS``.
TINY_SHARE = 0.3
SHORT_MEDIAN_NS = 20_000
# Communication of at most this many bytes per call, or (when the trace does
# not record sizes) at least ``MANY_CALLS_PER_STEP`` calls a step, is
# bucketing territory; larger or fewer calls are an overlap problem.
SMALL_MESSAGE_BYTES = 1 << 20
MANY_CALLS_PER_STEP = 16
MIN_BUCKET_CALLS_PER_STEP = 8
# Advice worth less than this share of a step is not given.
MIN_CEILING_PCT = 2.0
# At most this many recommendations per group.
MAX_PER_GROUP = 5

_COMM_FAMILY = re.compile(
    r"nccl|rccl|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|"
    r"broadcast|send|recv|comm|coll",
    re.IGNORECASE,
)


@dataclasses.dataclass(slots=True)
class Finding:
    """What one rule found on one timeline.

    Attributes:
        rule: The rule's name.
        title: What is wrong.
        action: What to try.
        ceiling_ns: The most it could save per step, in ns.
        step_ns: The mean step of the timeline's rank, in ns.
        evidence: The numbers behind it.
    """

    rule: str
    title: str
    action: str
    ceiling_ns: float
    step_ns: float
    evidence: dict[str, float | str]


def merge_intervals(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merges intervals into disjoint ones that touch no other.

    Args:
        starts: Start times.
        ends: End times; each at least its start.

    Returns:
        The sorted, disjoint union as (starts, ends).
    """
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts, kind="stable")
    sorted_starts = starts[order]
    sorted_ends = ends[order]
    running_end = np.maximum.accumulate(sorted_ends)
    opens = np.empty(len(sorted_starts), dtype=bool)
    opens[0] = True
    opens[1:] = sorted_starts[1:] > running_end[:-1]
    first = np.flatnonzero(opens)
    return sorted_starts[first], np.maximum.reduceat(sorted_ends, first)


def gaps_inside(
    kernel_starts: np.ndarray,
    kernel_ends: np.ndarray,
    step_starts: np.ndarray,
    step_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Finds the idle gaps of a device and how much of each is in a step.

    A gap is the time between the union of the kernels ending and the next
    kernel starting.

    Args:
        kernel_starts: Start times of the device's kernels.
        kernel_ends: End times of the device's kernels.
        step_starts: Start times of the rank's kept steps.
        step_ends: End times of the rank's kept steps.

    Returns:
        The length of every gap, and the part of it that lies inside a step
        (steps are merged first, so overlapping steps count once).
    """
    merged_starts, merged_ends = merge_intervals(kernel_starts, kernel_ends)
    if len(merged_starts) < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    gap_starts = merged_ends[:-1]
    gap_ends = merged_starts[1:]
    window_starts, window_ends = merge_intervals(step_starts, step_ends)
    before = np.concatenate([[0], np.cumsum(window_ends - window_starts)])

    def covered(time: np.ndarray) -> np.ndarray:
        """Returns how much of the steps lies before each time."""
        index = np.searchsorted(window_starts, time, side="right") - 1
        safe = np.maximum(index, 0)
        inside = np.clip(
            time - window_starts[safe],
            0,
            window_ends[safe] - window_starts[safe],
        )
        return np.where(index >= 0, before[safe] + inside, 0)

    return gap_ends - gap_starts, covered(gap_ends) - covered(gap_starts)


def _launch_bound(
    kernels: pl.DataFrame,
    step_starts: np.ndarray,
    step_ends: np.ndarray,
    step_count: int,
    step_ns: float,
) -> Finding | None:
    """Finds launch gaps between tiny kernels on one device, or None."""
    gap_length, gap_in_step = gaps_inside(
        kernels["start_ns"].to_numpy(),
        kernels["end_ns"].to_numpy(),
        step_starts,
        step_ends,
    )
    launch_gap_ns = float(gap_in_step[gap_length < LAUNCH_GAP_NS].sum())
    compute = kernels.filter(pl.col("category") == "compute")
    if compute.is_empty() or launch_gap_ns <= 0:
        return None
    durations = compute["duration_ns"].to_numpy()
    tiny_share = float((durations < TINY_KERNEL_NS).mean())
    median_ns = float(np.median(durations))
    if tiny_share < TINY_SHARE and median_ns >= SHORT_MEDIAN_NS:
        return None
    return Finding(
        rule="launch_bound",
        title="Launch-bound: the GPU waits between many tiny kernels",
        action=(
            "Capture the step in a CUDA/HIP graph (torch.cuda.CUDAGraph, "
            "torch.compile(mode='reduce-overhead')) and fuse small kernels "
            "(torch.compile) to cut the number of launches."
        ),
        ceiling_ns=launch_gap_ns / step_count,
        step_ns=step_ns,
        evidence={
            "launch_gap_ms_per_step": launch_gap_ns / step_count / 1e6,
            "gaps_per_step": float((gap_length < LAUNCH_GAP_NS).sum())
            / step_count,
            "kernels_per_step": compute.height / step_count,
            "median_kernel_us": median_ns / 1e3,
            "tiny_kernel_share": tiny_share,
        },
    )


def _comm_evidence(
    ops: pl.DataFrame, step_count: int
) -> tuple[str, dict[str, float | str], bool]:
    """Reads a rank's communication calls and decides bucketing or overlap.

    Args:
        ops: The rank's ``network_ops`` rows (may be empty).
        step_count: The rank's kept steps.

    Returns:
        The dominant operation (by time), the evidence, and whether the
        calls are many and small enough that bucketing is the advice.
    """
    if ops.is_empty():
        return "communication", {}, False
    dominant = (
        ops.group_by("op")
        .agg(pl.col("duration_ns").sum())
        .sort("duration_ns", descending=True)["op"][0]
    )
    sizes = ops["bytes"].to_numpy()
    sizes = sizes[sizes > 0]
    calls_per_step = ops.height / step_count
    evidence: dict[str, float | str] = {
        "dominant_op": dominant,
        "calls_per_step": calls_per_step,
    }
    if len(sizes):
        median_bytes = float(np.median(sizes))
        evidence["median_message_bytes"] = median_bytes
        is_small = (
            median_bytes <= SMALL_MESSAGE_BYTES
            and calls_per_step >= MIN_BUCKET_CALLS_PER_STEP
        )
    else:
        # nsys records no size for NCCL calls: only the call count is known.
        is_small = calls_per_step >= MANY_CALLS_PER_STEP
    return dominant, evidence, is_small


def _exposed(
    kernels: pl.DataFrame, selected: pl.Series, window: tuple[int, int]
) -> float:
    """Returns the time the selected kernels run with no compute beside them.

    Args:
        kernels: A device's kernels.
        selected: Boolean mask of the kernels to count.
        window: The rank's first step start and last step end.

    Returns:
        The exposed time in ns.
    """
    compute = kernels.filter(pl.col("category") == "compute")
    other = kernels.filter(selected)
    if other.is_empty():
        return 0.0
    overlap = intervals.compute_timeline_overlap(
        compute["start_ns"].to_numpy(),
        compute["end_ns"].to_numpy(),
        other["start_ns"].to_numpy(),
        other["end_ns"].to_numpy(),
        window_start=window[0],
        window_end=window[1],
    )
    return float(overlap.comm_only_ns)


def _timeline_findings(
    kernels: pl.DataFrame,
    rank_steps: pl.DataFrame,
    rank_ops: pl.DataFrame,
    rank: int,
    device: int,
) -> list[Finding]:
    """Applies the three rules to one (rank, device) timeline.

    Args:
        kernels: The timeline's kernels.
        rank_steps: The kept steps of the timeline's rank.
        rank_ops: The kept ``network_ops`` rows of that rank.
        rank: The timeline's rank.
        device: The timeline's device.

    Returns:
        The findings, at most one per rule.
    """
    step_count = rank_steps.height
    if step_count == 0 or kernels.is_empty():
        return []
    step_ns = float(np.mean(rank_steps["duration_ns"].to_numpy()))
    step_starts = rank_steps["start_ns"].to_numpy()
    step_ends = rank_steps["end_ns"].to_numpy()
    window = (int(step_starts.min()), int(step_ends.max()))
    where = f"rank {rank} device {device}"
    findings: list[Finding] = []

    launch = _launch_bound(kernels, step_starts, step_ends, step_count, step_ns)
    if launch:
        launch.evidence["timeline"] = where
        findings.append(launch)

    exposed_comm_ns = _exposed(kernels, kernels["category"] == "comm", window)
    if exposed_comm_ns > 0:
        dominant, evidence, is_small = _comm_evidence(rank_ops, step_count)
        evidence["exposed_comm_ms_per_step"] = (
            exposed_comm_ns / step_count / 1e6
        )
        evidence["timeline"] = where
        findings.append(
            Finding(
                rule="exposed_comm",
                title=f"Exposed {dominant} communication: no compute runs "
                "beside it",
                action=(
                    "Bucket the many small collectives into fewer, larger "
                    "ones (e.g. a larger DDP bucket_cap_mb, fused "
                    "reduce-scatter), then overlap them with the backward "
                    "pass."
                    if is_small
                    else "Overlap the collective with compute: start it "
                    "asynchronously as soon as its inputs are ready (DDP "
                    "bucket hooks, FSDP prefetch), and keep compute queued "
                    "on another stream while it runs."
                ),
                ceiling_ns=exposed_comm_ns / step_count,
                step_ns=step_ns,
                evidence=evidence,
            )
        )

    copies = kernels.filter(pl.col("category") == "memcpy")
    if copies.height:
        names = copies["name"].unique().to_list()
        directions = {name: copy_names.direction_of(name) for name in names}
        is_host = (kernels["category"] == "memcpy") & kernels[
            "name"
        ].replace_strict(directions, default="", return_dtype=pl.String).is_in(
            ["H2D", "D2H"]
        )
        exposed_copy_ns = _exposed(kernels, is_host, window)
        host_copies = kernels.filter(is_host)
        if exposed_copy_ns > 0:
            copy_ns = float(host_copies["duration_ns"].sum())
            copy_bytes = float(host_copies["bytes"].sum())
            pageable = float(
                host_copies["name"].str.contains("(?i)pageable").sum()
            )
            evidence = {
                "exposed_copy_ms_per_step": exposed_copy_ns / step_count / 1e6,
                "copies_per_step": host_copies.height / step_count,
                "mb_per_step": copy_bytes / step_count / 1e6,
                "timeline": where,
            }
            if copy_ns > 0 and copy_bytes > 0:
                evidence["achieved_gb_per_s"] = copy_bytes / copy_ns
            if pageable:
                evidence["pageable_copies"] = pageable
            if pageable:
                action = (
                    "The copies come from pageable host memory: pin it "
                    "(DataLoader pin_memory=True, tensor.pin_memory()) and "
                    "copy with non_blocking=True so the copy can overlap "
                    "compute."
                )
            else:
                action = (
                    "Issue the copy early on a side stream with "
                    "non_blocking=True and prefetch the next batch while the "
                    "current step computes, so compute covers it. If the "
                    "host buffers are pageable, pin them first (the "
                    "bandwidth shows whether they are)."
                )
            findings.append(
                Finding(
                    rule="exposed_copies",
                    title="Exposed host-device copies: no compute runs "
                    "beside them",
                    action=action,
                    ceiling_ns=exposed_copy_ns / step_count,
                    step_ns=step_ns,
                    evidence=evidence,
                )
            )
    return findings


def timeline_findings(
    steps: pl.DataFrame, kernels: pl.DataFrame, ops: pl.DataFrame
) -> list[Finding]:
    """Applies the timeline rules to every (rank, device) of a trial.

    For each rule the finding with the largest ceiling is kept, as the
    slowest device sets the step.

    Args:
        steps: The kept steps (``rank``, ``start_ns``, ``end_ns``,
            ``duration_ns``).
        kernels: The kept kernels with ``rank``, ``device``, ``category``,
            ``name``, ``bytes``, ``start_ns``, ``end_ns`` and
            ``duration_ns``.
        ops: The kept ``network_ops`` rows (may be empty).

    Returns:
        At most one finding per rule.
    """
    best: dict[str, Finding] = {}
    # Split once per rank: filtering per timeline costs a pass over every
    # step and call for each of the (rank, device) timelines.
    steps_of = {
        int(key[0]): frame  # type: ignore[index]
        for key, frame in steps.partition_by("rank", as_dict=True).items()
    }
    ops_of = {
        int(key[0]): frame  # type: ignore[index]
        for key, frame in ops.partition_by("rank", as_dict=True).items()
    }
    for (rank, device), timeline in kernels.group_by(
        ["rank", "device"], maintain_order=True
    ):
        rank_steps = steps_of.get(int(rank))
        if rank_steps is None:
            continue
        for finding in _timeline_findings(
            timeline,
            rank_steps,
            ops_of.get(int(rank), ops.clear()),
            int(rank),
            int(device),
        ):
            current = best.get(finding.rule)
            if current is None or finding.ceiling_ns > current.ceiling_ns:
                best[finding.rule] = finding
    return list(best.values())


def _read(
    table_paths: dict[str, pathlib.Path], key: str, schema: pl.Schema
) -> pl.DataFrame:
    """Reads a table, or an empty one of ``schema`` when it is absent."""
    if key in table_paths:
        return pl.read_parquet(table_paths[key])
    return pl.DataFrame(schema=schema)


def _group_findings(
    trials: Sequence[dict[str, pathlib.Path]], config: models.AnalysisConfig
) -> list[Finding]:
    """Returns a group's findings, averaged over its trials."""
    from cta import table_schemas

    per_rule: dict[str, list[Finding]] = {}
    for table_paths in trials:
        try:
            steps_raw, kernels_raw, _ = step_slicing.pick_timeline_tables(
                table_paths
            )
        except KeyError:
            continue
        steps, kernels = step_slicing.slice_steps_and_kernels(
            steps_raw, kernels_raw, config
        )
        if kernels.is_empty() or steps.is_empty():
            continue
        ops = _read(table_paths, "network_ops", table_schemas.NETWORK_OPS)
        _, ops = step_slicing.slice_steps_and_kernels(steps_raw, ops, config)
        for finding in timeline_findings(steps, kernels, ops):
            per_rule.setdefault(finding.rule, []).append(finding)
    merged: list[Finding] = []
    for findings in per_rule.values():
        first = findings[0]
        merged.append(
            dataclasses.replace(
                first,
                ceiling_ns=float(np.mean([f.ceiling_ns for f in findings])),
                step_ns=float(np.mean([f.step_ns for f in findings])),
            )
        )
    return merged


def whatif_recommendations(
    group_label: str, ceiling: pl.DataFrame, sweep: pl.DataFrame
) -> list[models.Recommendation]:
    """Turns a what-if sweep into recommendations.

    Args:
        group_label: The group the sweep was run on.
        ceiling: The ``ceiling`` table of the ``whatif`` aspect (see
            ``whatif.SweepResult``).
        sweep: The ``sweep`` table of the aspect.

    Returns:
        One recommendation per op family whose removal shortens the critical
        path by at least ``MIN_CEILING_PCT``, unranked.
    """
    if ceiling.is_empty() or sweep.is_empty():
        return []
    first = sweep.row(0, named=True)
    base_ms = first["makespan_ns"] * first["speedup"] / 1e6
    out: list[models.Recommendation] = []
    for row in ceiling.iter_rows(named=True):
        label = str(row["group"])
        if label.startswith("(") or row["ceiling_speedup"] <= 1.0:
            continue
        gain_pct = (1.0 - 1.0 / row["ceiling_speedup"]) * 100.0
        if gain_pct < MIN_CEILING_PCT:
            continue
        is_comm = bool(_COMM_FAMILY.search(label))
        knee = row["knee_factor"]
        out.append(
            models.Recommendation(
                group_label=group_label,
                priority=0,
                rule=f"whatif:{label}",
                title=(
                    f"{label} holds {row['critical_share_pct']:.0f}% of the "
                    "critical path"
                ),
                action=(
                    f"Overlap {label} with compute, fuse or bucket the "
                    "calls, or cut the bytes moved."
                    if is_comm
                    else f"Speed up {label}: fuse it, use a lower precision "
                    "or a better kernel, or move work off the critical path."
                ),
                ceiling_ms=base_ms * gain_pct / 100.0,
                ceiling_pct=gain_pct,
                per="run",
                basis="whatif-removal",
                evidence={
                    "time_share_pct": float(row["time_share_pct"]),
                    "critical_share_pct": float(row["critical_share_pct"]),
                    "ceiling_speedup": float(row["ceiling_speedup"]),
                    "knee_factor": (
                        "only removal" if knee == float("inf") else float(knee)
                    ),
                },
            )
        )
    return out


def _rank(
    group_label: str, recommendations: list[models.Recommendation]
) -> list[models.Recommendation]:
    """Sorts a group's recommendations by ceiling and numbers them."""
    ordered = sorted(recommendations, key=lambda r: -r.ceiling_ms)[
        :MAX_PER_GROUP
    ]
    return [
        dataclasses.replace(r, priority=i + 1) for i, r in enumerate(ordered)
    ]


def advise(
    groups: Sequence[models.TrialGroup],
    tables: dict[str, list[dict[str, pathlib.Path]]],
    config: models.AnalysisConfig,
    aspects: dict[str, models.AspectResult],
) -> list[models.Recommendation]:
    """Recommends optimizations for every group.

    Args:
        groups: The analysed groups; the first one is the baseline.
        tables: Per group id, one map of table key to Parquet path per trial.
        config: The run's options (they select the steps, ranks and devices
            the timeline rules read).
        aspects: The aspect results; the ``whatif`` aspect's sweep, if it
            ran, gives the baseline group graph-based advice.

    Returns:
        The recommendations of all groups, each group's ranked by ceiling
        and none worth less than ``MIN_CEILING_PCT`` of a step.
    """
    out: list[models.Recommendation] = []
    for index, group in enumerate(groups):
        found: list[models.Recommendation] = []
        for finding in _group_findings(tables[group.group_id], config):
            ceiling_ns = min(finding.ceiling_ns, finding.step_ns)
            pct = ceiling_ns / finding.step_ns * 100.0 if finding.step_ns else 0
            if pct < MIN_CEILING_PCT:
                continue
            found.append(
                models.Recommendation(
                    group_label=group.label,
                    priority=0,
                    rule=finding.rule,
                    title=finding.title,
                    action=finding.action,
                    ceiling_ms=ceiling_ns / 1e6,
                    ceiling_pct=pct,
                    per="step",
                    basis="measured-removal",
                    evidence=finding.evidence,
                )
            )
        whatif_result = aspects.get("whatif")
        if (
            index == 0
            and whatif_result is not None
            and "ceiling" in whatif_result.details
        ):
            found += whatif_recommendations(
                group.label,
                whatif_result.details["ceiling"],
                whatif_result.details["sweep"],
            )
        out += _rank(group.label, found)
    return out
