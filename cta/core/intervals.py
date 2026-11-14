"""Interval arithmetic: compute/comm overlap and NVTX-style range nesting."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(slots=True, frozen=True)
class OverlapBreakdown:
    """How a time window splits into compute, comm, overlap and idle time.

    Attributes:
        window_ns: Length of the analysed window.
        compute_only_ns: Time with compute active and no comm.
        comm_only_ns: Time with comm active and no compute.
        overlap_ns: Time with compute and comm both active.
        idle_ns: Time with neither active.
    """

    window_ns: int
    compute_only_ns: int
    comm_only_ns: int
    overlap_ns: int
    idle_ns: int

    @property
    def compute_total_ns(self) -> int:
        """The total time with compute active, overlapped or not."""
        return self.compute_only_ns + self.overlap_ns

    @property
    def comm_total_ns(self) -> int:
        """The total time with comm active, overlapped or not."""
        return self.comm_only_ns + self.overlap_ns

    @property
    def comm_overlap_ratio(self) -> float:
        """The fraction of comm time hidden behind compute (0 if no comm)."""
        comm_total_ns = self.comm_total_ns
        return (self.overlap_ns / comm_total_ns) if comm_total_ns > 0 else 0.0

    @property
    def gpu_busy_ratio(self) -> float:
        """The fraction of the window that is not idle (0 if empty)."""
        return (
            ((self.window_ns - self.idle_ns) / self.window_ns)
            if self.window_ns > 0
            else 0.0
        )


def compute_timeline_overlap(
    compute_starts: np.ndarray,
    compute_ends: np.ndarray,
    comm_starts: np.ndarray,
    comm_ends: np.ndarray,
    *,
    window_start: int | None = None,
    window_end: int | None = None,
) -> OverlapBreakdown:
    """Splits a window into compute-only, comm-only, overlap and idle time.

    An exact, vectorised sweep line in O((N + M) log(N + M)). Intervals may
    overlap within a class; empty or inverted intervals are ignored and the
    rest are clipped to the window.

    Args:
        compute_starts: Start times of the compute intervals (ns).
        compute_ends: End times of the compute intervals (ns).
        comm_starts: Start times of the comm/memcpy intervals (ns).
        comm_ends: End times of the comm/memcpy intervals (ns).
        window_start: Start of the window; defaults to the earliest start.
        window_end: End of the window; defaults to the latest end.

    Returns:
        The breakdown. Without valid intervals, the whole window (0 unless
        both bounds are given) is idle; an empty window gives all zeros.
    """
    compute_start = np.asarray(compute_starts, dtype=np.int64)
    compute_end = np.asarray(compute_ends, dtype=np.int64)
    comm_start = np.asarray(comm_starts, dtype=np.int64)
    comm_end = np.asarray(comm_ends, dtype=np.int64)

    is_valid_compute = compute_end > compute_start
    compute_start = compute_start[is_valid_compute]
    compute_end = compute_end[is_valid_compute]
    is_valid_comm = comm_end > comm_start
    comm_start = comm_start[is_valid_comm]
    comm_end = comm_end[is_valid_comm]

    num_compute, num_comm = len(compute_start), len(comm_start)
    if num_compute == 0 and num_comm == 0:
        idle_window_ns = (
            max(0, (window_end - window_start))
            if (window_start is not None and window_end is not None)
            else 0
        )
        return OverlapBreakdown(idle_window_ns, 0, 0, 0, idle_window_ns)

    if num_compute and num_comm:
        all_starts = np.concatenate([compute_start, comm_start])
        all_ends = np.concatenate([compute_end, comm_end])
    elif num_compute:
        all_starts, all_ends = compute_start, compute_end
    else:
        all_starts, all_ends = comm_start, comm_end
    start_ns = (
        int(np.min(all_starts)) if window_start is None else int(window_start)
    )
    end_ns = int(np.max(all_ends)) if window_end is None else int(window_end)
    if end_ns <= start_ns:
        return OverlapBreakdown(0, 0, 0, 0, 0)

    if num_compute:
        compute_start = np.clip(compute_start, start_ns, end_ns)
        compute_end = np.clip(compute_end, start_ns, end_ns)
    if num_comm:
        comm_start = np.clip(comm_start, start_ns, end_ns)
        comm_end = np.clip(comm_end, start_ns, end_ns)

    # Each boundary is an event that opens (+1) or closes (-1) an interval
    # of one class; the running sum is the number of active intervals.
    times = np.concatenate([compute_start, compute_end, comm_start, comm_end])
    compute_delta = np.concatenate(
        [
            np.ones(num_compute, dtype=np.int32),
            -np.ones(num_compute, dtype=np.int32),
            np.zeros(2 * num_comm, dtype=np.int32),
        ]
    )
    comm_delta = np.concatenate(
        [
            np.zeros(2 * num_compute, dtype=np.int32),
            np.ones(num_comm, dtype=np.int32),
            -np.ones(num_comm, dtype=np.int32),
        ]
    )

    order = np.argsort(times, kind="mergesort")
    segment_ns = np.diff(times[order])
    is_compute_active = np.cumsum(compute_delta[order])[:-1] > 0
    is_comm_active = np.cumsum(comm_delta[order])[:-1] > 0

    overlap_ns = int(np.sum(segment_ns[is_compute_active & is_comm_active]))
    compute_only_ns = int(
        np.sum(segment_ns[is_compute_active & ~is_comm_active])
    )
    comm_only_ns = int(np.sum(segment_ns[~is_compute_active & is_comm_active]))
    window_ns = end_ns - start_ns
    idle_ns = max(0, window_ns - (overlap_ns + compute_only_ns + comm_only_ns))
    return OverlapBreakdown(
        window_ns, compute_only_ns, comm_only_ns, overlap_ns, idle_ns
    )


@dataclasses.dataclass(slots=True)
class ThreadRanges:
    """NVTX ranges of one thread, sorted by start, linked to their parents.

    Attributes:
        starts: Start time of each range, ascending.
        ends: End time of each range.
        range_ids: Global id of each range.
        parent: Index (into these arrays) of the enclosing range, or -1.
    """

    starts: np.ndarray
    ends: np.ndarray
    range_ids: np.ndarray
    parent: np.ndarray


def build_thread_ranges(rows: list[tuple[int, int, int]]) -> ThreadRanges:
    """Sorts one thread's ranges and links each to its enclosing range.

    Ranges are sorted by start, longer first on equal starts, so a parent
    precedes its children. A range's parent is the latest earlier range
    that has not ended by the range's start.

    Args:
        rows: ``(start, end, global id)`` per range. Sorted in place.

    Returns:
        The thread's ranges.
    """
    rows.sort(key=lambda row: (row[0], -row[1]))
    starts = np.array([row[0] for row in rows], dtype=np.int64)
    ends = np.array([row[1] for row in rows], dtype=np.int64)
    gids = np.array([row[2] for row in rows], dtype=np.int64)
    parent = np.full(len(rows), -1, dtype=np.int64)
    open_ranges: list[int] = []
    for i in range(len(rows)):
        while open_ranges and ends[open_ranges[-1]] <= starts[i]:
            open_ranges.pop()
        parent[i] = open_ranges[-1] if open_ranges else -1
        open_ranges.append(i)
    return ThreadRanges(starts, ends, gids, parent)


def innermost_range(
    thread_ranges: ThreadRanges, query_ns: np.ndarray
) -> np.ndarray:
    """Returns the innermost range of a thread containing each query time.

    ``searchsorted`` finds the latest-starting range; if that one has
    already ended, walk up to the enclosing range. (A plain "latest start"
    lookup gets this wrong when a sibling range ended before the query time
    but the outer range is still open.)

    Args:
        thread_ranges: The thread's ranges.
        query_ns: Query times.

    Returns:
        The global id of the innermost containing range per query time, or
        -1 where no range contains it.
    """
    range_ids = np.full(len(query_ns), -1, dtype=np.int64)
    latest = np.searchsorted(thread_ranges.starts, query_ns, side="right") - 1
    has_latest = latest >= 0
    safe_latest = np.where(has_latest, latest, 0)
    is_inside = has_latest & (query_ns <= thread_ranges.ends[safe_latest])
    range_ids[is_inside] = thread_ranges.range_ids[latest[is_inside]]
    for i in np.flatnonzero(has_latest & ~is_inside):
        ancestor = int(thread_ranges.parent[latest[i]])
        while ancestor >= 0 and query_ns[i] > thread_ranges.ends[ancestor]:
            ancestor = int(thread_ranges.parent[ancestor])
        if ancestor >= 0:
            range_ids[i] = thread_ranges.range_ids[ancestor]
    return range_ids


def range_paths(
    threads: dict[int, ThreadRanges],
    range_ids: np.ndarray,
    names: list[str],
    separator: str = " > ",
) -> np.ndarray:
    """Returns the outermost-to-innermost chain of range names per range id.

    Args:
        threads: Ranges per thread id; they define the nesting.
        range_ids: Range ids to describe; -1 means no range.
        names: Range name per global range id.
        separator: Separator between the names in a chain.

    Returns:
        An object array of strings, ``""`` where ``range_ids`` is -1.
    """
    parent_by_gid: dict[int, int] = {}
    for thread in threads.values():
        for i in range(len(thread.range_ids)):
            parent_by_gid[int(thread.range_ids[i])] = (
                int(thread.range_ids[thread.parent[i]])
                if thread.parent[i] >= 0
                else -1
            )
    path_by_gid: dict[int, str] = {-1: ""}

    def path_of(range_id: int) -> str:
        """Returns (and memoises) the name chain of one range id."""
        if range_id not in path_by_gid:
            parent_id = parent_by_gid.get(range_id, -1)
            path_by_gid[range_id] = (
                f"{path_of(parent_id)}{separator}{names[range_id]}"
                if parent_id >= 0
                else names[range_id]
            )
        return path_by_gid[range_id]

    return np.array(
        [path_of(int(range_id)) for range_id in range_ids], dtype=object
    )


def assign_ranges(
    threads: dict[int, ThreadRanges],
    gpu_start_ns: np.ndarray,
    launch_ns: np.ndarray,
    launch_thread_id: np.ndarray,
    gpu_ns: np.ndarray,
) -> np.ndarray:
    """Returns the range each GPU activity belongs to.

    That is the innermost range on the *launching* thread at launch time.
    Launches are asynchronous, so the GPU start time says nothing about
    which CPU range issued the work. When the launch is unknown (no runtime
    row) or its thread has no enclosing range, fall back to the
    latest-starting containing range on any thread (queried at launch time
    if known, else at GPU start time).

    Args:
        threads: Ranges per thread id.
        gpu_start_ns: Start time per global range id.
        launch_ns: Launch time per activity; NaN when unknown.
        launch_thread_id: Launching thread id per activity.
        gpu_ns: GPU start time per activity.

    Returns:
        The global range id per activity, or -1 when no range contains it.
    """
    range_ids = np.full(len(gpu_ns), -1, dtype=np.int64)
    for thread_id, thread in threads.items():
        is_on_thread = launch_thread_id == thread_id
        if is_on_thread.any():
            range_ids[is_on_thread] = innermost_range(
                thread, launch_ns[is_on_thread]
            )
    is_unassigned = range_ids < 0
    if is_unassigned.any() and threads:
        query_t = np.where(
            np.isnan(launch_ns[is_unassigned]),
            gpu_ns[is_unassigned],
            launch_ns[is_unassigned],
        ).astype(np.int64)
        best_range = np.full(len(query_t), -1, dtype=np.int64)
        best_start = np.full(len(query_t), -1, dtype=np.int64)
        for thread in threads.values():
            candidate = innermost_range(thread, query_t)
            is_hit = candidate >= 0
            candidate_start = np.where(
                is_hit, gpu_start_ns[np.where(is_hit, candidate, 0)], -1
            )
            is_better = is_hit & (candidate_start > best_start)
            best_range[is_better] = candidate[is_better]
            best_start[is_better] = candidate_start[is_better]
        range_ids[np.flatnonzero(is_unassigned)] = best_range
    return range_ids
