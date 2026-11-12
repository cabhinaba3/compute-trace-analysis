from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class OverlapBreakdown:
    window_ns: int
    compute_only_ns: int
    comm_only_ns: int
    overlap_ns: int
    idle_ns: int

    @property
    def compute_total_ns(self) -> int:
        return self.compute_only_ns + self.overlap_ns

    @property
    def comm_total_ns(self) -> int:
        return self.comm_only_ns + self.overlap_ns

    @property
    def comm_overlap_ratio(self) -> float:
        tot = self.comm_total_ns
        return (self.overlap_ns / tot) if tot > 0 else 0.0

    @property
    def gpu_busy_ratio(self) -> float:
        return ((self.window_ns - self.idle_ns) / self.window_ns) if self.window_ns > 0 else 0.0


def compute_timeline_overlap(
    comp_starts: np.ndarray,
    comp_ends: np.ndarray,
    comm_starts: np.ndarray,
    comm_ends: np.ndarray,
    *,
    window_start: int | None = None,
    window_end: int | None = None,
) -> OverlapBreakdown:
    """Exact O((N+M) log(N+M)) vectorized sweep-line decomposition of compute vs comm/memcpy intervals."""
    cs = np.asarray(comp_starts, dtype=np.int64)
    ce = np.asarray(comp_ends, dtype=np.int64)
    ms = np.asarray(comm_starts, dtype=np.int64)
    me = np.asarray(comm_ends, dtype=np.int64)

    valid_c = ce > cs
    cs, ce = cs[valid_c], ce[valid_c]
    valid_m = me > ms
    ms, me = ms[valid_m], me[valid_m]

    n, m = len(cs), len(ms)
    if n == 0 and m == 0:
        w = max(0, (window_end - window_start)) if (window_start is not None and window_end is not None) else 0
        return OverlapBreakdown(w, 0, 0, 0, w)

    all_starts = np.concatenate([cs, ms]) if (n and m) else (cs if n else ms)
    all_ends = np.concatenate([ce, me]) if (n and m) else (ce if n else me)
    w_start = int(np.min(all_starts)) if window_start is None else int(window_start)
    w_end = int(np.max(all_ends)) if window_end is None else int(window_end)
    if w_end <= w_start:
        return OverlapBreakdown(0, 0, 0, 0, 0)

    if n:
        cs = np.clip(cs, w_start, w_end)
        ce = np.clip(ce, w_start, w_end)
    if m:
        ms = np.clip(ms, w_start, w_end)
        me = np.clip(me, w_start, w_end)

    times = np.concatenate([cs, ce, ms, me])
    d_comp = np.concatenate(
        [
            np.ones(n, dtype=np.int32),
            -np.ones(n, dtype=np.int32),
            np.zeros(2 * m, dtype=np.int32),
        ]
    )
    d_comm = np.concatenate(
        [
            np.zeros(2 * n, dtype=np.int32),
            np.ones(m, dtype=np.int32),
            -np.ones(m, dtype=np.int32),
        ]
    )

    order = np.argsort(times, kind="mergesort")
    t_sorted = times[order]
    dt = np.diff(t_sorted)
    act_comp = np.cumsum(d_comp[order])[:-1] > 0
    act_comm = np.cumsum(d_comm[order])[:-1] > 0

    overlap = int(np.sum(dt[act_comp & act_comm]))
    comp_only = int(np.sum(dt[act_comp & ~act_comm]))
    comm_only = int(np.sum(dt[~act_comp & act_comm]))
    window = w_end - w_start
    idle = max(0, window - (overlap + comp_only + comm_only))
    return OverlapBreakdown(window, comp_only, comm_only, overlap, idle)


@dataclass(slots=True)
class ThreadRanges:
    """NVTX ranges of one thread, sorted by start, with a parent link to the enclosing range."""

    starts: np.ndarray
    ends: np.ndarray
    gids: np.ndarray
    parent: np.ndarray


def build_thread_ranges(rows: list[tuple[int, int, int]]) -> ThreadRanges:
    rows.sort(key=lambda r: (r[0], -r[1]))
    starts = np.array([r[0] for r in rows], dtype=np.int64)
    ends = np.array([r[1] for r in rows], dtype=np.int64)
    gids = np.array([r[2] for r in rows], dtype=np.int64)
    parent = np.full(len(rows), -1, dtype=np.int64)
    stack: list[int] = []
    for i in range(len(rows)):
        while stack and ends[stack[-1]] <= starts[i]:
            stack.pop()
        parent[i] = stack[-1] if stack else -1
        stack.append(i)
    return ThreadRanges(starts, ends, gids, parent)


def innermost_range(th: ThreadRanges, q: np.ndarray) -> np.ndarray:
    """Global id of the innermost range of this thread containing each query time (-1 if none).

    ``searchsorted`` finds the latest-starting range; if it has already ended, walk up to the enclosing range
    (this is what a plain "latest start" lookup gets wrong for sibling-then-outer cases).
    """
    out = np.full(len(q), -1, dtype=np.int64)
    idx = np.searchsorted(th.starts, q, side="right") - 1
    ok = idx >= 0
    safe = np.where(ok, idx, 0)
    inside = ok & (q <= th.ends[safe])
    out[inside] = th.gids[idx[inside]]
    for i in np.flatnonzero(ok & ~inside):
        k = int(th.parent[idx[i]])
        while k >= 0 and q[i] > th.ends[k]:
            k = int(th.parent[k])
        if k >= 0:
            out[i] = th.gids[k]
    return out


def range_paths(threads: dict[int, ThreadRanges], gid: np.ndarray, names: list[str], sep: str = " > ") -> np.ndarray:
    """Outermost-to-innermost chain of range names for each assigned range id ("" where ``gid`` is -1)."""
    parent: dict[int, int] = {}
    for th in threads.values():
        for i in range(len(th.gids)):
            parent[int(th.gids[i])] = int(th.gids[th.parent[i]]) if th.parent[i] >= 0 else -1
    cache: dict[int, str] = {-1: ""}

    def path(g: int) -> str:
        if g not in cache:
            up = parent.get(g, -1)
            cache[g] = f"{path(up)}{sep}{names[g]}" if up >= 0 else names[g]
        return cache[g]

    return np.array([path(int(g)) for g in gid], dtype=object)


def assign_ranges(
    threads: dict[int, ThreadRanges],
    gstart: np.ndarray,
    launch_t: np.ndarray,
    launch_tid: np.ndarray,
    gpu_t: np.ndarray,
) -> np.ndarray:
    """Range id per GPU activity: the innermost range on the *launching* thread at launch time.

    Launches are asynchronous, so the GPU start time says nothing about which CPU range issued the work. When
    the launch is unknown (no runtime row) or its thread has no enclosing range, fall back to the latest-starting
    containing range on any thread (queried at launch time if known, else at GPU start time).
    """
    n = len(gpu_t)
    out = np.full(n, -1, dtype=np.int64)
    for tid, th in threads.items():
        sel = launch_tid == tid
        if sel.any():
            out[sel] = innermost_range(th, launch_t[sel])
    todo = out < 0
    if todo.any() and threads:
        q = np.where(np.isnan(launch_t[todo]), gpu_t[todo], launch_t[todo]).astype(np.int64)
        best = np.full(len(q), -1, dtype=np.int64)
        best_start = np.full(len(q), -1, dtype=np.int64)
        for th in threads.values():
            cand = innermost_range(th, q)
            hit = cand >= 0
            starts = np.where(hit, gstart[np.where(hit, cand, 0)], -1)
            better = hit & (starts > best_start)
            best[better] = cand[better]
            best_start[better] = starts[better]
        out[np.flatnonzero(todo)] = best
    return out
