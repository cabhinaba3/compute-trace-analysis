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
