"""Fixtures of the simulator tests, and a plain-Python twin of the engine.

``Oracle`` follows native/src/sim/engine.cpp step for step for the LogGOPS
and alpha-beta models (several CPUs and NICs per rank, eager and rendezvous
messages, intra-node parameters, NIC scale factors) and for collectives with
recorded costs, without OS noise. The engine must match it to the ns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import heapq
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from cta.synth import training

CALC, SEND, RECV, COLL, NOP = range(5)
DONE, ARRIVE, WAKE, READY = 0, 2, 4, 5  # the engine's event order
MSG = 1 << 31


def ns(x: float) -> int:
    return math.floor(x + 0.5)


def minus_one(nbytes: int) -> float:
    return float(max(nbytes - 1, 0))


class Oracle:
    """Runs a graph given as the engine's arrays and parameters."""

    def __init__(
        self,
        g: Mapping[str, np.ndarray],
        offsets: np.ndarray,
        members: np.ndarray,
        params: Mapping[str, Any],
    ) -> None:
        """Prepares a run."""
        self.g = g = {k: np.asarray(v).tolist() for k, v in g.items()}
        self.p = p = dict(params)
        self.model = p.get("model", "loggops")
        self.offsets, self.members = list(offsets), list(members)
        n = len(g["kind"])
        ranks = max(
            [1, *(r + 1 for r in g["rank"]), *(m + 1 for m in self.members)]
            + [
                g["peer"][i] + 1
                for i in range(n)
                if g["kind"][i] in (SEND, RECV)
            ]
        )
        self.cpus = [1] * ranks
        self.nics = [1] * ranks
        for r, c, k in zip(g["rank"], g["cpu"], g["nic"], strict=True):
            self.cpus[r] = max(self.cpus[r], c + 1)
            self.nics[r] = max(self.nics[r], k + 1)
        self.free: dict[tuple[str, int, int], int] = {}
        self.waiters: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
        self.wake: set[tuple[str, int, int]] = set()
        self.succ: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        self.pending = [[0, 0] for _ in range(n)]
        for s, d, k in zip(
            g["edge_src"], g["edge_dst"], g["edge_kind"], strict=True
        ):
            self.succ[s].append((d, k))
            self.pending[d][k] += 1
        sends = [i for i in range(n) if g["kind"][i] == SEND]
        self.msg_of = {op: m for m, op in enumerate(sends)}
        self.msgs: list[dict[str, Any]] = [{"op": op} for op in sends]
        seq: dict[tuple[int, int], int] = {}
        self.inst: dict[int, tuple[int, int]] = {}
        for i in range(n):
            if g["kind"][i] == COLL:
                key = (g["rank"][i], g["coll_group"][i])
                self.inst[i] = (g["coll_group"][i], seq.get(key, 0))
                seq[key] = seq.get(key, 0) + 1
        self.arrived: dict[tuple[int, int], int] = {}
        self.ready: list[int | None] = [None] * n
        self.start: list[int | None] = [None] * n
        self.end: list[int | None] = [None] * n
        self.posted: dict[tuple[int, int, int], list[int]] = {}
        self.unexpected: dict[tuple[int, int, int], list[int]] = {}
        self.events: list[tuple[int, int, int]] = []
        self.now = 0

    def _lg(self, a: int, b: int) -> dict[str, float]:
        k, p = self.p.get("ranks_per_node", 1), self.p
        base = {
            x: float(p.get(f"loggops.{x}", d))
            for x, d in zip("LogGO", (2500, 1500, 1000, 6, 0), strict=True)
        }
        if k > 1 and a // k == b // k:
            return {
                x: float(p.get(f"loggops.intra_{x}", v))
                for x, v in base.items()
            }
        return base

    def _scale(self, a: int, b: int) -> float:
        s = self.p.get("noise.nic_scale", [])
        return min(s[a] if a < len(s) else 1.0, s[b] if b < len(s) else 1.0)

    def _alpha_beta(self, a: int, b: int) -> tuple[float, float]:
        k, p = self.p.get("ranks_per_node", 1), self.p
        alpha, beta = (
            p.get("alpha_beta.alpha", 5500),
            p.get("alpha_beta.beta", 6),
        )
        if k > 1 and a // k == b // k:
            alpha = p.get("alpha_beta.intra_alpha", alpha)
            beta = p.get("alpha_beta.intra_beta", beta)
        return alpha, beta / self._scale(a, b)

    def _required(self, act: int) -> list[tuple[str, int, int]]:
        g = self.g
        if act & MSG:
            s = self.msgs[act & ~MSG]["op"]
            dst = g["peer"][s]
            cpu = ("cpu", dst, min(g["cpu"][s], self.cpus[dst] - 1))
            nic = ("recv", dst, min(g["nic"][s], self.nics[dst] - 1))
            return [] if self.model == "alpha_beta" else [cpu, nic]
        r = g["rank"][act]
        cpu = ("cpu", r, min(g["cpu"][act], self.cpus[r] - 1))
        if g["kind"][act] == CALC:
            return [cpu]
        return (
            []
            if self.model == "alpha_beta"
            else [cpu, ("send", r, min(g["nic"][act], self.nics[r] - 1))]
        )

    def _since(self, act: int) -> int:
        value = (
            self.msgs[act & ~MSG]["arrive"] if act & MSG else self.ready[act]
        )
        assert value is not None
        return value

    def _id(self, res: tuple[str, int, int]) -> tuple[int, int, int]:
        # The engine numbers CPUs, then send NICs, then recv NICs, rank by rank.
        order = {"cpu": 0, "send": 1, "recv": 2}[res[0]]
        return (order, res[1], res[2])

    def _push(self, kind: int, ident: Any, t: int) -> None:
        heapq.heappush(self.events, (t, kind, ident))

    def _request(self, act: int) -> None:
        res = self._required(act)
        busy = [r for r in res if self.free.get(r, 0) > self.now]
        block = max(busy, key=lambda r: self.free[r], default=None)
        if block is None:
            block = next((r for r in res if self.waiters.get(r)), None)
        if block is None:
            self._begin(act)
        else:
            self._enqueue(act, self._since(act), block)

    def _enqueue(self, act: int, since: int, res: tuple[str, int, int]) -> None:
        heapq.heappush(self.waiters.setdefault(res, []), (since, act))
        if res not in self.wake:
            self.wake.add(res)
            self._push(
                WAKE, self._id(res), max(self.free.get(res, 0), self.now)
            )

    def _on_wake(self, ident: Any) -> None:
        res = (("cpu", "send", "recv")[ident[0]], ident[1], ident[2])
        self.wake.discard(res)
        queue = self.waiters.get(res, [])
        while queue and self.free.get(res, 0) <= self.now:
            since, act = heapq.heappop(queue)
            other = next(
                (
                    o
                    for o in self._required(act)
                    if o != res
                    and (
                        self.free.get(o, 0) > self.now
                        or (
                            self.waiters.get(o)
                            and self.waiters[o][0] < (since, act)
                        )
                    )
                ),
                None,
            )
            if other is None:
                self._begin(act)
            else:
                self._enqueue(act, since, other)
        if queue and res not in self.wake:
            self.wake.add(res)
            self._push(WAKE, ident, max(self.free.get(res, 0), self.now))

    def _begin(self, act: int) -> None:
        if act & MSG:
            self._begin_proc(act & ~MSG)
        elif self.g["kind"][act] == CALC:
            self._start_op(act)
            end = self.now + self.g["duration"][act]
            self.free[self._required(act)[0]] = end
            self._push(DONE, act, end)
        else:
            self._begin_send(act)

    def _start_op(self, op: int) -> None:
        self.start[op] = self.now
        for t, k in self.succ[op]:
            if k == 1:
                self.pending[t][1] -= 1
                if self.pending[t] == [0, 0]:
                    self._push(READY, t, self.now)

    def _on_done(self, op: int) -> None:
        self.end[op] = self.now
        for t, k in self.succ[op]:
            if k == 0:
                self.pending[t][0] -= 1
                if self.pending[t] == [0, 0]:
                    self._push(READY, t, self.now)

    def _on_ready(self, op: int) -> None:
        g = self.g
        self.ready[op] = self.now
        kind = g["kind"][op]
        if kind == NOP:
            self._start_op(op)
            self._push(DONE, op, self.now)
        elif kind in (CALC, SEND):
            self._request(op)
        elif kind == RECV:
            self._start_op(op)
            key = (g["rank"][op], g["peer"][op], g["tag"][op])
            if self.unexpected.get(key):
                self._complete(op, self.unexpected[key].pop(0))
            else:
                self.posted.setdefault(key, []).append(op)
        else:
            inst = self.inst[op]
            self.arrived[inst] = self.arrived.get(inst, 0) + 1
            grp = g["coll_group"][op]
            if self.arrived[inst] == self.offsets[grp + 1] - self.offsets[grp]:
                ops = [x for x, i in self.inst.items() if i == inst]
                shortest = min(g["duration"][x] for x in ops)
                for x in sorted(ops):
                    self._start_op(x)
                    self._push(DONE, x, self.now + shortest)

    def _begin_send(self, op: int) -> None:
        g = self.g
        self._start_op(op)
        m = self.msgs[self.msg_of[op]]
        src, dst, b = g["rank"][op], g["peer"][op], g["bytes"][op]
        if self.model == "alpha_beta":
            alpha, beta = self._alpha_beta(src, dst)
            self._push(ARRIVE, op, self.now + ns(alpha + b * beta))
            self._push(DONE, op, self.now)
            return
        lg = self._lg(src, dst)
        cpu, nic = self._required(op)
        m["cpu_end"] = self.now + ns(lg["o"] + minus_one(b) * lg["O"])
        self.free[cpu] = m["cpu_end"]
        self.free[nic] = self.now + ns(
            lg["g"] + minus_one(b) * lg["G"] / self._scale(src, dst)
        )
        self._push(ARRIVE, op, self.now + ns(lg["o"]) + ns(lg["L"]))
        m["rendezvous"] = b > self.p.get("loggops.S", 65535)
        if not m["rendezvous"]:
            self._push(DONE, op, m["cpu_end"])

    def _begin_proc(self, i: int) -> None:
        g, m = self.g, self.msgs[i]
        s = m["op"]
        src, dst = g["rank"][s], g["peer"][s]
        m["proc_start"] = m["proc_end"] = self.now
        if self.model != "alpha_beta":
            lg = self._lg(src, dst)
            b1, cpu_nic = minus_one(g["bytes"][s]), self._required(i | MSG)
            per_byte = b1 * lg["G"] / self._scale(src, dst)
            self.free[cpu_nic[1]] = self.now + ns(lg["g"] + per_byte)
            m["proc_end"] = self.now + ns(lg["o"] + max(b1 * lg["O"], per_byte))
            self.free[cpu_nic[0]] = m["proc_end"]
        key = (dst, src, g["tag"][s])
        if self.posted.get(key):
            self._complete(self.posted[key].pop(0), s)
        else:
            self.unexpected.setdefault(key, []).append(s)

    def _complete(self, recv: int, send: int) -> None:
        m = self.msgs[self.msg_of[send]]
        start = self.start[recv]
        assert start is not None
        self._push(DONE, recv, max(start, m["proc_end"]))
        if m.get("rendezvous"):
            self._push(DONE, send, max(m["cpu_end"], m["proc_start"], start))

    def run(self) -> dict[str, list[int | None]]:
        """Every op's ready, start and end time."""
        for i, pending in enumerate(self.pending):
            if pending == [0, 0]:
                self._push(READY, i, 0)
        while self.events:
            t, kind, ident = heapq.heappop(self.events)
            self.now = t
            if kind == DONE:
                self._on_done(ident)
            elif kind == ARRIVE:
                m = self.msg_of[ident]
                self.msgs[m]["arrive"] = t
                self._request(m | MSG)
            elif kind == WAKE:
                self._on_wake(ident)
            else:
                self._on_ready(ident)
        return {
            "op_ready_ns": self.ready,
            "op_start_ns": self.start,
            "op_end_ns": self.end,
        }


@pytest.fixture
def oracle() -> type[Oracle]:
    """The Oracle class."""
    return Oracle


def build_graph(
    ops: Sequence[Mapping[str, Any]],
    edges: Sequence[tuple[int, ...]] = (),
) -> dict[str, np.ndarray]:
    """The engine's arrays from op dicts and (src, dst[, kind]) edges."""
    columns = {
        "kind": (CALC, np.uint8),
        "rank": (0, np.uint32),
        "cpu": (0, np.uint16),
        "nic": (0, np.uint16),
        "duration": (0, np.int64),
        "bytes": (0, np.uint64),
        "peer": (-1, np.int32),
        "tag": (0, np.uint64),
        "coll_type": (-1, np.int8),
        "coll_group": (-1, np.int32),
        "coll_root": (0, np.int32),
    }
    out = {
        name: np.array([op.get(name, default) for op in ops], dtype=dtype)
        for name, (default, dtype) in columns.items()
    }
    out["edge_src"] = np.array([e[0] for e in edges], dtype=np.uint32)
    out["edge_dst"] = np.array([e[1] for e in edges], dtype=np.uint32)
    out["edge_kind"] = np.array(
        [e[2] if len(e) > 2 else 0 for e in edges], dtype=np.uint8
    )
    return out


@pytest.fixture
def graph_builder() -> Any:
    """``build_graph``."""
    return build_graph


@pytest.fixture(scope="session")
def training_traces(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, pathlib.Path]:
    """A small 3D-parallel job as GOAL text, GOAL binary and Chakra.

    Also a job with MoE all-to-alls in both formats, and an ingest cache.
    """
    root = tmp_path_factory.mktemp("training")
    small = training.TrainingSpec(
        dp=2,
        pp=2,
        tp=2,
        microbatches=2,
        layers_per_stage=2,
        steps=1,
        dp_buckets=2,
    )
    moe = training.TrainingSpec(
        dp=2,
        pp=1,
        tp=2,
        microbatches=2,
        layers_per_stage=2,
        steps=1,
        dp_buckets=1,
        moe_every=2,
        ep_size=2,
    )
    paths = {
        "goal": root / "train.goal",
        "goal_bin": root / "bin" / "train.bin",
        "chakra": root / "chakra",
        "moe_goal": root / "moe.goal",
        "moe_chakra": root / "moe_chakra",
        "cache": root / "cache",
    }
    paths["goal_bin"].parent.mkdir()
    training.write_goal(paths["goal"], small)
    training.write_goal_bin(paths["goal_bin"], small)
    training.write_chakra(paths["chakra"], small)
    training.write_goal(paths["moe_goal"], moe)
    training.write_chakra(paths["moe_chakra"], moe)
    return paths
