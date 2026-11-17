"""Fixtures of the simulator tests, and a plain-Python twin of the engine.

``Oracle`` is a slow, plain-Python twin of the native engine, a test oracle.

It follows native/src/sim/engine.cpp step for step for the LogGOPS and
alpha-beta models (both rendezvous protocols, several CPUs and NICs per rank,
requires and irequires, intra-node parameters, NIC scale factors) and for
collectives costed with their recorded duration. It has no OS noise, no flow
model and no analytical collective costs; the tests of those use
hand-computed cases instead. Its only job is to agree with the engine to the
nanosecond on every op's ready, start and end time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import heapq
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from cta.synth import training

CALC, SEND, RECV, COLL, NOP = range(5)
REQUIRES, IREQUIRES = 0, 1
DONE, FLOW_DONE, ARRIVE, FLOW_START, RTS, CTS, WAKE, READY = range(8)
OP, MSG, DATA = range(3)
CPU, NIC_SEND, NIC_RECV = range(3)

_DEFAULTS: dict[str, Any] = {
    "model": "loggops",
    "world_size": 0,
    "ranks_per_node": 1,
    "L": 2500.0,
    "o": 1500.0,
    "g": 1000.0,
    "G": 6.0,
    "O": 0.0,
    "S": 65535,
    "rendezvous": "sync",
    "alpha": 5500.0,
    "beta": 6.0,
    "nic_scale": [],
}


def ns(x: float) -> int:
    """std::llround for non-negative values."""
    return math.floor(x + 0.5)


def minus_one(nbytes: int) -> float:
    return float(nbytes - 1) if nbytes > 0 else 0.0


class Oracle:
    """Runs a graph given as the engine's arrays and parameter dict."""

    def __init__(
        self,
        g: Mapping[str, np.ndarray],
        offsets: np.ndarray,
        members: np.ndarray,
        params: Mapping[str, Any],
    ) -> None:
        """Prepares a run of graph ``g`` with the engine's parameters."""
        self.g = {k: np.asarray(v).tolist() for k, v in g.items()}
        self.p = {**_DEFAULTS, **params}
        if self.p["model"] not in ("loggops", "alpha_beta"):
            raise ValueError("the oracle runs loggops and alpha_beta only")
        self.offsets = [int(x) for x in offsets]
        self.members = [int(x) for x in members]
        g_ = self.g
        n = self.n = len(g_["kind"])
        ranks = int(self.p["world_size"])
        for i in range(n):
            ranks = max(ranks, g_["rank"][i] + 1)
            if g_["kind"][i] in (SEND, RECV):
                ranks = max(ranks, g_["peer"][i] + 1)
        for m in self.members:
            ranks = max(ranks, m + 1)
        self.ranks = max(ranks, 1)
        cpus = [1] * self.ranks
        nics = [1] * self.ranks
        for i in range(n):
            r = g_["rank"][i]
            cpus[r] = max(cpus[r], g_["cpu"][i] + 1)
            nics[r] = max(nics[r], g_["nic"][i] + 1)
        self.cpus, self.nics = cpus, nics
        self.cpu_base = [0]
        self.nic_base = [0]
        for r in range(self.ranks):
            self.cpu_base.append(self.cpu_base[-1] + cpus[r])
            self.nic_base.append(self.nic_base[-1] + nics[r])
        self.total_cpus = self.cpu_base[-1]
        self.total_nics = self.nic_base[-1]
        count = self.total_cpus + 2 * self.total_nics
        self.free_at = [0] * count
        self.waiters: list[list[tuple[int, int]]] = [[] for _ in range(count)]
        self.wake = [False] * count

        self.succ: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        self.pend_req = [0] * n
        self.pend_ireq = [0] * n
        for s, d, k in zip(
            g_["edge_src"], g_["edge_dst"], g_["edge_kind"], strict=True
        ):
            self.succ[s].append((d, k))
            if k == IREQUIRES:
                self.pend_ireq[d] += 1
            else:
                self.pend_req[d] += 1

        self.msg_of = [-1] * n
        self.msgs: list[dict[str, Any]] = []
        for i in range(n):
            if g_["kind"][i] == SEND:
                self.msg_of[i] = len(self.msgs)
                self.msgs.append({"op": i, "rendezvous": False, "rts": False})

        # Collective instances.
        next_seq: dict[tuple[int, int], int] = {}
        most: dict[int, int] = {}
        seq = [-1] * n
        for i in range(n):
            if g_["kind"][i] != COLL:
                continue
            key = (g_["rank"][i], g_["coll_group"][i])
            seq[i] = next_seq.get(key, 0)
            next_seq[key] = seq[i] + 1
            grp = g_["coll_group"][i]
            most[grp] = max(most.get(grp, 0), seq[i] + 1)
        groups = len(self.offsets) - 1 if self.offsets else 0
        base = [0]
        for grp in range(groups):
            base.append(base[-1] + most.get(grp, 0))
        self.inst = {
            i: base[g_["coll_group"][i]] + seq[i]
            for i in range(n)
            if seq[i] >= 0
        }
        self.inst_ops: dict[int, list[int]] = {}
        for i in range(n):
            if i in self.inst:
                self.inst_ops.setdefault(self.inst[i], []).append(i)
        self.arrived: dict[int, int] = {}

        self.ready: list[int | None] = [None] * n
        self.start: list[int | None] = [None] * n
        self.end: list[int | None] = [None] * n
        self.matched = [-1] * n
        self.posted: dict[tuple[int, int, int], list[int]] = {}
        self.unexpected: dict[tuple[int, int, int], list[int]] = {}
        self.events: list[tuple[int, int, int]] = []
        self.now = 0

    # --- Parameters.

    def _same_node(self, a: int, b: int) -> bool:
        k = self.p["ranks_per_node"]
        return k > 1 and a // k == b // k

    def _lg(self, a: int, b: int) -> dict[str, float]:
        p = self.p
        out = {name: float(p[name]) for name in ("L", "o", "g", "G", "O")}
        if self._same_node(a, b) and any(f"intra_{k}" in p for k in out):
            out = {k: float(p.get(f"intra_{k}", v)) for k, v in out.items()}
        return out

    def _scale(self, a: int, b: int) -> float:
        s = self.p["nic_scale"]
        return min(s[a] if a < len(s) else 1.0, s[b] if b < len(s) else 1.0)

    def _big_g(self, a: int, b: int) -> float:
        return self._lg(a, b)["G"] / self._scale(a, b)

    def _alpha(self, a: int, b: int) -> float:
        p = self.p
        has = "intra_alpha" in p or "intra_beta" in p
        if has and self._same_node(a, b):
            return float(p.get("intra_alpha", p["alpha"]))
        return float(p["alpha"])

    def _beta(self, a: int, b: int) -> float:
        p = self.p
        has = "intra_alpha" in p or "intra_beta" in p
        value = (
            float(p.get("intra_beta", p["beta"]))
            if has and self._same_node(a, b)
            else float(p["beta"])
        )
        return value / self._scale(a, b)

    def _loggops(self) -> bool:
        return self.p["model"] == "loggops"

    def _uses_rts(self, op: int) -> bool:
        return (
            self.p["model"] != "alpha_beta"
            and self.p["rendezvous"] == "rts_cts"
            and self.g["bytes"][op] > self.p["S"]
        )

    # --- Resources.

    def _cpu_id(self, rank: int, cpu: int) -> int:
        return self.cpu_base[rank] + min(cpu, self.cpus[rank] - 1)

    def _nic_send_id(self, rank: int, nic: int) -> int:
        return (
            self.total_cpus
            + self.nic_base[rank]
            + min(nic, self.nics[rank] - 1)
        )

    def _nic_recv_id(self, rank: int, nic: int) -> int:
        return (
            self.total_cpus
            + self.total_nics
            + self.nic_base[rank]
            + min(nic, self.nics[rank] - 1)
        )

    def _required(self, act: tuple[int, int]) -> list[int]:
        kind, i = act
        g = self.g
        if kind == OP:
            if g["kind"][i] == CALC:
                return [self._cpu_id(g["rank"][i], g["cpu"][i])]
            if self.p["model"] == "alpha_beta":
                return []
            if not self._loggops() or self._uses_rts(i):
                return [self._cpu_id(g["rank"][i], g["cpu"][i])]
            return [
                self._cpu_id(g["rank"][i], g["cpu"][i]),
                self._nic_send_id(g["rank"][i], g["nic"][i]),
            ]
        s = self.msgs[i]["op"]
        if kind == MSG:
            if self.p["model"] == "alpha_beta":
                return []
            dst = g["peer"][s]
            return [
                self._cpu_id(dst, g["cpu"][s]),
                self._nic_recv_id(dst, g["nic"][s]),
            ]
        return [
            self._cpu_id(g["rank"][s], g["cpu"][s]),
            self._nic_send_id(g["rank"][s], g["nic"][s]),
        ]

    def _since(self, act: tuple[int, int]) -> int:
        kind, i = act
        if kind == OP:
            value = self.ready[i]
        elif kind == MSG:
            value = self.msgs[i]["arrive"]
        else:
            value = self.msgs[i]["cts_arrive"]
        assert value is not None
        return value

    @staticmethod
    def _encode(act: tuple[int, int]) -> int:
        return (act[0] << 30) | act[1]

    @staticmethod
    def _decode(code: int) -> tuple[int, int]:
        return code >> 30, code & ((1 << 30) - 1)

    def _push(self, kind: int, ident: int, t: int) -> None:
        heapq.heappush(self.events, (t, kind, ident))

    def _request(self, act: tuple[int, int]) -> None:
        res = self._required(act)
        block = -1
        latest = self.now
        for k, r in enumerate(res):
            if self.free_at[r] > latest:
                latest = self.free_at[r]
                block = k
        if block < 0:
            for k, r in enumerate(res):
                if self.waiters[r]:
                    block = k
                    break
        if block < 0:
            self._begin(act)
        else:
            self._enqueue(act, self._since(act), res[block])

    def _enqueue(self, act: tuple[int, int], since: int, r: int) -> None:
        heapq.heappush(self.waiters[r], (since, self._encode(act)))
        if not self.wake[r]:
            self.wake[r] = True
            self._push(WAKE, r, max(self.free_at[r], self.now))

    def _on_wake(self, r: int) -> None:
        self.wake[r] = False
        while self.waiters[r] and self.free_at[r] <= self.now:
            since, code = heapq.heappop(self.waiters[r])
            act = self._decode(code)
            other = -1
            for o in self._required(act):
                if o == r:
                    continue
                if self.free_at[o] > self.now or (
                    self.waiters[o] and self.waiters[o][0] < (since, code)
                ):
                    other = o
                    break
            if other < 0:
                self._begin(act)
            else:
                self._enqueue(act, since, other)
        if self.waiters[r] and not self.wake[r]:
            self.wake[r] = True
            self._push(WAKE, r, max(self.free_at[r], self.now))

    def _begin(self, act: tuple[int, int]) -> None:
        kind, i = act
        if kind == OP:
            if self.g["kind"][i] == CALC:
                self._begin_calc(i)
            else:
                self._begin_send(i)
        elif kind == MSG:
            self._begin_proc(i)
        else:
            self._begin_data(i)

    # --- Ops.

    def _start_op(self, op: int) -> None:
        self.start[op] = self.now
        for t, k in self.succ[op]:
            if k != IREQUIRES:
                continue
            self.pend_ireq[t] -= 1
            if self.pend_ireq[t] == 0 and self.pend_req[t] == 0:
                self._push(READY, t, self.now)

    def _on_done(self, op: int) -> None:
        self.end[op] = self.now
        for t, k in self.succ[op]:
            if k != REQUIRES:
                continue
            self.pend_req[t] -= 1
            if self.pend_req[t] == 0 and self.pend_ireq[t] == 0:
                self._push(READY, t, self.now)

    def _on_ready(self, op: int) -> None:
        self.ready[op] = self.now
        kind = self.g["kind"][op]
        if kind == NOP:
            self._start_op(op)
            self._push(DONE, op, self.now)
        elif kind in (CALC, SEND):
            self._request((OP, op))
        elif kind == RECV:
            self._start_op(op)
            self._post_recv(op)
        else:
            self._coll_arrive(op)

    def _begin_calc(self, op: int) -> None:
        self._start_op(op)
        end = self.now + self.g["duration"][op]
        self.free_at[self._cpu_id(self.g["rank"][op], self.g["cpu"][op])] = end
        self._push(DONE, op, end)

    def _begin_send(self, op: int) -> None:
        self._start_op(op)
        g = self.g
        m = self.msgs[self.msg_of[op]]
        src, dst, b = g["rank"][op], g["peer"][op], g["bytes"][op]
        if self.p["model"] == "alpha_beta":
            m["cpu_end"] = self.now
            self._push(
                ARRIVE,
                op,
                self.now + ns(self._alpha(src, dst) + b * self._beta(src, dst)),
            )
            self._push(DONE, op, self.now)
            return
        lg = self._lg(src, dst)
        if self._uses_rts(op):
            m["rts"] = True
            o = ns(lg["o"])
            self.free_at[self._cpu_id(src, g["cpu"][op])] = self.now + o
            self._push(RTS, op, self.now + o + ns(lg["L"]))
            return
        m["rendezvous"] = b > self.p["S"]
        self._send_data(op, m)
        if not m["rendezvous"]:
            self._push(DONE, op, m["cpu_end"])

    def _send_data(self, op: int, m: dict[str, Any]) -> None:
        g = self.g
        src, dst, b = g["rank"][op], g["peer"][op], g["bytes"][op]
        lg = self._lg(src, dst)
        busy = ns(lg["o"] + minus_one(b) * lg["O"])
        m["cpu_end"] = self.now + busy
        self.free_at[self._cpu_id(src, g["cpu"][op])] = m["cpu_end"]
        gap = ns(lg["g"] + minus_one(b) * self._big_g(src, dst))
        self.free_at[self._nic_send_id(src, g["nic"][op])] = self.now + gap
        self._push(ARRIVE, op, self.now + ns(lg["o"]) + ns(lg["L"]))

    def _begin_proc(self, i: int) -> None:
        m = self.msgs[i]
        g = self.g
        s = m["op"]
        src, dst, b = g["rank"][s], g["peer"][s], g["bytes"][s]
        m["proc_start"] = m["proc_end"] = self.now
        if self.p["model"] != "alpha_beta":
            lg = self._lg(src, dst)
            b1 = minus_one(b)
            big_g = self._big_g(src, dst)
            busy = ns(lg["o"] + max(b1 * lg["O"], b1 * big_g))
            self.free_at[self._nic_recv_id(dst, g["nic"][s])] = self.now + ns(
                lg["g"] + b1 * big_g
            )
            m["proc_end"] = self.now + busy
            self.free_at[self._cpu_id(dst, g["cpu"][s])] = m["proc_end"]
        self._deliver(i)

    def _deliver(self, i: int) -> None:
        g = self.g
        s = self.msgs[i]["op"]
        if self.matched[s] >= 0:
            self._complete_recv(self.matched[s], s)
            return
        key = (g["peer"][s], g["rank"][s], g["tag"][s])
        queue = self.posted.get(key)
        if queue:
            recv = queue.pop(0)
            self.matched[recv], self.matched[s] = s, recv
            self._complete_recv(recv, s)
        else:
            self.unexpected.setdefault(key, []).append(s)

    def _complete_recv(self, recv: int, send: int) -> None:
        m = self.msgs[self.msg_of[send]]
        start = self.start[recv]
        assert start is not None
        self._push(DONE, recv, max(start, m["proc_end"]))
        if m["rendezvous"]:
            self._push(
                DONE, send, max(m["cpu_end"], max(m["proc_start"], start))
            )

    def _post_recv(self, op: int) -> None:
        g = self.g
        key = (g["rank"][op], g["peer"][op], g["tag"][op])
        queue = self.unexpected.get(key)
        if queue:
            send = queue.pop(0)
            self.matched[op], self.matched[send] = send, op
            if self.msgs[self.msg_of[send]]["rts"]:
                self._rts_matched(send)
            else:
                self._complete_recv(op, send)
        else:
            self.posted.setdefault(key, []).append(op)

    def _on_rts(self, op: int) -> None:
        g = self.g
        key = (g["peer"][op], g["rank"][op], g["tag"][op])
        queue = self.posted.get(key)
        if queue:
            recv = queue.pop(0)
            self.matched[recv], self.matched[op] = op, recv
            self._rts_matched(op)
        else:
            self.unexpected.setdefault(key, []).append(op)

    def _rts_matched(self, send: int) -> None:
        g = self.g
        m = self.msgs[self.msg_of[send]]
        src, dst = g["rank"][send], g["peer"][send]
        lg = self._lg(dst, src)
        m["cts_arrive"] = self.now + ns(lg["o"]) + ns(lg["L"])
        self._push(CTS, send, m["cts_arrive"])

    def _begin_data(self, i: int) -> None:
        m = self.msgs[i]
        self._send_data(m["op"], m)
        self._push(DONE, m["op"], m["cpu_end"])

    def _coll_arrive(self, op: int) -> None:
        inst = self.inst[op]
        self.arrived[inst] = self.arrived.get(inst, 0) + 1
        grp = self.g["coll_group"][op]
        size = self.offsets[grp + 1] - self.offsets[grp]
        if self.arrived[inst] == size:
            ops = self.inst_ops[inst]
            shortest = min(self.g["duration"][x] for x in ops)
            for x in ops:
                self._start_op(x)
                self._push(DONE, x, self.now + shortest)

    def run(self) -> dict[str, list[int | None]]:
        """Runs the graph; returns every op's ready, start and end time."""
        for i in range(self.n):
            if self.pend_req[i] == 0 and self.pend_ireq[i] == 0:
                self._push(READY, i, 0)
        while self.events:
            t, kind, ident = heapq.heappop(self.events)
            self.now = t
            if kind == DONE:
                self._on_done(ident)
            elif kind == ARRIVE:
                m = self.msg_of[ident]
                self.msgs[m]["arrive"] = t
                self._request((MSG, m))
            elif kind == RTS:
                self._on_rts(ident)
            elif kind == CTS:
                self._request((DATA, self.msg_of[ident]))
            elif kind == WAKE:
                self._on_wake(ident)
            elif kind == READY:
                self._on_ready(ident)
            else:
                raise AssertionError(f"unexpected event {kind}")
        return {
            "ready_ns": self.ready,
            "start_ns": self.start,
            "end_ns": self.end,
        }


@pytest.fixture
def oracle() -> type[Oracle]:
    """The Oracle class."""
    return Oracle


GraphBuilder = Callable[..., dict[str, np.ndarray]]


def build_graph(
    ops: Sequence[Mapping[str, Any]],
    edges: Sequence[tuple[int, int] | tuple[int, int, int]] = (),
) -> dict[str, np.ndarray]:
    """Builds the engine's arrays from op dicts and (src, dst[, kind]) edges.

    An op dict has ``kind`` (CALC, SEND, ...), ``rank`` and, as needed,
    ``duration``, ``bytes``, ``peer``, ``tag``, ``cpu``, ``nic``,
    ``coll_type``, ``coll_group``.
    """

    def column(name: str, default: int, dtype: Any) -> np.ndarray:
        return np.array([op.get(name, default) for op in ops], dtype=dtype)

    return {
        "kind": column("kind", CALC, np.uint8),
        "rank": column("rank", 0, np.uint32),
        "cpu": column("cpu", 0, np.uint16),
        "nic": column("nic", 0, np.uint16),
        "duration": column("duration", 0, np.int64),
        "bytes": column("bytes", 0, np.uint64),
        "peer": column("peer", -1, np.int32),
        "tag": column("tag", 0, np.uint64),
        "coll_type": column("coll_type", -1, np.int8),
        "coll_group": column("coll_group", -1, np.int32),
        "coll_root": column("coll_root", 0, np.int32),
        "edge_src": np.array([e[0] for e in edges], dtype=np.uint32),
        "edge_dst": np.array([e[1] for e in edges], dtype=np.uint32),
        "edge_kind": np.array(
            [e[2] if len(e) > 2 else REQUIRES for e in edges], dtype=np.uint8
        ),
    }


@pytest.fixture
def graph_builder() -> GraphBuilder:
    """``build_graph``."""
    return build_graph


# A small 3D-parallel job: 8 ranks, tensor- and data-parallel all-reduces and
# pipeline sends, so every kind of traffic shows up.
SMALL_TRAINING = training.TrainingSpec(
    dp=2, pp=2, tp=2, microbatches=2, layers_per_stage=2, steps=1, dp_buckets=2
)


@pytest.fixture(scope="session")
def training_traces(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, pathlib.Path]:
    """The same small job as GOAL text, GOAL binary and Chakra.

    Also a job with MoE all-to-alls in both formats, and a shared ingest
    cache directory.
    """
    root = tmp_path_factory.mktemp("training")
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
    training.write_goal(paths["goal"], SMALL_TRAINING)
    training.write_goal_bin(paths["goal_bin"], SMALL_TRAINING)
    training.write_chakra(paths["chakra"], SMALL_TRAINING)
    training.write_goal(paths["moe_goal"], moe)
    training.write_chakra(paths["moe_chakra"], moe)
    return paths
