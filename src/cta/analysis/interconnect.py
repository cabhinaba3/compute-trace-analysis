"""An alpha-beta model of the interconnect, to price communication ops.

A link moves ``S`` bytes in ``alpha + S * beta`` (alpha is the latency of one
step in ns, beta the ns per byte, one over the bandwidth in GB/s). A
collective is a number of such steps and a multiple of ``S``, as in Thakur,
Rabenseifner and Gropp's models of MPI collectives:

========== =========================== ===============================
collective ring                        tree (binomial / doubling)
========== =========================== ===============================
allreduce  2(n-1) alpha + 2(n-1)/n S b 2 ceil(log2 n) (alpha + S b)
allgather  (n-1) alpha + (n-1)/n S b   ceil(log2 n) alpha + (n-1)/n S b
(and reduce-scatter)
alltoall   (n-1) alpha + (n-1)/n S b   same as ring
broadcast  (n-1) alpha + S b           ceil(log2 n) (alpha + S b)
(and reduce)
barrier    (n-1) alpha                 ceil(log2 n) alpha
========== =========================== ===============================

``algorithm="auto"`` takes the cheaper of the two for every op, and
``"flat"`` prices every op as one transfer, ``alpha + S * beta``: the single
nominal link the what-if solver has always used for ops with no recorded time.

Two links tell a node's GPUs apart from the network: ``intra`` between GPUs
of a node and ``inter`` between nodes, with ``gpus_per_node`` GPUs on each.
A transfer between ranks uses ``intra`` when both are on one node. A
collective of ``n`` ranks is taken to occupy the next ``ceil(n/gpus_per_node)``
nodes in blocks of ranks; on ``k > 1`` nodes the ring crosses between nodes
``k`` times per ``n`` steps (the tree ``ceil(log2 k)`` of its steps does) and
its bandwidth is that of the ``inter`` link. These are first-order models
that fit a clean, uncontended network; they do not model protocol changes
with message size, congestion or overlapping collectives.

A trace does not say how many ranks a collective had, so the group size is
set per collective type (``group.ALL_REDUCE=64``) and is the number of ranks
in the graph where it is not set.

The numbers of a link are not datasheet values: give them, or fit them from a
run's ``network_ops`` with ``fit_link``.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import math
import re

import numpy as np
import polars as pl

ALGORITHMS = ("flat", "ring", "tree", "auto")

# The what-if solver's original constants: 400GbE (50 GB/s), 500 ns latency
# and 150 ns of overhead at each end.
NOMINAL_ALPHA_NS = 500.0 + 2 * 150.0
NOMINAL_NS_PER_BYTE = 0.02

# Names of ``network_ops`` operations -> collective types of an execution graph.
_OP_TYPES = {
    "allreduce": "ALL_REDUCE",
    "allgather": "ALL_GATHER",
    "reducescatter": "REDUCE_SCATTER",
    "alltoall": "ALL_TO_ALL",
    "broadcast": "BROADCAST",
    "reduce": "REDUCE",
    "barrier": "BARRIER",
    "send": "SEND",
    "recv": "SEND",
    "sendrecv": "SEND",
}


@dataclasses.dataclass(slots=True, frozen=True)
class Link:
    """One kind of link.

    Attributes:
        alpha_ns: Latency of one step, in ns.
        ns_per_byte: Time per byte, in ns (one over the bandwidth in GB/s).
    """

    alpha_ns: float
    ns_per_byte: float

    @property
    def gb_per_s(self) -> float:
        """The bandwidth in GB/s (bytes per ns); infinite if beta is 0."""
        return math.inf if self.ns_per_byte == 0 else 1.0 / self.ns_per_byte


@dataclasses.dataclass(slots=True, frozen=True)
class Interconnect:
    """The links of a cluster and how collectives use them.

    Attributes:
        gpus_per_node: Ranks on each node, in rank order.
        intra: The link between ranks of one node.
        inter: The link between nodes.
        algorithm: ``flat``, ``ring``, ``tree`` or ``auto``.
        group_sizes: Ranks per collective type (upper case), ``"*"`` for all
            the others; a type that is not listed has every rank of the graph.
    """

    gpus_per_node: int = 8
    intra: Link = Link(NOMINAL_ALPHA_NS, NOMINAL_NS_PER_BYTE)
    inter: Link = Link(NOMINAL_ALPHA_NS, NOMINAL_NS_PER_BYTE)
    algorithm: str = "flat"
    group_sizes: Mapping[str, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Checks the values."""
        if self.gpus_per_node < 1:
            raise ValueError("gpus_per_node must be at least 1")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {ALGORITHMS}, got {self.algorithm!r}"
            )
        for link in (self.intra, self.inter):
            if link.alpha_ns < 0 or link.ns_per_byte < 0:
                raise ValueError(
                    "a link's latency and time per byte must not be negative"
                )
        for name, size in self.group_sizes.items():
            if size < 1:
                raise ValueError(f"group size of {name} must be at least 1")


# What the solver used before there was a model: one 400GbE link, every op
# priced as one transfer.
NOMINAL = Interconnect()

_TIME_UNITS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
# Bytes per ns, which is GB/s.
_BANDWIDTH_UNITS = {
    "gb/s": 1.0,
    "mb/s": 1e-3,
    "tb/s": 1e3,
    "gbps": 1.0 / 8.0,
    "mbps": 1e-3 / 8.0,
    "tbps": 1e3 / 8.0,
}
_NUMBER = r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"


def _parse_link(text: str) -> Link:
    """Parses ``<latency>:<bandwidth>``, such as ``1.5us:300GB/s``."""
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"A link is '<latency>:<bandwidth>' such as '2us:50GB/s', got "
            f"{text!r}"
        )
    latency = re.fullmatch(_NUMBER + r"\s*(ns|us|ms|s)", parts[0].strip(), re.I)
    bandwidth = re.fullmatch(_NUMBER + r"\s*([a-z/]+)", parts[1].strip(), re.I)
    if latency is None:
        raise ValueError(
            f"Cannot read the latency {parts[0]!r}: use ns, us, ms or s"
        )
    if bandwidth is None or bandwidth.group(2).lower() not in _BANDWIDTH_UNITS:
        raise ValueError(
            f"Cannot read the bandwidth {parts[1]!r}: use GB/s, MB/s, TB/s, "
            "Gbps, Mbps or Tbps"
        )
    per_ns = (
        float(bandwidth.group(1)) * _BANDWIDTH_UNITS[bandwidth.group(2).lower()]
    )
    if per_ns <= 0:
        raise ValueError("A bandwidth must be positive")
    return Link(
        float(latency.group(1)) * _TIME_UNITS[latency.group(2).lower()],
        1.0 / per_ns,
    )


def parse(text: str, base: Interconnect = NOMINAL) -> Interconnect:
    """Parses an interconnect from ``key=value`` pairs joined by commas.

    Keys: ``gpus_per_node=8``; ``intra=<latency>:<bandwidth>`` and
    ``inter=...`` (``1us:300GB/s``); ``algo=flat|ring|tree|auto``; and
    ``group.<TYPE>=<ranks>`` such as ``group.ALL_REDUCE=64`` (``group.*`` for
    every type not listed). A key that is left out keeps its value in ``base``.

    Args:
        text: The specification, such as
            ``"gpus_per_node=8,intra=1us:300GB/s,inter=5us:50GB/s,algo=ring"``.
        base: The interconnect that keys left out take their values from.

    Returns:
        The interconnect.

    Raises:
        ValueError: A key is unknown or a value cannot be read.
    """
    fields: dict[str, object] = {}
    groups = dict(base.group_sizes)
    for item in filter(None, (part.strip() for part in text.split(","))):
        key, sep, value = item.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if not sep or not value:
            raise ValueError(f"Expected key=value in {item!r}")
        if key == "gpus_per_node":
            fields["gpus_per_node"] = int(value)
        elif key in ("intra", "inter"):
            fields[key] = _parse_link(value)
        elif key in ("algo", "algorithm"):
            fields["algorithm"] = value.lower()
        elif key.startswith("group."):
            groups[key.removeprefix("group.").upper()] = int(value)
        else:
            raise ValueError(
                f"Unknown interconnect key {key!r}; use gpus_per_node, intra, "
                "inter, algo or group.<TYPE>"
            )
    return dataclasses.replace(base, group_sizes=groups, **fields)  # type: ignore[arg-type]


def _ceil_log2(n: np.ndarray) -> np.ndarray:
    """Returns ``ceil(log2(n))`` for ranks counts of at least 1."""
    return np.ceil(np.log2(np.maximum(n, 1))).astype(np.float64)


def _steps_and_volume(
    kind: np.ndarray,
    ranks: np.ndarray,
    nodes: np.ndarray,
    algorithm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts the steps and bytes-multiple of collectives under one algorithm.

    Args:
        kind: Collective type of each op (upper case); ``SEND`` is one
            transfer.
        ranks: Ranks in each op's group.
        nodes: Nodes each group occupies.
        algorithm: ``ring`` or ``tree``.

    Returns:
        For every op: the steps on intra-node links, the steps on inter-node
        links, and the multiple of the message size that crosses the
        bottleneck link.
    """
    n = ranks.astype(np.float64)
    k = nodes.astype(np.float64)
    steps = np.zeros(len(kind))
    volume = np.zeros(len(kind))
    inter = np.zeros(len(kind))
    ring = algorithm == "ring"
    two_ring = 2.0 * (n - 1.0)
    log_n = _ceil_log2(n)
    log_k = _ceil_log2(k)
    share = (n - 1.0) / np.maximum(n, 1.0)

    def pick(*names: str) -> np.ndarray:
        return np.isin(kind, names)

    allreduce = pick("ALL_REDUCE")
    gather = pick("ALL_GATHER", "REDUCE_SCATTER")
    alltoall = pick("ALL_TO_ALL")
    rooted = pick("BROADCAST", "REDUCE")
    barrier = pick("BARRIER")
    single = pick("SEND", "RECV")
    other = ~(allreduce | gather | alltoall | rooted | barrier | single)

    steps[single] = 1.0
    volume[single] = 1.0
    if ring:
        steps[allreduce] = two_ring[allreduce]
        volume[allreduce] = 2.0 * share[allreduce]
        for mask in (gather | other, alltoall):
            steps[mask] = (n - 1.0)[mask]
            volume[mask] = share[mask]
        steps[rooted | barrier] = (n - 1.0)[rooted | barrier]
        volume[rooted] = 1.0
        inter = np.where(k > 1, steps * k / np.maximum(n, 1.0), 0.0)
    else:
        steps[allreduce] = 2.0 * log_n[allreduce]
        volume[allreduce] = 2.0 * log_n[allreduce]
        steps[gather | other] = log_n[gather | other]
        volume[gather | other] = share[gather | other]
        steps[alltoall] = (n - 1.0)[alltoall]
        volume[alltoall] = share[alltoall]
        steps[rooted] = log_n[rooted]
        volume[rooted] = log_n[rooted]
        steps[barrier] = log_n[barrier]
        factor = np.where(allreduce, 2.0, 1.0)
        tree_inter = np.where(k > 1, factor * log_k, 0.0)
        ring_inter = np.where(k > 1, steps * k / np.maximum(n, 1.0), 0.0)
        inter = np.where(alltoall, ring_inter, tree_inter)
    inter = np.minimum(inter, steps)
    return steps - inter, inter, volume


def collective_cost_ns(
    kind: np.ndarray,
    size_bytes: np.ndarray,
    ranks: np.ndarray,
    interconnect: Interconnect,
) -> np.ndarray:
    """Prices collectives, each as if it ran alone on the interconnect.

    Args:
        kind: Collective type of each op, upper case (``ALL_REDUCE``, ...).
        size_bytes: Message size of each op, as the trace counts it.
        ranks: Ranks in each op's group.
        interconnect: The links and the algorithm.

    Returns:
        The time of each op in ns.
    """
    size = size_bytes.astype(np.float64)
    nodes = np.maximum(
        np.ceil(ranks / interconnect.gpus_per_node).astype(np.int64), 1
    )
    if interconnect.algorithm == "flat":
        spans = nodes > 1
        return np.where(
            spans, interconnect.inter.alpha_ns, interconnect.intra.alpha_ns
        ) + size * np.where(
            spans,
            interconnect.inter.ns_per_byte,
            interconnect.intra.ns_per_byte,
        )
    costs = []
    for algorithm in (
        ("ring", "tree")
        if interconnect.algorithm == "auto"
        else (interconnect.algorithm,)
    ):
        intra_steps, inter_steps, volume = _steps_and_volume(
            kind, ranks, nodes, algorithm
        )
        ns_per_byte = np.where(
            nodes > 1,
            interconnect.inter.ns_per_byte,
            interconnect.intra.ns_per_byte,
        )
        cost = (
            intra_steps * interconnect.intra.alpha_ns
            + inter_steps * interconnect.inter.alpha_ns
            + volume * size * ns_per_byte
        )
        costs.append(np.where(ranks > 1, cost, 0.0))
    return np.minimum.reduce(costs)


def transfer_cost_ns(
    size_bytes: np.ndarray,
    src_node: np.ndarray,
    dst_node: np.ndarray,
    interconnect: Interconnect,
) -> np.ndarray:
    """Prices point-to-point transfers on the link between their nodes.

    Args:
        size_bytes: Bytes of each transfer.
        src_node: Node of the sender; negative when unknown.
        dst_node: Node of the receiver; negative when unknown.
        interconnect: The links.

    Returns:
        The time of each transfer in ns (the whole transfer, not one end).
    """
    size = size_bytes.astype(np.float64)
    between_nodes = (src_node != dst_node) & (src_node >= 0) & (dst_node >= 0)
    alpha = np.where(
        between_nodes, interconnect.inter.alpha_ns, interconnect.intra.alpha_ns
    )
    beta = np.where(
        between_nodes,
        interconnect.inter.ns_per_byte,
        interconnect.intra.ns_per_byte,
    )
    return alpha + size * beta


def node_cost_ns(
    nodes: pl.DataFrame, interconnect: Interconnect
) -> tuple[np.ndarray, list[str]]:
    """Prices every communication op of an execution graph.

    A send or receive carries half of its transfer, as the solver has always
    split the cost between its two ends. Ops that are not communication cost
    0.

    Args:
        nodes: The node table (``rank``, ``op_class``, ``comm_type``,
            ``size_bytes``, ``src_rank``, ``dst_rank``).
        interconnect: The links and the algorithm.

    Returns:
        The cost of every op in ns, and the collective types that had no
        group size set and were priced with every rank of the graph.
    """
    op_class = nodes["op_class"].fill_null("").to_numpy()
    is_p2p = np.isin(op_class, ["comm_send", "comm_recv"])
    is_coll = ~is_p2p & np.char.startswith(op_class.astype(str), "comm_")
    cost = np.zeros(nodes.height)
    defaulted: list[str] = []
    size = nodes["size_bytes"].to_numpy().astype(np.int64)
    per_node = interconnect.gpus_per_node
    if is_p2p.any():
        src = nodes["src_rank"].cast(pl.Int64).to_numpy()[is_p2p]
        dst = nodes["dst_rank"].cast(pl.Int64).to_numpy()[is_p2p]
        cost[is_p2p] = 0.5 * transfer_cost_ns(
            size[is_p2p],
            np.where(src >= 0, src // per_node, -1),
            np.where(dst >= 0, dst // per_node, -1),
            interconnect,
        )
    if is_coll.any():
        kind = (nodes["comm_type"].fill_null("").str.to_uppercase().to_numpy())[
            is_coll
        ]
        world = int(nodes["rank"].n_unique())
        groups = interconnect.group_sizes
        ranks = np.full(len(kind), world, dtype=np.int64)
        for name in np.unique(kind):
            listed = groups.get(str(name), groups.get("*"))
            if listed is None:
                if world > 1:
                    defaulted.append(str(name))
            else:
                ranks[kind == name] = listed
        cost[is_coll] = collective_cost_ns(
            kind, size[is_coll], ranks, interconnect
        )
    return cost, sorted(defaulted)


@dataclasses.dataclass(slots=True, frozen=True)
class Fit:
    """A link fitted to measured communication times.

    Attributes:
        link: The fitted latency and time per byte.
        calls: Calls the fit used.
        median_relative_error: Median of ``|predicted - measured| / measured``
            over those calls.
        r_squared: Share of the variance of the measured times the model
            explains (1 is exact).
    """

    link: Link
    calls: int
    median_relative_error: float
    r_squared: float


def _nonnegative_two(
    features: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Solves a weighted two-variable least squares with both values >= 0."""
    scaled = features * weight[:, None]
    goal = target * weight
    best = None
    best_error = math.inf
    candidates = [np.linalg.lstsq(scaled, goal, rcond=None)[0]]
    for column in (0, 1):
        only = np.zeros(2)
        col = scaled[:, column]
        denominator = float(col @ col)
        if denominator > 0:
            only[column] = max(float(col @ goal) / denominator, 0.0)
        candidates.append(only)
    for candidate in candidates:
        if np.any(candidate < 0):
            continue
        error = float(np.sum((scaled @ candidate - goal) ** 2))
        if error < best_error:
            best, best_error = candidate, error
    return best if best is not None else np.zeros(2)


def fit_link(network_ops: pl.DataFrame, *, algorithm: str = "ring") -> Fit:
    """Fits one link's latency and bandwidth to a run's communication calls.

    Every call is ``steps * alpha + volume * bytes * beta`` for the steps and
    volume of its operation and group size under ``algorithm`` (the whole
    group on one link, so use the calls of a single node or of a single
    network). Calls with no size or fewer than 2 ranks are left out. The fit
    minimizes the relative error, so small calls count as much as large ones,
    and neither value is negative.

    Args:
        network_ops: A ``network_ops`` table (``op``, ``bytes``,
            ``group_size``, ``duration_ns``).
        algorithm: ``ring`` or ``tree``; what to assume the calls ran.

    Returns:
        The fit.

    Raises:
        ValueError: Fewer than two calls are usable, or ``algorithm`` is not
            ``ring`` or ``tree``.
    """
    if algorithm not in ("ring", "tree"):
        raise ValueError("Fit with algorithm 'ring' or 'tree'")
    ops = network_ops.filter(
        (pl.col("duration_ns") > 0)
        & (pl.col("bytes") > 0)
        & (pl.col("group_size") >= 2)
        & pl.col("op").is_in(list(_OP_TYPES))
    )
    if ops.height < 2:
        raise ValueError(
            "Need at least 2 calls with a size, a duration and 2+ ranks to "
            f"fit a link; found {ops.height}"
        )
    kind = ops["op"].replace_strict(_OP_TYPES, return_dtype=pl.String)
    ranks = ops["group_size"].to_numpy().astype(np.int64)
    intra, inter, volume = _steps_and_volume(
        kind.to_numpy(), ranks, np.ones(ops.height, dtype=np.int64), algorithm
    )
    size = ops["bytes"].to_numpy().astype(np.float64)
    measured = ops["duration_ns"].to_numpy().astype(np.float64)
    features = np.column_stack([intra + inter, volume * size])
    solution = _nonnegative_two(features, measured, 1.0 / measured)
    predicted = features @ solution
    relative = np.abs(predicted - measured) / measured
    total = float(np.sum((measured - measured.mean()) ** 2))
    residual = float(np.sum((measured - predicted) ** 2))
    return Fit(
        link=Link(float(solution[0]), float(solution[1])),
        calls=ops.height,
        median_relative_error=float(np.median(relative)),
        r_squared=1.0 - residual / total if total > 0 else 1.0,
    )
