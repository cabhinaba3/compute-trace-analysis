"""What-if analysis of an execution graph.

Given a recorded run, answers "if these ops were X times faster, how much
shorter would the whole run be?" by solving the critical path again with
changed op times. The graph and its dependencies stay as recorded, so the
answer is first-order: it does not model contention, so a faster collective
that used to leave the GPU idle may in practice slow the compute it overlaps.

A scenario is a list of rules ``<selector>:<factor>``:

* Selectors: ``all``, ``comm``, ``compute``, ``class=comm_coll,compute``,
  ``family=GEMM`` (the op family, see ``op_label``), ``type=ALL_REDUCE``
  (the communication type), ``name~regex`` (case-insensitive) and
  ``rank=7``, ``rank=0-3,9`` or ``rank=*``. Join selectors with ``&``.
* Factors: ``2x`` (twice as fast), ``0.5x`` (twice as slow), ``-30%`` or
  ``+20%`` (that much less or more time), ``remove`` (takes no time) and
  ``median`` (only with a ``rank=`` selector: each selected rank's compute
  takes as long as the median rank's, if it took longer).

The solver is native (``cta._native.DagPlan``): the graph is prepared once and
every scenario then costs one pass over it, with several scenarios sharing the
walk and blocks of them running on separate threads.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import math
import pathlib
import re
import tempfile
import time
import typing
import warnings

import numpy as np
import polars as pl

from cta import _native as native
from cta.analysis import alignment
from cta.analysis import dag
from cta.analysis import interconnect

DEFAULT_FACTORS = (1.1, 1.25, 1.5, 2.0, 4.0, math.inf)

_CHUNK_BITS = 64
_OTHER = "(other)"
_OP_CLASS_ARRAY = np.array(
    [
        "other",
        "memory",
        "compute",
        "comm_send",
        "comm_recv",
        "comm_coll",
        "storage",
        "data_op",
    ],
    dtype=object,
)


@dataclasses.dataclass(slots=True, frozen=True)
class Rule:
    """One change of op times.

    Attributes:
        selector: Which ops it applies to (see the module docstring).
        factor: How their time changes (see the module docstring).
    """

    selector: str
    factor: str


@dataclasses.dataclass(slots=True, frozen=True)
class Scenario:
    """A named set of rules applied together.

    Attributes:
        label: Name of the scenario in result tables.
        rules: The rules; an op matched by several gets all their factors.
    """

    label: str
    rules: tuple[Rule, ...]


def parse_rule(text: str) -> Rule:
    """Parses ``<selector>:<factor>``.

    Args:
        text: The rule, such as ``"comm:2x"`` or ``"name~attn:-30%"``.

    Returns:
        The rule, with its selector and factor checked.

    Raises:
        ValueError: The text has no factor, or the selector or the factor is
            malformed.
    """
    selector, separator, factor = text.rpartition(":")
    if not separator or not selector.strip() or not factor.strip():
        raise ValueError(
            f"Invalid what-if rule '{text}'. Expected '<selector>:<factor>', "
            "e.g. 'comm:2x'"
        )
    rule = Rule(selector.strip(), factor.strip())
    _parse_atoms(rule.selector)
    parse_factor(rule.factor)
    return rule


def parse_scenario(text: str, label: str | None = None) -> Scenario:
    """Parses rules joined by ``;`` into one scenario.

    Args:
        text: For example ``"comm:2x;name~attention:1.3x"``.
        label: Name of the scenario; defaults to the text.

    Returns:
        The scenario.

    Raises:
        ValueError: A rule is malformed or there is none.
    """
    rules = tuple(parse_rule(part) for part in text.split(";") if part.strip())
    if not rules:
        raise ValueError("A what-if scenario needs at least one rule")
    return Scenario(label or text.strip(), rules)


def parse_factor(text: str) -> float | None:
    """Turns a factor into the multiplier of an op's time.

    Args:
        text: ``2x``, ``0.5x``, ``-30%``, ``+20%``, ``remove`` or
            ``median``.

    Returns:
        The multiplier (0 for ``remove``), or None for ``median``, which
        depends on the graph.

    Raises:
        ValueError: The text is none of these, or the speedup is not
            positive, or the change makes the time negative.
    """
    lowered = text.strip().lower()
    if lowered == "median":
        return None
    if lowered in ("remove", "inf", "free"):
        return 0.0
    try:
        if lowered.endswith("x"):
            speedup = float(lowered[:-1])
            if not speedup > 0.0:
                raise ValueError("a speedup must be positive")
            return 1.0 / speedup
        if lowered.endswith("%"):
            multiplier = 1.0 + float(lowered[:-1]) / 100.0
            if multiplier < 0.0:
                raise ValueError("an op cannot take negative time")
            return multiplier
    except ValueError as exc:
        raise ValueError(f"Invalid what-if factor '{text}': {exc}") from None
    raise ValueError(
        f"Invalid what-if factor '{text}'. Use e.g. '2x', '-30%', 'remove' "
        "or 'median'"
    )


def parse_factors(text: str) -> tuple[float, ...]:
    """Parses sweep speedups such as ``"1.25,2,inf"``.

    Args:
        text: Comma-separated speedups; ``inf`` removes the op.

    Returns:
        The speedups in the order given.

    Raises:
        ValueError: An entry is not a positive number or ``inf``.
    """
    factors: list[float] = []
    for part in text.split(","):
        part = part.strip().lower().rstrip("x")
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            raise ValueError(f"Invalid sweep factor '{part}'") from None
        if not value > 0.0:
            raise ValueError(f"A sweep factor must be positive, got '{part}'")
        factors.append(value)
    if not factors:
        raise ValueError("No sweep factors given")
    return tuple(factors)


def _parse_atoms(selector: str) -> tuple[tuple[str, str], ...]:
    """Splits a selector into (kind, argument) atoms."""
    atoms: list[tuple[str, str]] = []
    for raw_atom in selector.split("&"):
        atom = raw_atom.strip()
        if atom in ("all", "*"):
            atoms.append(("all", ""))
        elif atom in ("comm", "compute"):
            atoms.append((atom, ""))
        elif "~" in atom and atom.split("~", 1)[0].strip() == "name":
            pattern = atom.split("~", 1)[1]
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex in selector '{atom}': {exc}"
                ) from None
            atoms.append(("name", pattern))
        elif "=" in atom and atom.split("=", 1)[0].strip() in (
            "class",
            "family",
            "type",
            "rank",
        ):
            kind, argument = (part.strip() for part in atom.split("=", 1))
            if not argument:
                raise ValueError(f"Selector '{atom}' has no value")
            if kind == "rank":
                _parse_ranks(argument)
            atoms.append((kind, argument))
        else:
            raise ValueError(
                f"Invalid what-if selector '{atom}'. Use all, comm, compute, "
                "class=, family=, type=, name~ or rank="
            )
    return tuple(atoms)


def _parse_ranks(argument: str) -> list[tuple[int, int]] | None:
    """Parses ``3,5-7`` into inclusive ranges; None for ``*``."""
    if argument.strip() == "*":
        return None
    ranges: list[tuple[int, int]] = []
    for part in argument.split(","):
        low, dash, high = part.strip().partition("-")
        try:
            ranges.append((int(low), int(high if dash else low)))
        except ValueError:
            raise ValueError(f"Invalid rank list '{argument}'") from None
    return ranges


def op_label(frame: pl.DataFrame) -> pl.Series:
    """Names the family of every op of a node table.

    The label is the op family of the name (``GEMM``, ``NCCL_AllReduce``,
    ...); an op that matches no family takes its communication type (for
    example ``SEND``) or else its ``op_class``.

    Args:
        frame: A node table with ``name``, ``op_class`` and ``comm_type``.

    Returns:
        One label per row.
    """
    keys = frame.select(
        pl.col("name").fill_null(""),
        pl.col("op_class").fill_null(""),
        pl.col("comm_type").fill_null(""),
    )
    unique = keys.unique()
    family = alignment.op_family_expr("name")
    labelled = unique.with_columns(
        pl.when(family != "Other")
        .then(family)
        .when(pl.col("comm_type") != "")
        .then(pl.col("comm_type"))
        .otherwise(pl.col("op_class"))
        .alias("label")
    )
    return keys.join(
        labelled, on=["name", "op_class", "comm_type"], how="left"
    )["label"]


@dataclasses.dataclass(slots=True)
class WhatIfModel:
    """A graph prepared for repeated what-if solving.

    Attributes:
        nodes: The node table the graph was built from.
        graph: The graph's arrays.
        plan: The native plan.
        prepare_seconds: Time to build the arrays and the plan.
    """

    nodes: pl.DataFrame
    graph: dag.DagGraph
    plan: native.DagPlan
    prepare_seconds: float
    _labels: pl.Series | None = None

    @property
    def labels(self) -> pl.Series:
        """The op label of every vertex, computed once."""
        if self._labels is None:
            self._labels = op_label(self.nodes)
        return self._labels

    @property
    def num_nodes(self) -> int:
        """Number of vertices in the graph."""
        return self.graph.num_nodes

    @property
    def num_edges(self) -> int:
        """Number of deduplicated edges in the graph."""
        return self.plan.num_edges


@dataclasses.dataclass(slots=True, frozen=True)
class StreamingWhatIfModel:
    """A disk-backed `.cta_graph` model for bounded-RAM what-if solving.

    Holds no full-graph vertex table or DP arrays in memory; scenarios, sweeps,
    and interconnect repricing stream one topological block at a time via
    Engine D (`cta._native.solve_whatif_graph_cache`).

    Attributes:
        cache_path: Path to the built `.cta_graph` binary cache.
        reprice_steps: Chained `(target, base_or_none)` interconnect repricings
            applied lazily per block during solve and sweep passes.
        prepare_seconds: Time spent building or opening the `.cta_graph` cache.
        scratch_dir: Optional directory for temporary winner-log and spill I/O.
        num_nodes: Real vertex count in the graph.
        num_edges: Deduplicated edge count in the graph.
        world_size: Number of ranks recorded in the graph.
    """

    cache_path: pathlib.Path
    reprice_steps: tuple[
        tuple[interconnect.Interconnect, interconnect.Interconnect | None], ...
    ] = ()
    prepare_seconds: float = 0.0
    scratch_dir: str = ""
    num_nodes: int = 0
    num_edges: int = 0
    world_size: int = 0


def _ic_to_native(net: interconnect.Interconnect) -> dict[str, object]:
    """Serializes an Interconnect dataclass into a native Engine D dict."""
    return {
        "gpus_per_node": int(net.gpus_per_node),
        "intra_alpha_ns": float(net.intra.alpha_ns),
        "intra_ns_per_byte": float(net.intra.ns_per_byte),
        "inter_alpha_ns": float(net.inter.alpha_ns),
        "inter_ns_per_byte": float(net.inter.ns_per_byte),
        "algorithm": str(net.algorithm),
        "group_sizes": {k: int(v) for k, v in net.group_sizes.items()},
    }


def _steps_to_native(
    steps: Sequence[
        tuple[interconnect.Interconnect, interconnect.Interconnect | None]
    ],
) -> list[tuple[dict[str, object], dict[str, object] | None]]:
    """Serializes a reprice chain for native Engine D calls."""
    return [
        (
            _ic_to_native(target),
            _ic_to_native(base) if base is not None else None,
        )
        for target, base in steps
    ]


def build_streaming_model_from_artifact(
    artifact: object,
    *,
    cache_path: pathlib.Path,
    goal_semantics: str = "cta_legacy",
    collective_sync: str = "none",
    memory_budget_bytes: int = 0,
    scratch_dir: str = "",
) -> StreamingWhatIfModel:
    """Builds or reuses a `.cta_graph` cache for streaming what-if analysis.

    Args:
        artifact: A `TraceArtifact` for a Chakra or GOAL execution graph.
        cache_path: Destination path of the persistent `.cta_graph` file.
        goal_semantics: `"cta_legacy"` or `"strict"`.
        collective_sync: `"none"` or `"barrier"`.
        memory_budget_bytes: RAM cap for cache construction (0 = auto).
        scratch_dir: Optional scratch directory for out-of-core passes.

    Returns:
        A `StreamingWhatIfModel` backed by `cache_path`.
    """
    from cta import models
    from cta.analysis import streaming

    if not isinstance(artifact, models.TraceArtifact):
        raise TypeError(f"Expected TraceArtifact, got {type(artifact)!r}")
    start = time.perf_counter()
    streaming.ensure_graph_cache(
        artifact,
        cache_path,
        goal_semantics=goal_semantics,
        collective_sync=collective_sync,
        memory_budget_bytes=memory_budget_bytes,
        scratch_dir=scratch_dir,
    )
    cache_handle = native.GraphCache(str(cache_path))
    return StreamingWhatIfModel(
        cache_path=cache_path,
        prepare_seconds=time.perf_counter() - start,
        scratch_dir=scratch_dir,
        num_nodes=int(cache_handle.num_vertices),
        num_edges=int(cache_handle.num_edges),
        world_size=int(cache_handle.world_size),
    )


def load_streaming_model(
    source: str | pathlib.Path,
    *,
    cache_dir: str | pathlib.Path = ".cta_cache",
    goal_semantics: str = "cta_legacy",
    collective_sync: str = "none",
    memory_budget_bytes: int = 0,
    scratch_dir: str = "",
) -> StreamingWhatIfModel:
    """Opens a trace directly into a disk-backed `.cta_graph` what-if model.

    Bypasses full-table Parquet materialization and in-memory `DagPlan`
    allocation, allowing multi-million-node traces to run what-if scenarios,
    sweeps, and interconnect repricing within a bounded RAM budget.

    Args:
        source: A Chakra `.et` trace or directory, or a GOAL schedule.
        cache_dir: Directory storing persistent `.cta_graph` cache files.
        goal_semantics: `"cta_legacy"` or `"strict"`.
        collective_sync: `"none"` or `"barrier"`.
        memory_budget_bytes: RAM cap for cache build (0 = auto).
        scratch_dir: Optional scratch directory for out-of-core passes.

    Returns:
        The streaming what-if model of the first trial.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The source holds no execution graph.
    """
    from cta.analysis import streaming
    from cta.ingest import discovery

    group = discovery.resolve_trial_group(source)
    trial = group.trials[0]
    artifact = streaming.pick_graph_artifact(trial)
    if artifact is None:
        raise ValueError(
            f"'{source}' holds no execution graph (Chakra or GOAL)"
        )
    hasher = hashlib.sha256()
    for path in sorted(artifact.paths):
        stat = path.stat()
        hasher.update(
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode()
        )
    hasher.update(f"{goal_semantics}:{collective_sync}".encode())
    digest = hasher.hexdigest()[:16]
    cache_path = pathlib.Path(cache_dir) / "graph_cache" / f"{digest}.cta_graph"
    return build_streaming_model_from_artifact(
        artifact,
        cache_path=cache_path,
        goal_semantics=goal_semantics,
        collective_sync=collective_sync,
        memory_budget_bytes=memory_budget_bytes,
        scratch_dir=scratch_dir,
    )


def build_model(nodes: pl.DataFrame, edges: pl.DataFrame) -> WhatIfModel:
    """Prepares a graph for what-if solving.

    Args:
        nodes: The node table, as for ``dag.solve_distributed_dag``.
        edges: The intra-rank edge table.

    Returns:
        The model.
    """
    start = time.perf_counter()
    graph = dag.prepare_dag(nodes, edges)
    plan = native.DagPlan(graph.num_nodes, graph.edge_src, graph.edge_dst)
    return WhatIfModel(nodes, graph, plan, time.perf_counter() - start)


def load_model(
    source: str | pathlib.Path,
    *,
    cache_dir: str | pathlib.Path = ".cta_cache",
) -> WhatIfModel:
    """Reads a trace's execution graph and prepares it for what-if solving.

    Args:
        source: A Chakra ``.et`` trace or directory, or a GOAL schedule.
        cache_dir: Directory of the Parquet ingest cache.

    Returns:
        The model of the first trial.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The source holds no execution graph.
    """
    from cta import engine
    from cta.aspects.distributed import graph_tables
    from cta.ingest import cache
    from cta.ingest import discovery

    group = discovery.resolve_trial_group(source)
    tables = engine.ingest_group_tables(
        [group], cache.CacheStore(pathlib.Path(cache_dir))
    )
    try:
        nodes, edges, _ = graph_tables.read_graph_tables(
            tables[group.group_id][0]
        )
    except KeyError:
        raise ValueError(
            f"'{source}' holds no execution graph (Chakra or GOAL)"
        ) from None
    return build_model(nodes, edges)


@typing.overload
def reprice(
    model: WhatIfModel,
    new: interconnect.Interconnect,
    base: interconnect.Interconnect | None = None,
) -> WhatIfModel: ...


@typing.overload
def reprice(
    model: StreamingWhatIfModel,
    new: interconnect.Interconnect,
    base: interconnect.Interconnect | None = None,
) -> StreamingWhatIfModel: ...


def reprice(
    model: WhatIfModel | StreamingWhatIfModel,
    new: interconnect.Interconnect,
    base: interconnect.Interconnect | None = None,
) -> WhatIfModel | StreamingWhatIfModel:
    """Prices the graph's communication on another interconnect.

    A communication op with no recorded time (GOAL) takes the cost ``new``
    gives it. A recorded one keeps what the measurement holds (overheads,
    contention, a protocol the model does not know) and is scaled by how much
    ``new`` costs against ``base``, the interconnect the run was measured on;
    without ``base`` the recorded times are left as they are. With the default
    interconnect the weights are exactly the ones ``prepare_dag`` gave.

    Args:
        model: The prepared graph (in-memory or streaming).
        new: The interconnect to price the communication on.
        base: The interconnect the recorded times were measured on.

    Returns:
        A model of the same graph with new communication times. Rules and
        sweeps work on it as on any other.

    Warns:
        RuntimeWarning: A collective type has no group size, so every rank of
            the graph is assumed, or recorded times are kept for want of a
            ``base``.
    """
    if isinstance(model, StreamingWhatIfModel):
        cache_handle = native.GraphCache(str(model.cache_path))
        audit = cache_handle.inspect_reprice(
            _ic_to_native(new),
            _ic_to_native(base) if base is not None else None,
        )
        if audit["has_recorded_comm_without_base"]:
            warnings.warn(
                "The communication ops have recorded times and no baseline "
                "interconnect was given, so they keep them.",
                RuntimeWarning,
                stacklevel=2,
            )
        defaulted_coll = audit["defaulted_collectives"]
        if defaulted_coll:
            warnings.warn(
                f"No group size for {', '.join(defaulted_coll)}: assuming "
                "every rank of the graph (set group.<TYPE>=<ranks> in the "
                "interconnect).",
                RuntimeWarning,
                stacklevel=2,
            )
        return dataclasses.replace(
            model, reprice_steps=(*model.reprice_steps, (new, base))
        )

    nodes = model.nodes
    weights = model.graph.weights_ns.copy()
    is_comm = model.graph.is_comm
    recorded = nodes["duration_ns"].to_numpy().astype(np.float64) > 0.0
    cost_new, defaulted = interconnect.node_cost_ns(nodes, new)
    unrecorded = is_comm & ~recorded
    weights[unrecorded] = cost_new[unrecorded]
    scaled = is_comm & recorded
    if scaled.any():
        if base is None:
            warnings.warn(
                "The communication ops have recorded times and no baseline "
                "interconnect was given, so they keep them.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            cost_base, base_defaulted = interconnect.node_cost_ns(nodes, base)
            defaulted = sorted(set(defaulted) | set(base_defaulted))
            ratio = np.divide(
                cost_new,
                cost_base,
                out=np.ones_like(cost_new),
                where=cost_base > 0,
            )
            weights[scaled] = weights[scaled] * ratio[scaled]
    if defaulted:
        warnings.warn(
            f"No group size for {', '.join(defaulted)}: assuming every rank "
            "of the graph (set group.<TYPE>=<ranks> in the interconnect).",
            RuntimeWarning,
            stacklevel=2,
        )
    graph = dataclasses.replace(model.graph, weights_ns=weights)
    return dataclasses.replace(model, graph=graph)


@dataclasses.dataclass(slots=True, frozen=True)
class WhatIfRun:
    """The answers of one batch of scenarios.

    Attributes:
        table: One row per scenario, the unchanged graph first: ``scenario``,
            ``makespan_ns``, ``speedup`` (against the unchanged graph),
            ``cp_compute_ns``, ``cp_exposed_comm_ns``, ``cp_comm_bytes``
            and ``dag_depth``.
        critical_by_group: Time on the critical path per group of ops, one
            row per scenario.
        group_labels: What each group is, when the batch was built from
            groups (a sweep); empty otherwise.
        solve_seconds: Time of the native solve.
        lanes: Scenarios that shared one walk over the graph.
        workers: Threads used.
    """

    table: pl.DataFrame
    critical_by_group: np.ndarray
    group_labels: tuple[str, ...]
    solve_seconds: float
    lanes: int
    workers: int


def _solve(
    model: WhatIfModel,
    group: np.ndarray,
    scale: np.ndarray,
    scenario_labels: Sequence[str],
    group_labels: tuple[str, ...],
    threads: int,
    memory_budget_bytes: int,
) -> WhatIfRun:
    """Solves the scenarios in the rows of ``scale`` over ``group``."""
    graph = model.graph
    start = time.perf_counter()
    solved = model.plan.solve(
        graph.weights_ns,
        graph.is_comm,
        graph.is_compute,
        graph.size_bytes,
        np.ascontiguousarray(group, dtype=np.uint32),
        np.ascontiguousarray(scale, dtype=np.float64),
        threads,
        memory_budget_bytes,
    )
    seconds = time.perf_counter() - start
    makespan = solved["makespan"]
    baseline = makespan[0] if len(makespan) else 0.0
    table = pl.DataFrame(
        {
            "scenario": list(scenario_labels),
            "makespan_ns": makespan,
            "speedup": np.divide(
                baseline,
                makespan,
                out=np.ones_like(makespan),
                where=makespan > 0,
            ),
            "cp_compute_ns": solved["critical_compute"],
            "cp_exposed_comm_ns": solved["critical_comm"],
            "cp_comm_bytes": solved["critical_comm_bytes"],
            "dag_depth": solved["depth"],
        }
    )
    by_group = solved["critical_by_group"].reshape(len(scale), scale.shape[1])
    return WhatIfRun(
        table,
        by_group,
        group_labels,
        seconds,
        solved["lanes"],
        solved["workers"],
    )


class _Columns:
    """The node columns selectors read, converted once."""

    def __init__(self, model: WhatIfModel) -> None:
        nodes = model.nodes
        self.size = nodes.height
        self.model = model
        self.rank = nodes["rank"].cast(pl.Int64).to_numpy()
        self.is_comm = model.graph.is_comm
        self.is_compute = model.graph.is_compute

    def mask(self, atoms: tuple[tuple[str, str], ...]) -> np.ndarray:
        """Returns which vertices all the atoms select."""
        nodes = self.model.nodes
        selected = np.ones(self.size, dtype=np.bool_)
        for kind, argument in atoms:
            if kind == "all":
                continue
            if kind == "comm":
                selected &= self.is_comm
            elif kind == "compute":
                selected &= self.is_compute
            elif kind == "class":
                wanted = [part.strip() for part in argument.split(",")]
                selected &= nodes["op_class"].is_in(wanted).to_numpy()
            elif kind == "type":
                wanted = [part.strip().upper() for part in argument.split(",")]
                selected &= (
                    nodes["comm_type"]
                    .str.to_uppercase()
                    .is_in(wanted)
                    .fill_null(False)
                    .to_numpy()
                )
            elif kind == "family":
                wanted = [part.strip().lower() for part in argument.split(",")]
                selected &= (
                    self.model.labels.str.to_lowercase()
                    .is_in(wanted)
                    .to_numpy()
                )
            elif kind == "name":
                selected &= (
                    nodes["name"]
                    .str.contains(f"(?i){argument}")
                    .fill_null(False)
                    .to_numpy()
                )
            elif kind == "rank":
                ranges = _parse_ranks(argument)
                if ranges is None:
                    continue
                in_ranks = np.zeros(self.size, dtype=np.bool_)
                for low, high in ranges:
                    in_ranks |= (self.rank >= low) & (self.rank <= high)
                selected &= in_ranks
        return selected


def _median_rules(
    model: WhatIfModel, columns: _Columns, atoms: tuple[tuple[str, str], ...]
) -> list[tuple[tuple[tuple[str, str], ...], float]]:
    """Expands a ``median`` rule into one (selector, multiplier) per rank."""
    rank_atoms = [atom for atom in atoms if atom[0] == "rank"]
    if not rank_atoms:
        raise ValueError("The 'median' factor needs a rank= selector")
    other = tuple(atom for atom in atoms if atom[0] != "rank")
    busy_ns = np.bincount(
        columns.rank[columns.is_compute],
        weights=model.graph.weights_ns[columns.is_compute],
    )
    present = np.unique(columns.rank)
    chosen = columns.mask(tuple(rank_atoms))
    selected_ranks = np.unique(columns.rank[chosen])
    median_ns = float(np.median(busy_ns[present])) if len(present) else 0.0
    rules: list[tuple[tuple[tuple[str, str], ...], float]] = []
    for rank in selected_ranks:
        rank_busy = float(busy_ns[rank]) if rank < len(busy_ns) else 0.0
        if rank_busy <= median_ns or rank_busy <= 0.0:
            continue
        atoms_for_rank = (*other, ("rank", str(int(rank))), ("compute", ""))
        rules.append((atoms_for_rank, median_ns / rank_busy))
    return rules


def _dense(key: np.ndarray) -> np.ndarray:
    """Renumbers the distinct values of ``key`` as 0, 1, 2, ..."""
    return (
        pl.Series(key).rank("dense").to_numpy().astype(np.uint64, copy=False)
        - 1
    )


def _median_rules_streaming(
    rank_stats: Sequence[tuple[int, int, float]],
    atoms: tuple[tuple[str, str], ...],
) -> list[tuple[tuple[tuple[str, str], ...], float]]:
    """Expands a ``median`` rule from O(P) header rank compute stats."""
    rank_atoms = [atom for atom in atoms if atom[0] == "rank"]
    if not rank_atoms:
        raise ValueError("The 'median' factor needs a rank= selector")
    other = tuple(atom for atom in atoms if atom[0] != "rank")
    present = [int(r) for r, _, _ in rank_stats]
    busy = {int(r): float(comp_ns) for r, _, comp_ns in rank_stats}
    median_ns = float(np.median([busy[r] for r in present])) if present else 0.0
    selected = list(present)
    for _, arg in rank_atoms:
        ranges = _parse_ranks(arg)
        if ranges is None:
            continue
        selected = [
            r for r in selected if any(low <= r <= high for low, high in ranges)
        ]
    rules: list[tuple[tuple[tuple[str, str], ...], float]] = []
    for rank in selected:
        rank_busy = busy.get(rank, 0.0)
        if rank_busy <= median_ns or rank_busy <= 0.0:
            continue
        atoms_for_rank = (*other, ("rank", str(rank)), ("compute", ""))
        rules.append((atoms_for_rank, median_ns / rank_busy))
    return rules


def _global_name_families(global_names: Sequence[str]) -> np.ndarray:
    """Computes op_family once for every interned global name."""
    if not global_names:
        return np.empty(0, dtype=object)
    families = (
        pl.DataFrame({"name": list(global_names)})
        .select(alignment.op_family_expr("name").alias("f"))["f"]
        .to_list()
    )
    return np.array(families, dtype=object)


def _block_names(
    name_id: np.ndarray,
    local_names: Sequence[str],
    global_names_arr: np.ndarray,
) -> np.ndarray:
    """Materializes op names for one topological block."""
    if len(name_id) == 0:
        return np.empty(0, dtype=object)
    if int(name_id.min()) >= 0:
        return global_names_arr[name_id]
    out = np.empty(len(name_id), dtype=object)
    non_neg = name_id >= 0
    if non_neg.any():
        out[non_neg] = global_names_arr[name_id[non_neg]]
    for idx in np.nonzero(~non_neg)[0]:
        out[int(idx)] = local_names[int(idx)]
    return out


def _block_labels(
    name_id: np.ndarray,
    local_names: Sequence[str],
    op_class_idx: np.ndarray,
    comm_type_id: np.ndarray,
    global_families: np.ndarray,
    comm_types_arr: np.ndarray,
) -> np.ndarray:
    """Computes `op_label` for one topological block without full-graph RAM."""
    size = len(name_id)
    if size == 0:
        return np.empty(0, dtype=object)
    op_class_arr = _OP_CLASS_ARRAY[op_class_idx]
    comm_arr = np.where(
        comm_type_id >= 0,
        comm_types_arr[np.maximum(comm_type_id, 0)],
        "",
    )
    if int(name_id.min()) >= 0:
        fam_arr = global_families[name_id]
    else:
        fam_arr = np.empty(size, dtype=object)
        non_neg = name_id >= 0
        if non_neg.any():
            fam_arr[non_neg] = global_families[name_id[non_neg]]
        neg_idx = np.nonzero(~non_neg)[0]
        local_slice = [local_names[int(i)] for i in neg_idx]
        local_fams = (
            pl.DataFrame({"name": local_slice})
            .select(alignment.op_family_expr("name").alias("f"))["f"]
            .to_list()
        )
        for pos, fam in zip(neg_idx, local_fams, strict=True):
            fam_arr[int(pos)] = fam
    return np.where(
        fam_arr != "Other",
        fam_arr,
        np.where(comm_arr != "", comm_arr, op_class_arr),
    )


def _solve_streaming_raw(
    model: StreamingWhatIfModel,
    groups_path: str,
    scale: np.ndarray,
    scenario_labels: Sequence[str],
    group_labels: tuple[str, ...],
    threads: int,
    memory_budget_bytes: int,
) -> WhatIfRun:
    """Runs Engine D over a prepared groups file and formats WhatIfRun."""
    native_steps = _steps_to_native(model.reprice_steps)
    start = time.perf_counter()
    solved = native.solve_whatif_graph_cache(
        str(model.cache_path),
        groups_path,
        np.ascontiguousarray(scale, dtype=np.float64),
        reprice_steps=native_steps,
        threads=threads,
        memory_budget_bytes=memory_budget_bytes,
        scratch_dir=model.scratch_dir,
    )
    seconds = time.perf_counter() - start
    makespan = solved["makespan"]
    baseline = makespan[0] if len(makespan) else 0.0
    table = pl.DataFrame(
        {
            "scenario": list(scenario_labels),
            "makespan_ns": makespan,
            "speedup": np.divide(
                baseline,
                makespan,
                out=np.ones_like(makespan),
                where=makespan > 0,
            ),
            "cp_compute_ns": solved["critical_compute"],
            "cp_exposed_comm_ns": solved["critical_comm"],
            "cp_comm_bytes": solved["critical_comm_bytes"],
            "dag_depth": solved["depth"],
        }
    )
    by_group = solved["critical_by_group"].reshape(len(scale), scale.shape[1])
    return WhatIfRun(
        table,
        by_group,
        group_labels,
        seconds,
        int(solved["lanes"]),
        int(solved["workers"]),
    )


def _run_scenarios_streaming(
    model: StreamingWhatIfModel,
    scenarios: Sequence[Scenario],
    *,
    threads: int,
    memory_budget_bytes: int,
) -> WhatIfRun:
    """Streams selector evaluation per block and runs Engine D."""
    cache_handle = native.GraphCache(str(model.cache_path))
    rank_stats = cache_handle.rank_stats()

    atom_index: dict[tuple[tuple[str, str], ...], int] = {}
    atom_list: list[tuple[tuple[str, str], ...]] = []
    scenario_terms: list[list[tuple[int, float]]] = []
    for scenario in scenarios:
        terms: list[tuple[int, float]] = []
        for rule in scenario.rules:
            atoms = _parse_atoms(rule.selector)
            multiplier = parse_factor(rule.factor)
            expanded = (
                _median_rules_streaming(rank_stats, atoms)
                if multiplier is None
                else [(atoms, multiplier)]
            )
            for rule_atoms, rule_multiplier in expanded:
                if rule_atoms not in atom_index:
                    atom_index[rule_atoms] = len(atom_list)
                    atom_list.append(rule_atoms)
                terms.append((atom_index[rule_atoms], rule_multiplier))
        scenario_terms.append(terms)

    global_names = cache_handle.names()
    global_comm_types = cache_handle.comm_types()
    comm_types_arr = (
        np.array(global_comm_types, dtype=object)
        if global_comm_types
        else np.array([""], dtype=object)
    )
    comm_types_upper = np.array(
        [ct.upper() for ct in (global_comm_types or [""])], dtype=object
    )

    need_family = any(
        kind == "family" for atoms in atom_list for kind, _ in atoms
    )
    global_families = (
        _global_name_families(global_names)
        if need_family
        else np.empty(0, dtype=object)
    )
    name_regex_cache: dict[str, np.ndarray] = {}
    for atoms in atom_list:
        for kind, arg in atoms:
            if kind == "name" and arg not in name_regex_cache:
                name_regex_cache[arg] = (
                    pl.Series("n", global_names)
                    .str.contains(f"(?i){arg}")
                    .fill_null(False)
                    .to_numpy()
                    if global_names
                    else np.empty(0, dtype=np.bool_)
                )

    n_words = max(1, (len(atom_list) + _CHUNK_BITS - 1) // _CHUNK_BITS)
    mask_to_gid: dict[tuple[int, ...], int] = {}
    gid_to_mask: list[tuple[int, ...]] = []

    scratch_base = model.scratch_dir or None
    if scratch_base:
        pathlib.Path(scratch_base).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="cta_whatif_grp_", dir=scratch_base
    ) as tmp_dir:
        groups_path = pathlib.Path(tmp_dir) / "groups.bin"
        with groups_path.open("wb") as gfp:
            for block_idx in range(cache_handle.num_blocks):
                meta = cache_handle.read_block_meta(block_idx, [])
                rank = meta["rank"].astype(np.int64, copy=False)
                size = len(rank)
                if size == 0:
                    continue
                if not atom_list:
                    if not gid_to_mask:
                        zero_key = (0,)
                        mask_to_gid[zero_key] = 0
                        gid_to_mask.append(zero_key)
                    gfp.write(np.zeros(size, dtype=np.uint32).tobytes())
                    continue

                is_comm = meta["is_comm"].astype(np.bool_, copy=False)
                is_compute = meta["is_compute"].astype(np.bool_, copy=False)
                is_virtual = meta["is_virtual"].astype(np.bool_, copy=False)
                op_class_idx = meta["op_class"]
                comm_type_id = meta["comm_type_id"]
                name_id = meta["name_id"]
                local_names = meta["local_names"]
                op_class_arr = _OP_CLASS_ARRAY[op_class_idx]
                comm_upper_arr = np.where(
                    comm_type_id >= 0,
                    comm_types_upper[np.maximum(comm_type_id, 0)],
                    "",
                )
                block_fam_lower: np.ndarray | None = None
                if need_family:
                    lbl = _block_labels(
                        name_id,
                        local_names,
                        op_class_idx,
                        comm_type_id,
                        global_families,
                        comm_types_arr,
                    )
                    block_fam_lower = np.array(
                        [str(x).lower() for x in lbl], dtype=object
                    )

                words = [
                    np.zeros(size, dtype=np.uint64) for _ in range(n_words)
                ]
                for idx, atoms in enumerate(atom_list):
                    sel = np.ones(size, dtype=np.bool_)
                    for kind, arg in atoms:
                        if kind == "all":
                            continue
                        if kind == "comm":
                            sel &= is_comm
                        elif kind == "compute":
                            sel &= is_compute
                        elif kind == "class":
                            wanted = [p.strip() for p in arg.split(",")]
                            sel &= np.isin(op_class_arr, wanted)
                        elif kind == "type":
                            wanted = [p.strip().upper() for p in arg.split(",")]
                            sel &= np.isin(comm_upper_arr, wanted)
                        elif kind == "family":
                            wanted = [p.strip().lower() for p in arg.split(",")]
                            assert block_fam_lower is not None
                            sel &= np.isin(block_fam_lower, wanted)
                        elif kind == "name":
                            gmatch = name_regex_cache[arg]
                            if int(name_id.min()) >= 0:
                                sel &= gmatch[name_id]
                            else:
                                m = np.zeros(size, dtype=np.bool_)
                                non_neg = name_id >= 0
                                if non_neg.any():
                                    m[non_neg] = gmatch[name_id[non_neg]]
                                neg_idx = np.nonzero(~non_neg)[0]
                                local_m = (
                                    pl.Series(
                                        "n",
                                        [local_names[int(i)] for i in neg_idx],
                                    )
                                    .str.contains(f"(?i){arg}")
                                    .fill_null(False)
                                    .to_numpy()
                                )
                                m[neg_idx] = local_m
                                sel &= m
                        elif kind == "rank":
                            ranges = _parse_ranks(arg)
                            if ranges is None:
                                continue
                            in_r = np.zeros(size, dtype=np.bool_)
                            for low, high in ranges:
                                in_r |= (rank >= low) & (rank <= high)
                            sel &= in_r
                    sel &= ~is_virtual
                    words[idx // _CHUNK_BITS] |= sel.astype(
                        np.uint64
                    ) << np.uint64(idx % _CHUNK_BITS)

                if n_words == 1:
                    uniq, inv = np.unique(words[0], return_inverse=True)
                    remap = np.empty(len(uniq), dtype=np.uint32)
                    for i, val in enumerate(uniq):
                        key: tuple[int, ...] = (int(val),)
                        gid = mask_to_gid.get(key)
                        if gid is None:
                            gid = len(gid_to_mask)
                            mask_to_gid[key] = gid
                            gid_to_mask.append(key)
                        remap[i] = gid
                    block_gids = remap[inv]
                else:
                    stacked = np.column_stack(words)
                    uniq_rows, inv = np.unique(
                        stacked, axis=0, return_inverse=True
                    )
                    remap = np.empty(len(uniq_rows), dtype=np.uint32)
                    for i, row_vals in enumerate(uniq_rows):
                        row_key = tuple(int(x) for x in row_vals)
                        gid = mask_to_gid.get(row_key)
                        if gid is None:
                            gid = len(gid_to_mask)
                            mask_to_gid[row_key] = gid
                            gid_to_mask.append(row_key)
                        remap[i] = gid
                    block_gids = remap[inv]
                gfp.write(block_gids.tobytes())

        num_groups = max(len(gid_to_mask), 1)
        scale = np.ones((1 + len(scenarios), num_groups), dtype=np.float64)
        for gid, mask_key in enumerate(gid_to_mask):
            for row, terms in enumerate(scenario_terms, start=1):
                for idx, multiplier in terms:
                    if (
                        mask_key[idx // _CHUNK_BITS] >> (idx % _CHUNK_BITS)
                    ) & 1:
                        scale[row, gid] *= multiplier
        labels = ["baseline"] + [scenario.label for scenario in scenarios]
        return _solve_streaming_raw(
            model,
            str(groups_path),
            scale,
            labels,
            (),
            threads,
            memory_budget_bytes,
        )


def _sweep_streaming(
    model: StreamingWhatIfModel,
    *,
    by: str,
    factors: Sequence[float],
    max_groups: int,
    threads: int,
    memory_budget_bytes: int,
) -> SweepResult:
    """Streams group discovery and assignment per block and runs Engine D."""
    if by not in ("family", "class", "rank", "name"):
        raise ValueError(
            f"Cannot group by '{by}'. Use family, class, rank or name"
        )
    cache_handle = native.GraphCache(str(model.cache_path))
    native_steps = _steps_to_native(model.reprice_steps)
    global_names = cache_handle.names()
    global_names_arr = (
        np.array(global_names, dtype=object)
        if global_names
        else np.array([""], dtype=object)
    )
    global_comm_types = cache_handle.comm_types()
    comm_types_arr = (
        np.array(global_comm_types, dtype=object)
        if global_comm_types
        else np.array([""], dtype=object)
    )
    global_families = (
        _global_name_families(global_names)
        if by == "family"
        else np.empty(0, dtype=object)
    )

    def block_group_strings(meta: dict[str, typing.Any]) -> np.ndarray:
        if by == "family":
            return _block_labels(
                meta["name_id"],
                meta["local_names"],
                meta["op_class"],
                meta["comm_type_id"],
                global_families,
                comm_types_arr,
            )
        if by == "class":
            return _OP_CLASS_ARRAY[meta["op_class"]]
        if by == "rank":
            ranks = meta["rank"]
            return np.array([f"rank{int(r):05d}" for r in ranks], dtype=object)
        return _block_names(
            meta["name_id"], meta["local_names"], global_names_arr
        )

    # Pass 1 over block metadata: accumulate total repriced weight per label.
    totals: dict[str, float] = {}
    for block_idx in range(cache_handle.num_blocks):
        meta = cache_handle.read_block_meta(block_idx, native_steps)
        is_virtual = meta["is_virtual"].astype(np.bool_, copy=False)
        real_mask = ~is_virtual
        if not real_mask.any():
            continue
        lbls = block_group_strings(meta)[real_mask]
        weights = meta["weight_ns"][real_mask]
        agg = (
            pl.DataFrame({"label": lbls.tolist(), "weight": weights})
            .group_by("label")
            .agg(pl.col("weight").sum())
        )
        for row_dict in agg.iter_rows(named=True):
            lbl = str(row_dict["label"])
            totals[lbl] = totals.get(lbl, 0.0) + float(row_dict["weight"])

    sorted_totals = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [lbl for lbl, _ in sorted_totals[:max_groups]]
    has_rest = len(sorted_totals) > len(kept)
    group_labels = list(kept)
    if has_rest:
        group_labels.append(_OTHER)
    num_groups = len(group_labels)
    if num_groups == 0:
        empty_long = pl.DataFrame(
            schema={
                "group": pl.String,
                "factor": pl.Float64,
                "makespan_ns": pl.Float64,
                "speedup": pl.Float64,
                "gain_pct": pl.Float64,
                "bottleneck": pl.String,
                "bottleneck_share_pct": pl.Float64,
            }
        )
        empty_ceil = pl.DataFrame(
            schema={
                "group": pl.String,
                "time_share_pct": pl.Float64,
                "critical_share_pct": pl.Float64,
                "ceiling_speedup": pl.Float64,
                "knee_factor": pl.Float64,
            }
        )
        return SweepResult(empty_long, empty_ceil, 0.0, 0.0, 0, 0)

    label_to_gid = {lbl: idx for idx, lbl in enumerate(kept)}
    rest_gid = len(kept)
    time_ns = np.zeros(num_groups, dtype=np.float64)
    for idx, lbl in enumerate(kept):
        time_ns[idx] = totals[lbl]
    if has_rest:
        time_ns[rest_gid] = sum(w for _, w in sorted_totals[len(kept) :])

    speedups = list(dict.fromkeys(float(f) for f in factors))
    if math.inf not in speedups:
        speedups.append(math.inf)
    rows = 1 + num_groups * len(speedups)
    scale = np.ones((rows, num_groups), dtype=np.float64)
    scenario_labels = ["baseline"]
    row = 1
    for g, label in enumerate(group_labels):
        for factor in speedups:
            scale[row, g] = 0.0 if math.isinf(factor) else 1.0 / factor
            scenario_labels.append(f"{label} x{factor:g}")
            row += 1

    # Pass 2 over block metadata: stream uint32 group IDs to scratch groups.bin.
    scratch_base = model.scratch_dir or None
    if scratch_base:
        pathlib.Path(scratch_base).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="cta_whatif_swp_", dir=scratch_base
    ) as tmp_dir:
        groups_path = pathlib.Path(tmp_dir) / "groups.bin"
        with groups_path.open("wb") as gfp:
            for block_idx in range(cache_handle.num_blocks):
                meta = cache_handle.read_block_meta(block_idx, [])
                is_virtual = meta["is_virtual"].astype(np.bool_, copy=False)
                size = len(is_virtual)
                if size == 0:
                    continue
                lbls = block_group_strings(meta)
                gids = np.array(
                    [label_to_gid.get(str(x), rest_gid) for x in lbls],
                    dtype=np.uint32,
                )
                if is_virtual.any():
                    gids[is_virtual] = 0
                gfp.write(gids.tobytes())

        run = _solve_streaming_raw(
            model,
            str(groups_path),
            scale,
            scenario_labels,
            tuple(group_labels),
            threads,
            memory_budget_bytes,
        )

    makespan = run.table["makespan_ns"].to_numpy()
    baseline = float(makespan[0])
    by_group = run.critical_by_group
    top = np.argmax(by_group, axis=1)
    top_share = np.divide(
        by_group[np.arange(rows), top] * 100.0,
        makespan,
        out=np.zeros(rows),
        where=makespan > 0,
    )
    long_rows = (
        pl.DataFrame(
            {
                "group": [label for label in group_labels for _ in speedups],
                "factor": np.tile(np.asarray(speedups), num_groups),
                "makespan_ns": makespan[1:],
                "speedup": run.table["speedup"].to_numpy()[1:],
                "bottleneck": [group_labels[i] for i in top[1:]],
                "bottleneck_share_pct": top_share[1:],
            }
        )
        .with_columns(((pl.col("speedup") - 1.0) * 100.0).alias("gain_pct"))
        .select(
            "group",
            "factor",
            "makespan_ns",
            "speedup",
            "gain_pct",
            "bottleneck",
            "bottleneck_share_pct",
        )
    )
    total_ns = float(time_ns.sum()) or 1.0
    ceiling_rows: list[dict[str, object]] = []
    for g, label in enumerate(group_labels):
        block = long_rows.filter(pl.col("group") == label).sort("factor")
        gains = block["speedup"].to_numpy() - 1.0
        factor_values = block["factor"].to_numpy()
        ceiling_gain = float(gains[-1])
        ceiling = 1.0 + ceiling_gain
        knee = math.inf
        if ceiling_gain > 0.0:
            reached = np.nonzero(gains >= 0.9 * ceiling_gain)[0]
            if len(reached):
                knee = float(factor_values[reached[0]])
        ceiling_rows.append(
            {
                "group": label,
                "time_share_pct": float(time_ns[g]) / total_ns * 100.0,
                "critical_share_pct": (
                    float(by_group[0, g]) / baseline * 100.0
                    if baseline > 0
                    else 0.0
                ),
                "ceiling_speedup": ceiling,
                "knee_factor": knee,
            }
        )
    ceiling_table = pl.DataFrame(ceiling_rows).sort(
        "ceiling_speedup", descending=True
    )
    return SweepResult(
        long_rows,
        ceiling_table,
        baseline,
        run.solve_seconds,
        run.lanes,
        run.workers,
    )


def run_scenarios(
    model: WhatIfModel | StreamingWhatIfModel,
    scenarios: Sequence[Scenario],
    *,
    threads: int = 0,
    memory_budget_bytes: int = 0,
) -> WhatIfRun:
    """Solves scenarios of a graph, each against the recorded run.

    Ops are put into groups that no rule tells apart, and each scenario
    scales the groups; so the cost of a scenario does not depend on how many
    rules it has.

    Args:
        model: The prepared graph (in-memory or streaming).
        scenarios: The scenarios. The unchanged graph is solved first and
            appears as ``baseline``.
        threads: Worker threads; 0 uses every usable CPU.
        memory_budget_bytes: Memory the solve may use; 0 uses about 60% of
            what the machine has free. Each worker needs 12 bytes per op per
            lane in memory, or O(block_vertices) in streaming mode.

    Returns:
        The results, the unchanged graph first.

    Raises:
        ValueError: A rule is malformed or ``median`` has no rank selector.
        RuntimeError: The graph does not fit the memory budget.
    """
    if isinstance(model, StreamingWhatIfModel):
        return _run_scenarios_streaming(
            model,
            scenarios,
            threads=threads,
            memory_budget_bytes=memory_budget_bytes,
        )
    columns = _Columns(model)
    size = columns.size
    # Every distinct selector, with the multiplier each scenario gives it.
    atom_index: dict[tuple[tuple[str, str], ...], int] = {}
    atom_list: list[tuple[tuple[str, str], ...]] = []
    scenario_terms: list[list[tuple[int, float]]] = []
    for scenario in scenarios:
        terms: list[tuple[int, float]] = []
        for rule in scenario.rules:
            atoms = _parse_atoms(rule.selector)
            multiplier = parse_factor(rule.factor)
            expanded = (
                _median_rules(model, columns, atoms)
                if multiplier is None
                else [(atoms, multiplier)]
            )
            for rule_atoms, rule_multiplier in expanded:
                if rule_atoms not in atom_index:
                    atom_index[rule_atoms] = len(atom_list)
                    atom_list.append(rule_atoms)
                terms.append((atom_index[rule_atoms], rule_multiplier))
        scenario_terms.append(terms)

    # One 64-bit word per 64 selectors holds which of them select each op.
    words: list[np.ndarray] = []
    for index, atoms in enumerate(atom_list):
        if index % _CHUNK_BITS == 0:
            words.append(np.zeros(size, dtype=np.uint64))
        words[-1] |= columns.mask(atoms).astype(np.uint64) << np.uint64(
            index % _CHUNK_BITS
        )
    if not words:
        group = np.zeros(size, dtype=np.uint64)
    else:
        group = _dense(words[0])
        for word in words[1:]:
            group = _dense((group << np.uint64(32)) | _dense(word))
    num_groups = int(group.max()) + 1 if size else 1
    first = pl.Series(group).arg_unique().to_numpy()
    representative = np.empty(num_groups, dtype=np.int64)
    representative[group[first].astype(np.int64)] = first

    scale = np.ones((1 + len(scenarios), num_groups), dtype=np.float64)
    for row, terms in enumerate(scenario_terms, start=1):
        for index, multiplier in terms:
            bits = (
                words[index // _CHUNK_BITS][representative]
                >> np.uint64(index % _CHUNK_BITS)
            ) & np.uint64(1)
            scale[row, bits.astype(np.bool_)] *= multiplier
    labels = ["baseline"] + [scenario.label for scenario in scenarios]
    return _solve(model, group, scale, labels, (), threads, memory_budget_bytes)


@dataclasses.dataclass(slots=True, frozen=True)
class SweepResult:
    """The result of speeding up every group of ops in turn.

    Attributes:
        table: One row per (group, speedup): ``group``, ``factor``,
            ``makespan_ns``, ``speedup``, ``gain_pct``, ``bottleneck`` (the
            group with most time on the critical path afterwards) and
            ``bottleneck_share_pct`` (its share of the new critical path).
        ceiling: One row per group, most valuable first: ``group``,
            ``time_share_pct`` (of all op time), ``critical_share_pct`` (of
            the critical path), ``ceiling_speedup`` (if the group took no
            time at all) and ``knee_factor`` (the smallest swept speedup
            that gets 90% of that gain; ``inf`` if only removal does).
        baseline_makespan_ns: The unchanged graph's critical path.
        solve_seconds: Time of the native solve.
        lanes: Scenarios that shared one walk over the graph.
        workers: Threads used.
    """

    table: pl.DataFrame
    ceiling: pl.DataFrame
    baseline_makespan_ns: float
    solve_seconds: float
    lanes: int
    workers: int


def _group_ids(
    model: WhatIfModel, by: str, max_groups: int
) -> tuple[np.ndarray, list[str]]:
    """Assigns every op to a group; the heaviest groups keep their own."""
    nodes = model.nodes
    if by == "family":
        names = model.labels
    elif by == "class":
        names = nodes["op_class"].fill_null("")
    elif by == "rank":
        names = nodes["rank"].cast(pl.String).str.zfill(5)
        names = "rank" + names
    elif by == "name":
        names = nodes["name"].fill_null("")
    else:
        raise ValueError(
            f"Cannot group by '{by}'. Use family, class, rank or name"
        )
    frame = pl.DataFrame({"label": names, "weight": model.graph.weights_ns})
    totals = (
        frame.group_by("label")
        .agg(pl.col("weight").sum())
        .sort(["weight", "label"], descending=[True, False])
    )
    kept = totals["label"].head(max_groups).to_list()
    has_rest = totals.height > len(kept)
    mapping = pl.DataFrame(
        {"label": kept, "gid": np.arange(len(kept), dtype=np.uint32)}
    )
    ids = (
        frame.select("label")
        .join(mapping, on="label", how="left")["gid"]
        .fill_null(len(kept))
        .to_numpy()
    )
    labels = list(kept)
    if has_rest:
        labels.append(_OTHER)
    return ids.astype(np.uint32, copy=False), labels


def sweep(
    model: WhatIfModel | StreamingWhatIfModel,
    *,
    by: str = "family",
    factors: Sequence[float] = DEFAULT_FACTORS,
    max_groups: int = 48,
    threads: int = 0,
    memory_budget_bytes: int = 0,
) -> SweepResult:
    """Speeds up each group of ops in turn and measures the whole run.

    Args:
        model: The prepared graph (in-memory or streaming).
        by: How to group ops: ``family`` (see ``op_label``), ``class``,
            ``rank`` or ``name``.
        factors: Speedups to try. ``inf`` (removing the group's time) is
            always added, as the ceiling of each group.
        max_groups: At most this many groups, heaviest first; the rest are
            pooled as ``(other)``.
        threads: Worker threads; 0 uses every usable CPU.
        memory_budget_bytes: Memory the solve may use; 0 uses about 60% of
            what the machine has free.

    Returns:
        The sweep.

    Raises:
        ValueError: ``by`` is unknown.
        RuntimeError: The graph does not fit the memory budget.
    """
    if isinstance(model, StreamingWhatIfModel):
        return _sweep_streaming(
            model,
            by=by,
            factors=factors,
            max_groups=max_groups,
            threads=threads,
            memory_budget_bytes=memory_budget_bytes,
        )
    group, group_labels = _group_ids(model, by, max_groups)
    num_groups = len(group_labels)
    speedups = list(dict.fromkeys(float(f) for f in factors))
    if math.inf not in speedups:
        speedups.append(math.inf)  # the ceiling of every group
    rows = 1 + num_groups * len(speedups)
    scale = np.ones((rows, num_groups), dtype=np.float64)
    scenario_labels = ["baseline"]
    row = 1
    for g, label in enumerate(group_labels):
        for factor in speedups:
            scale[row, g] = 0.0 if math.isinf(factor) else 1.0 / factor
            scenario_labels.append(f"{label} x{factor:g}")
            row += 1
    run = _solve(
        model,
        group,
        scale,
        scenario_labels,
        tuple(group_labels),
        threads,
        memory_budget_bytes,
    )
    makespan = run.table["makespan_ns"].to_numpy()
    baseline = float(makespan[0])
    by_group = run.critical_by_group
    top = np.argmax(by_group, axis=1)
    top_share = np.divide(
        by_group[np.arange(rows), top] * 100.0,
        makespan,
        out=np.zeros(rows),
        where=makespan > 0,
    )
    long_rows = (
        pl.DataFrame(
            {
                "group": [label for label in group_labels for _ in speedups],
                "factor": np.tile(np.asarray(speedups), num_groups),
                "makespan_ns": makespan[1:],
                "speedup": run.table["speedup"].to_numpy()[1:],
                "bottleneck": [group_labels[i] for i in top[1:]],
                "bottleneck_share_pct": top_share[1:],
            }
        )
        .with_columns(((pl.col("speedup") - 1.0) * 100.0).alias("gain_pct"))
        .select(
            "group",
            "factor",
            "makespan_ns",
            "speedup",
            "gain_pct",
            "bottleneck",
            "bottleneck_share_pct",
        )
    )

    weights = model.graph.weights_ns
    time_ns = np.bincount(group, weights=weights, minlength=num_groups)
    total_ns = float(time_ns.sum()) or 1.0
    ceiling_rows: list[dict[str, object]] = []
    for g, label in enumerate(group_labels):
        block = long_rows.filter(pl.col("group") == label).sort("factor")
        gains = block["speedup"].to_numpy() - 1.0
        factor_values = block["factor"].to_numpy()
        ceiling_gain = float(gains[-1])  # the last factor is inf
        ceiling = 1.0 + ceiling_gain
        knee = math.inf
        if ceiling_gain > 0.0:
            reached = np.nonzero(gains >= 0.9 * ceiling_gain)[0]
            if len(reached):
                knee = float(factor_values[reached[0]])
        ceiling_rows.append(
            {
                "group": label,
                "time_share_pct": float(time_ns[g]) / total_ns * 100.0,
                "critical_share_pct": (
                    float(by_group[0, g]) / baseline * 100.0
                    if baseline > 0
                    else 0.0
                ),
                "ceiling_speedup": ceiling,
                "knee_factor": knee,
            }
        )
    ceiling_table = pl.DataFrame(ceiling_rows).sort(
        "ceiling_speedup", descending=True
    )
    return SweepResult(
        long_rows,
        ceiling_table,
        baseline,
        run.solve_seconds,
        run.lanes,
        run.workers,
    )
