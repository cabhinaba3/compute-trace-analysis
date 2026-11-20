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
import math
import pathlib
import re
import time

import numpy as np
import polars as pl

from cta import _native as native
from cta.analysis import alignment
from cta.analysis import dag

DEFAULT_FACTORS = (1.1, 1.25, 1.5, 2.0, 4.0, math.inf)

_CHUNK_BITS = 64
_OTHER = "(other)"


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


def run_scenarios(
    model: WhatIfModel,
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
        model: The prepared graph.
        scenarios: The scenarios. The unchanged graph is solved first and
            appears as ``baseline``.
        threads: Worker threads; 0 uses every usable CPU.
        memory_budget_bytes: Memory the solve may use; 0 uses about 60% of
            what the machine has free. Each worker needs 12 bytes per op per
            lane.

    Returns:
        The results, the unchanged graph first.

    Raises:
        ValueError: A rule is malformed or ``median`` has no rank selector.
        RuntimeError: The graph does not fit the memory budget.
    """
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
    model: WhatIfModel,
    *,
    by: str = "family",
    factors: Sequence[float] = DEFAULT_FACTORS,
    max_groups: int = 48,
    threads: int = 0,
    memory_budget_bytes: int = 0,
) -> SweepResult:
    """Speeds up each group of ops in turn and measures the whole run.

    Args:
        model: The prepared graph.
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
