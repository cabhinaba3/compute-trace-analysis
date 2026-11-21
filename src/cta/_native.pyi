"""Type stub for the compiled ``cta._native`` extension built from ``native/``.

One flat module: the readers return dicts of NumPy columns (see the column
structs in ``native/src/readers``), the writers back ``cta.synth``, and the
analysis functions back ``cta.analysis``. Names, arguments and defaults match
``native/src/bindings``.
"""

from typing import Any

import numpy as np

# Readers.

def read_kineto(
    path: str,
    threads: int = 0,
    chunk_bytes: int = 33554432,
    block_bytes: int = 8388608,
) -> dict[str, Any]:
    """Reads a Kineto trace (.json or .json.gz) into coded columns."""
    ...

def read_chakra(
    paths: list[str], ranks: list[int], threads: int = 0
) -> dict[str, Any]:
    """Reads Chakra .et files into dictionary-coded node and edge columns."""
    ...

def read_nsys(path: str) -> dict[str, Any]:
    """Reads the GPU activities, NVTX ranges and GPU metrics of an export."""
    ...

def read_goal(
    path: str, threads: int = 0, chunk_bytes: int = 33554432
) -> dict[str, Any]:
    """Reads a GOAL text schedule into node and edge columns."""
    ...

def read_goal_bin(path: str) -> dict[str, Any]:
    """Reads a GOAL binary (.bin) schedule into node and edge columns."""
    ...

# Streaming readers and memory helpers.

def available_memory_bytes() -> int:
    """Returns available host/cgroup memory in bytes, or 0 if unavailable."""
    ...

def resolve_memory_budget(requested_bytes: int = 0) -> int:
    """Returns requested_bytes if > 0, else 60% of available memory."""
    ...

class GraphStream:
    """Bounded-memory streaming reader over a GOAL or Chakra graph."""

    is_seekable: bool
    format_name: str
    estimated_bytes: int

    def next_chunk(self, max_nodes: int = 0) -> dict[str, Any] | None:
        """Returns the next chunk dict of NumPy arrays, or None at EOF."""
        ...

    def __iter__(self) -> GraphStream: ...
    def __next__(self) -> dict[str, Any]: ...
    def reset(self) -> None:
        """Resets a seekable stream back to the start."""
        ...

def open_goal_bin_stream(path: str, chunk_nodes: int = 524288) -> GraphStream:
    """Opens a GOAL binary (.bin) file or pipe as a chunked GraphStream."""
    ...

def open_goal_text_stream(
    path: str,
    goal_semantics: str = "cta_legacy",
    chunk_nodes: int = 524288,
) -> GraphStream:
    """Opens a GOAL text (.goal) file or pipe as a chunked GraphStream."""
    ...

def open_chakra_et_stream(
    paths: list[str], ranks: list[int], chunk_nodes: int = 524288
) -> GraphStream:
    """Opens Chakra .et protobuf files as a chunked GraphStream."""
    ...

def open_chakra_host_stream(
    paths: list[str], ranks: list[int], chunk_nodes: int = 524288
) -> GraphStream:
    """Opens Chakra host JSON files as a chunked GraphStream."""
    ...

def stream_aggregates(
    stream: GraphStream,
    include_collectives_in_traffic: bool = True,
    min_world_size: int = 1,
    quantile_sketch: str = "ddsketch",
    quantile_exact_cap: int = 65536,
    quantile_accuracy: float = 0.001,
    chunk_nodes: int = 524288,
) -> dict[str, Any]:
    """Runs Engine A (1-pass O(P^2) aggregate analysis) over a GraphStream."""
    ...

class HybridQuantile:
    """Exact-first quantile estimator spilling to DDSketch, KLL, or GK."""

    count: int
    is_exact: bool
    sketch_name: str
    error_bound: float

    def __init__(
        self,
        sketch_kind: str = "ddsketch",
        exact_cap: int = 65536,
        accuracy: float = 0.001,
    ) -> None: ...
    def add(self, value: int, weight: int = 1) -> None: ...
    def add_array(self, values: np.ndarray) -> None: ...
    def quantile(self, q: float) -> float: ...

def available_graph_engines() -> list[str]:
    """Returns the registered pluggable streaming/out-of-core DAG engines."""
    ...

def solve_dag_stream(
    stream: GraphStream,
    engine: str = "frontier",
    memory_budget_bytes: int = 0,
    goal_semantics: str = "cta_legacy",
    collective_sync: str = "none",
    world_size_hint: int = 0,
    coll_group_sizes: dict[str, int] | None = None,
    window_size: int = 262144,
    max_lookahead: int = 65536,
    fail_on_cycle: bool = False,
    extract_critical_path_ops: bool = False,
    delay_attribution: bool = False,
    scratch_dir: str = "",
    chunk_nodes: int = 524288,
) -> dict[str, Any]:
    """Solves an execution graph via a registered streaming GraphEngine."""
    ...

def solve_goal_bin_frontier(
    path: str,
    goal_semantics: str = "cta_legacy",
    fail_on_cycle: bool = False,
    extract_critical_path_ops: bool = False,
    scratch_dir: str = "",
) -> dict[str, Any]:
    """Exact O(frontier) Kahn DAG solver directly over mmapped GOAL .bin."""
    ...

def solve_chakra_et_frontier(
    paths: list[str],
    ranks: list[int],
    collective_sync: str = "none",
    coll_group_sizes: dict[str, int] | None = None,
    window_size: int = 262144,
    max_lookahead: int = 65536,
    fail_on_cycle: bool = False,
    extract_critical_path_ops: bool = False,
    scratch_dir: str = "",
) -> dict[str, Any]:
    """Lockstep 1-pass frontier DAG solver over multi-rank Chakra .et files."""
    ...

# Analysis.

def build_dag_edges(
    rank: np.ndarray,
    node_id: np.ndarray,
    is_send: np.ndarray,
    is_recv: np.ndarray,
    src_rank: np.ndarray,
    dst_rank: np.ndarray,
    tag: np.ndarray,
    e_rank: np.ndarray,
    e_src: np.ndarray,
    e_dst: np.ndarray,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Returns the vertex-id edges of a graph and where the time went."""
    ...

def solve_dag(
    weights: np.ndarray,
    is_comm: np.ndarray,
    is_compute: np.ndarray,
    size_bytes: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> dict[str, Any]:
    """Returns the critical path and slack of a graph as a dict."""
    ...

def nesting_parents(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Returns the enclosing range of each range, -1 for none."""
    ...

class DagPlan:
    """An execution graph prepared for repeated what-if solving."""

    num_vertices: int
    num_edges: int
    unordered: int

    def __init__(
        self, num_vertices: int, edge_src: np.ndarray, edge_dst: np.ndarray
    ) -> None: ...
    def solve(
        self,
        weights: np.ndarray,
        is_comm: np.ndarray,
        is_compute: np.ndarray,
        size_bytes: np.ndarray,
        group: np.ndarray,
        scale: np.ndarray,
        threads: int = 0,
        memory_budget_bytes: int = 0,
    ) -> dict[str, Any]:
        """Solves one scenario per row of ``scale`` (scenarios x groups)."""
        ...

# Synthetic trace writers.

def write_kineto_trace(
    path: str,
    steps: int = 3,
    attn_us: float = 400.0,
    gemm_us: float = 600.0,
    kernel_variant: str = "fp16",
) -> None:
    """Writes a small Kineto trace."""
    ...

def write_kineto_trace_large(
    path: str,
    target_bytes: int = 209715200,
    seed: int = 1,
    rank: int = 0,
    world_size: int = 1,
    layers_per_step: int = 24,
    child_cpu_ops_per_layer: int = 3,
    python_events_per_layer: int = 2,
    tensors_per_op: int = 3,
    distributed_info_after: bool = True,
) -> int:
    """Writes a Kineto trace of about ``target_bytes``; returns its size."""
    ...

def write_goal_allreduce(
    path: str,
    world_size: int = 4,
    compute_ns: int = 1000000,
    allreduce_bytes: int = 400000,
    steps: int = 2,
    algorithm: str = "ring",
) -> None:
    """Writes a GOAL text schedule of compute steps and allreduces."""
    ...

def write_goal_allreduce_bin(
    path: str,
    world_size: int = 4,
    compute_ns: int = 1000000,
    allreduce_bytes: int = 400000,
    steps: int = 2,
    algorithm: str = "ring",
) -> int:
    """Writes the same schedule as LogGOPSim binary; returns its size."""
    ...

def write_nsys_sqlite(
    path: str,
    steps: int = 4,
    gemm_us: int = 800,
    comm_us: int = 400,
    comm_overlap_us: int = 100,
    warmup_penalty_us: int = 1200,
    lock_wait_us: int = 50,
    tensor_active_pct: float = 65.0,
    legacy_gpu_metrics: bool = False,
) -> None:
    """Writes an Nsight Systems SQLite export."""
    ...

def write_chakra_run(
    out_dir: str,
    world_size: int = 4,
    compute_us: int = 1000,
    allreduce_bytes: int = 400000,
    steps: int = 2,
    straggler_rank: int | None = None,
    straggler_factor: float = 1.0,
) -> list[str]:
    """Writes one Chakra .et file per rank; returns their paths."""
    ...

def write_chakra_host(
    path: str,
    steps: int = 2,
    compute_us: float = 500.0,
    comm_bytes: int = 200000,
) -> None:
    """Writes a Chakra host execution trace (JSON)."""
    ...

def write_ncu_csv(
    path: str,
    gemm_dur_ns: float = 500000.0,
    sm_pct: float = 40.0,
    dram_pct: float = 82.0,
    tensor_pct: float = 35.0,
    flops: float = 2.0e10,
    dram_bytes: float = 1.0e9,
) -> None:
    """Writes an Nsight Compute CSV export."""
    ...

def write_jtop_trace(
    path: str,
    duration_s: int = 5,
    num_cores: int = 8,
    power_mw: float = 15000.0,
    gpu_util: float = 85.0,
    gpu_mem_mb: float = 2048.0,
    temp_c: float | None = 62.0,
) -> None:
    """Writes a Jetson jtop telemetry trace."""
    ...

def write_trtexec_timing(
    path: str,
    queries: int = 20,
    h2d_ms: float = 0.4,
    compute_ms: float = 2.0,
    d2h_ms: float = 0.3,
    overlap_ms: float = 0.2,
    idle_gap_ms: float = 0.1,
) -> None:
    """Writes a trtexec --exportTimes JSON file."""
    ...

def write_trtexec_layers(
    profile_path: str,
    engine_path: str,
    fused: bool = False,
    precision: str = "FP16",
) -> None:
    """Writes trtexec --exportProfile and --exportLayerInfo JSON files."""
    ...

class TrainingSpec:
    """Shape and costs of a 3D-parallel training graph."""

    dp: int
    pp: int
    tp: int
    microbatches: int
    layers_per_stage: int
    steps: int
    moe_every: int
    ep_size: int
    dp_buckets: int
    layer_fwd_ns: int
    layer_bwd_ns: int
    optimizer_ns: int
    expert_ns: int
    tp_bytes: int
    pp_bytes: int
    dp_bytes: int
    moe_bytes: int

    def __init__(self) -> None: ...
    def ranks(self) -> int:
        """Returns dp * pp * tp."""
        ...

    def validate(self) -> None:
        """Raises ValueError if the spec is inconsistent."""
        ...

def write_goal_training(path: str, spec: TrainingSpec) -> int:
    """Writes the graph as GOAL text; returns the bytes written."""
    ...

def write_goal_training_bin(path: str, spec: TrainingSpec) -> int:
    """Writes the graph as LogGOPSim binary; returns the bytes written."""
    ...

def write_chakra_training(
    out_dir: str, spec: TrainingSpec, threads: int = 0
) -> int:
    """Writes the graph as one Chakra .et file per rank; returns the bytes."""
    ...
