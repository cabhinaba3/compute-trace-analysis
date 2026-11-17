"""Type stub for the compiled ``cta._native`` extension built from ``native/``.

One flat module: the readers return dicts of NumPy columns (see the column
structs in ``native/src/readers``), the writers back ``cta.synth``, the
analysis functions back ``cta.analysis`` and the simulator backs ``cta.sim``.
Names, arguments and defaults match ``native/src/bindings``.
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

# Simulator (see native/src/sim and cta.sim).

def simulate(
    graph: dict[str, np.ndarray],
    group_offsets: np.ndarray,
    group_members: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Runs a graph; returns per-op, per-rank, critical-path, link columns."""
    ...

def lower_collectives(
    graph: dict[str, np.ndarray],
    group_offsets: np.ndarray,
    group_members: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Returns the graph with collectives replaced by their algorithms."""
    ...

def collective_schedule(
    coll_type: int,
    algorithm: str,
    size: int,
    bytes: int,
    root: int = 0,
    node_size: int = 1,
    chunks: int = 1,
    reduce: bool = False,
    all_to_all_per_peer: bool = False,
) -> dict[str, np.ndarray]:
    """Returns every member's actions of one collective."""
    ...

def collective_times(
    coll_type: int,
    bytes: int,
    members: list[int],
    params: dict[str, Any],
    root: int = 0,
) -> list[int]:
    """Returns each member's contention-free finish time in ns."""
    ...

def algorithm_supported(
    coll_type: int, algorithm: str, size: int, node_size: int = 1
) -> bool:
    """Whether an algorithm can run a collective type on a group size."""
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
