"""Type stub for the compiled ``cta._native`` extension built from ``native/``.

Only the Kineto reader is typed; every other function is reachable through the
module-level ``__getattr__`` and is typed as ``Any``.
"""

from typing import Any

import numpy as np

def read_kineto_trace(
    path: str,
    threads: int = 0,
    chunk_bytes: int = 33554432,
    block_bytes: int = 8388608,
) -> dict[str, Any]:
    """Reads a Kineto trace (.json or .json.gz) into coded columns."""
    ...

def read_chakra_et(
    paths: list[str], ranks: list[int], threads: int = 0
) -> dict[str, Any]:
    """Reads Chakra .et files into dictionary-coded node and edge columns."""
    ...

def read_goal_text(
    path: str, threads: int = 0, chunk_bytes: int = 33554432
) -> dict[str, Any]:
    """Reads a GOAL text schedule into node and edge columns."""
    ...

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

def __getattr__(name: str) -> Any: ...
