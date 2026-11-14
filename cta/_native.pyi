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

def nesting_parents(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Returns the enclosing range of each range, -1 for none."""
    ...

def __getattr__(name: str) -> Any: ...
