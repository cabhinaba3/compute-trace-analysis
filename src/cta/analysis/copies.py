"""Which way a memory copy goes, read from the name a profiler gave it."""

from __future__ import annotations

import re

DIRECTIONS = (
    ("memset", re.compile(r"memset", re.I)),
    ("H2D", re.compile(r"htod|h2d", re.I)),
    ("D2H", re.compile(r"dtoh|d2h", re.I)),
    ("D2D", re.compile(r"dtod|d2d", re.I)),
    ("P2P", re.compile(r"ptop|p2p|peer", re.I)),
    ("H2H", re.compile(r"htoh|h2h", re.I)),
)
DIRECTION_NAMES = [name for name, _ in DIRECTIONS] + ["other"]


def direction_of(name: str) -> str:
    """Names the direction of a copy from its kernel name.

    Args:
        name: A copy's name, such as ``Memcpy_H2D`` (nsys) or
            ``Memcpy HtoD (Pinned -> Device)`` (Kineto).

    Returns:
        ``H2D``, ``D2H``, ``D2D``, ``P2P``, ``H2H`` or ``memset``; ``other``
        when the name says none of them.
    """
    for direction, pattern in DIRECTIONS:
        if pattern.search(name):
            return direction
    return "other"
