"""Parses quantities with units: times, sizes, bandwidths, rates, FLOP rates.

Each parser takes a number (already in the base unit) or a string such as
``"1.5us"``, ``"64KiB"``, ``"400Gbps"`` or ``"1kHz"``, and returns the value
in the simulator's base unit: nanoseconds, bytes, bytes per nanosecond (which
is GB/s), events per nanosecond and FLOP per second.
"""

from __future__ import annotations

import re

_NUMBER_AND_UNIT = re.compile(
    r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z/]*)\s*$"
)

_TIME_NS = {
    "": 1.0,
    "ns": 1.0,
    "us": 1e3,
    "µs": 1e3,
    "ms": 1e6,
    "s": 1e9,
}
_SIZE_BYTES = {
    "": 1.0,
    "b": 1.0,
    "kb": 1e3,
    "mb": 1e6,
    "gb": 1e9,
    "tb": 1e12,
    "kib": 2.0**10,
    "mib": 2.0**20,
    "gib": 2.0**30,
    "tib": 2.0**40,
}
# A bandwidth unit: an SI or binary prefix, "b"/"bit" (bits) or "B"/"byte"
# (bytes), then "ps" or "/s". Case matters only for b versus B.
_BANDWIDTH_UNIT = re.compile(r"^([kKmMgGtT]?)(i?)(bit|byte|b|B)(ps|/s)$")
_PREFIX = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}
_BINARY_PREFIX = {
    "": 1.0,
    "k": 2.0**10,
    "m": 2.0**20,
    "g": 2.0**30,
    "t": 2.0**40,
}
_RATE_PER_SECOND = {"": 1.0, "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
_FLOPS = {
    "": 1.0,
    "flops": 1.0,
    "flop/s": 1.0,
    "gflops": 1e9,
    "gflop/s": 1e9,
    "tflops": 1e12,
    "tflop/s": 1e12,
    "pflops": 1e15,
    "pflop/s": 1e15,
}


def _parse(
    value: float | int | str, units: dict[str, float], kind: str
) -> float:
    """Returns value times the factor of its unit; numbers pass unchanged."""
    if isinstance(value, bool):
        raise ValueError(f"expected a {kind}, got {value!r}")
    if isinstance(value, int | float):
        return float(value)
    match = _NUMBER_AND_UNIT.match(str(value))
    if match is None:
        raise ValueError(f"cannot read {value!r} as a {kind}")
    number, unit = match.groups()
    factor = units.get(unit.lower())
    if factor is None:
        raise ValueError(
            f"unknown {kind} unit {unit!r} in {value!r}; "
            f"use one of {', '.join(sorted(k for k in units if k))}"
        )
    return float(number) * factor


def time_ns(value: float | int | str) -> float:
    """Returns a duration in nanoseconds ("2.5us", "1ms", or ns)."""
    return _parse(value, _TIME_NS, "time")


def size_bytes(value: float | int | str) -> int:
    """Returns a size in bytes ("64KiB", "8MB", or bytes)."""
    return int(round(_parse(value, _SIZE_BYTES, "size")))


def bandwidth(value: float | int | str) -> float:
    """Returns a bandwidth in bytes per ns, i.e. GB/s.

    Numbers are GB/s already. Strings: ``"400Gbps"`` and ``"400Gb/s"`` are
    bits (50 GB/s), ``"50GB/s"`` and ``"50GBps"`` bytes, ``"64GiB/s"`` binary.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected a bandwidth, got {value!r}")
    if isinstance(value, int | float):
        return float(value)
    match = _NUMBER_AND_UNIT.match(str(value))
    if match and not match.group(2):
        return float(match.group(1))
    unit_match = _BANDWIDTH_UNIT.match(match.group(2)) if match else None
    if match is None or unit_match is None:
        raise ValueError(
            f"cannot read {value!r} as a bandwidth; write e.g. 400Gbps, "
            "50GB/s or 64GiB/s"
        )
    prefix, binary, unit, _ = unit_match.groups()
    scale = (_BINARY_PREFIX if binary else _PREFIX)[prefix.lower()]
    bits = unit in ("b", "bit")
    return float(match.group(1)) * scale / (8.0 if bits else 1.0) / 1e9


def ns_per_byte(value: float | int | str) -> float:
    """Returns a per-byte time in ns: a number, or the inverse of a bandwidth.

    ``6`` is 6 ns per byte; ``"50GB/s"`` is 0.02 ns per byte.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    match = _NUMBER_AND_UNIT.match(str(value))
    if match and not match.group(2):
        return float(match.group(1))
    rate = bandwidth(value)
    if rate <= 0:
        raise ValueError(f"bandwidth {value!r} must be positive")
    return 1.0 / rate


def rate_per_second(value: float | int | str) -> float:
    """Returns events per second ("1kHz", or a number)."""
    return _parse(value, _RATE_PER_SECOND, "frequency")


def flops(value: float | int | str) -> float:
    """Returns FLOP per second ("989TFLOPS", or a number)."""
    return _parse(value, _FLOPS, "FLOP rate")
