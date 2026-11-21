"""Which device the analysis runs on: the CPU, or a GPU when there is one.

The option is ``accelerator`` (``--accelerator``), not ``device``, because
``devices`` already names the GPU ids of a trace. It is ``"auto"``, ``"cpu"``,
``"cuda"`` or ``"metal"``. The CPU path
is always there and is the fallback for everything: a device that is not
available, a build without GPU support, a driver or memory error. Asking for
a device that is missing is not an error; :func:`resolve` returns the CPU, says
why once, and the analysis goes on. Only a name that is no device is refused.

``auto`` takes a GPU only when the job is big enough to repay moving its data,
so small analyses never pay for a transfer.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import contextvars
import dataclasses
import warnings

import polars as pl

from cta import _optional

DEVICES = ("auto", "cpu", "cuda", "metal")

# `auto` moves a job to a GPU from this much work on (vertices x scenarios for
# the what-if solver). Measured with scripts/native/bench_gpu.py on an A100
# against 8 CPU threads: 0.1x at 5e6, 9.6x at 8e7, 3x to 44x from there up. The
# CPU wins below a few million because of the transfer and launch cost.
AUTO_MIN_WORK = 50_000_000

AUTO_BACKENDS = ("cuda",)

# `auto` runs a table query on the GPU engine from this many rows on. A
# group-by of 2000 keys on an A100 (scripts/native/bench_tables.py) was 1.0x
# the CPU at 1e6 rows, 1.7x at 1e7 and 1.2x at 5e7, so the gain is small:
# tables take the GPU on request (`--accelerator cuda`), and `auto` only when
# huge.
AUTO_MIN_ROWS = 50_000_000

_requested: contextvars.ContextVar[str] = contextvars.ContextVar(
    "cta_device", default="auto"
)
_warned: set[str] = set()


@dataclasses.dataclass(frozen=True, slots=True)
class Device:
    """The device a job runs on.

    Attributes:
        name: ``"cpu"``, ``"cuda"`` or ``"metal"``: where it runs.
        requested: What was asked for, ``"auto"`` included.
        note: Why the CPU was used when a GPU was asked for; empty if the
            request was met.
    """

    name: str
    requested: str
    note: str = ""


def check(name: str) -> str:
    """Returns an accelerator name in lower case, or raises if it is not one.

    Args:
        name: The device name from a flag, a manifest or the API.

    Raises:
        ValueError: If ``name`` is not one of :data:`DEVICES`.
    """
    value = str(name).strip().lower()
    if value not in DEVICES:
        raise ValueError(
            f"Unknown accelerator {name!r}; expected auto, cpu, cuda or metal"
        )
    return value


def backends() -> list[str]:
    """Returns the GPU backends this build has and finds a device for.

    A backend is listed only if it was compiled in *and* a device answers, so
    a build with CUDA on a machine without a GPU lists nothing.
    """
    try:
        from cta import _native

        found = getattr(_native, "gpu_backends", None)
        return list(found()) if found is not None else []
    except Exception:  # a broken driver must not break the analysis
        return []


def reset_warnings() -> None:
    """Forgets which fallbacks were already reported (for tests)."""
    _warned.clear()


def resolve(requested: str = "auto", *, work: int | None = None) -> Device:
    """Picks the device for a job.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"metal"``.
        work: The size of the job, so ``auto`` can tell whether a GPU repays
            the transfer; None means unknown, which stays on the CPU.

    Returns:
        The device. A GPU that was asked for but is missing gives the CPU
        with ``note`` set, and a warning the first time.

    Raises:
        ValueError: If ``requested`` is not a device name.
    """
    name = check(requested)
    if name == "cpu":
        return Device("cpu", name)
    present = backends()
    if name == "auto":
        eligible = [b for b in present if b in AUTO_BACKENDS]
        if eligible and work is not None and work >= AUTO_MIN_WORK:
            return Device(eligible[0], name)
        return Device("cpu", name)
    if name in present:
        return Device(name, name)
    note = f"{name} is not available here; running on the CPU"
    if name not in _warned:
        _warned.add(name)
        warnings.warn(note, stacklevel=2)
    return Device("cpu", name, note)


def polars_gpu_available() -> bool:
    """Returns whether Polars' GPU engine (``cudf-polars``) is installed."""
    return hasattr(pl, "GPUEngine") and _optional.available("cudf_polars")


def _collect_on_gpu(lazy: pl.LazyFrame) -> pl.DataFrame:
    """Runs a query on the Polars GPU engine.

    The engine itself falls back to the CPU for an operation it does not
    support (``raise_on_fail=False``); anything worse raises.
    """
    return lazy.collect(engine=pl.GPUEngine(raise_on_fail=False))


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, stacklevel=3)


def collect(lazy: pl.LazyFrame, *, rows: int | None = None) -> pl.DataFrame:
    """Runs a table query on the device the analysis asked for.

    ``cuda`` (or ``auto`` with at least :data:`AUTO_MIN_ROWS` rows and a CUDA
    device) uses Polars' GPU engine. A missing ``cudf-polars``, or a GPU
    error, gives the CPU result and a warning the first time. Polars has no
    Metal engine, so ``metal`` runs on the CPU.

    Args:
        lazy: The query.
        rows: The size of the main input, so ``auto`` can tell whether a GPU
            repays the transfer; None means unknown, which stays on the CPU.

    Returns:
        The query's result.
    """
    asked = requested()
    use_gpu = False
    if asked == "cuda" and "cuda" in backends():
        use_gpu = True
    elif asked == "auto" and "cuda" in backends():
        use_gpu = rows is not None and rows >= AUTO_MIN_ROWS
    if use_gpu:
        if not polars_gpu_available():
            if asked == "cuda":
                _warn_once(
                    "cudf-polars",
                    "cudf-polars is not installed; tables run on the CPU "
                    f"({_optional.install_hint('gpu')})",
                )
        else:
            try:
                return _collect_on_gpu(lazy)
            except Exception as exc:
                _warn_once(
                    f"gpu-table:{exc}",
                    f"the Polars GPU engine failed ({exc}); "
                    "running the query on the CPU",
                )
    return lazy.collect()


def requested() -> str:
    """Returns the device the current analysis asked for (``auto`` if unset)."""
    return _requested.get()


@contextlib.contextmanager
def use(name: str) -> Iterator[None]:
    """Sets the requested device for the code run inside the block.

    Args:
        name: The device name; see :func:`check`.
    """
    token = _requested.set(check(name))
    try:
        yield
    finally:
        _requested.reset(token)
