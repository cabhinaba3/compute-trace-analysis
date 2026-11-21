"""The what-if solver on a GPU gives the CPU's answers, or the CPU gives them.

CUDA solves in 64-bit floats and must agree to 1e-9; Metal has no 64-bit
floats and agrees to 1e-5. A GPU that is not there, or that fails, falls back
to the CPU solve (`CTA_GPU_FAULT=1` makes it fail on purpose). Tests that need
a GPU are skipped on a machine or build without one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cta import _native as native
from cta.analysis import whatif

BACKENDS = (
    list(native.gpu_backends()) if hasattr(native, "gpu_backends") else []
)
TOLERANCE = {"cuda": 1e-9, "metal": 1e-5}


def _graph(
    n: int, seed: int, *, integer_weights: bool = False
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    src: list[int] = []
    dst: list[int] = []
    for v in range(1, n):
        for u in rng.choice(
            v, size=min(v, int(rng.integers(1, 4))), replace=False
        ):
            src.append(int(u))
            dst.append(v)
    weights = (
        rng.integers(1, 50, n).astype(np.float64)
        if integer_weights
        else rng.uniform(1.0, 1000.0, n)
    )
    is_comm = rng.random(n) < 0.3
    return {
        "plan": native.DagPlan(
            n, np.array(src, dtype=np.uint32), np.array(dst, dtype=np.uint32)
        ),
        "weights": weights,
        "is_comm": is_comm,
        "is_compute": ~is_comm,
        "size_bytes": rng.integers(0, 1 << 20, n).astype(np.int64),
        "group": rng.integers(0, 5, n).astype(np.uint32),
    }


def _solve(
    g: dict[str, Any], scale: np.ndarray, **options: Any
) -> dict[str, Any]:
    return g["plan"].solve(
        g["weights"],
        g["is_comm"],
        g["is_compute"],
        g["size_bytes"],
        g["group"],
        scale,
        0,
        0,
        **options,
    )


def _scenarios(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.0, 2.0, (count, 5))
    scale[0] = 1.0
    return scale


def test_the_build_says_which_gpu_backends_it_has() -> None:
    assert hasattr(native, "gpu_backends")
    assert set(native.gpu_backends()) <= {"cuda", "metal"}


def test_a_cpu_solve_says_so() -> None:
    g = _graph(300, 1)

    solved = _solve(g, _scenarios(8, 2), device="cpu")

    assert solved["device"] == "cpu"
    assert solved["device_note"] == ""


def test_the_default_device_is_the_cpu() -> None:
    g = _graph(300, 1)
    scale = _scenarios(8, 2)

    default = _solve(g, scale)
    cpu = _solve(g, scale, device="cpu")

    assert default["device"] == "cpu"
    np.testing.assert_array_equal(default["makespan"], cpu["makespan"])


@pytest.mark.parametrize("name", ["cuda", "metal"])
def test_a_gpu_that_is_not_there_gives_the_cpu_answer_and_says_why(
    name: str,
) -> None:
    if name in BACKENDS:
        pytest.skip(f"{name} is available here")
    g = _graph(300, 3)
    scale = _scenarios(16, 4)

    asked = _solve(g, scale, device=name)
    cpu = _solve(g, scale, device="cpu")

    assert asked["device"] == "cpu"
    assert name in asked["device_note"]
    np.testing.assert_array_equal(asked["makespan"], cpu["makespan"])


def test_an_unknown_device_is_refused() -> None:
    g = _graph(50, 1)

    with pytest.raises(ValueError, match="device"):
        _solve(g, _scenarios(2, 1), device="tpu")


@pytest.mark.parametrize("name", BACKENDS or ["none"])
def test_a_gpu_agrees_with_the_cpu(name: str) -> None:
    if name == "none":
        pytest.skip("no GPU backend in this build or machine")
    g = _graph(3000, 5)
    scale = _scenarios(700, 6)

    gpu = _solve(g, scale, device=name)
    cpu = _solve(g, scale, device="cpu")

    assert gpu["device"] == name, gpu["device_note"]
    tol = TOLERANCE[name]
    np.testing.assert_allclose(gpu["makespan"], cpu["makespan"], rtol=tol)
    np.testing.assert_allclose(
        gpu["critical_compute"], cpu["critical_compute"], rtol=tol, atol=1e-6
    )
    np.testing.assert_allclose(
        gpu["critical_comm"], cpu["critical_comm"], rtol=tol, atol=1e-6
    )
    np.testing.assert_allclose(
        gpu["critical_by_group"], cpu["critical_by_group"], rtol=tol, atol=1e-6
    )
    # Random float weights give no ties, so even fp32 finds the same path.
    np.testing.assert_array_equal(gpu["depth"], cpu["depth"])
    np.testing.assert_array_equal(
        gpu["critical_comm_bytes"], cpu["critical_comm_bytes"]
    )


def test_cuda_breaks_ties_like_the_cpu() -> None:
    """Integer weights give many equal finishes; depth, then id, decides."""
    if "cuda" not in BACKENDS:
        pytest.skip("no CUDA here")
    g = _graph(2000, 7, integer_weights=True)
    scale = np.ones((300, 5))

    gpu = _solve(g, scale, device="cuda")
    cpu = _solve(g, scale, device="cpu")

    np.testing.assert_array_equal(gpu["depth"], cpu["depth"])
    np.testing.assert_array_equal(gpu["makespan"], cpu["makespan"])
    np.testing.assert_array_equal(
        gpu["critical_comm_bytes"], cpu["critical_comm_bytes"]
    )


@pytest.mark.parametrize("name", BACKENDS or ["none"])
def test_a_gpu_that_fails_falls_back_to_the_cpu(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if name == "none":
        pytest.skip("no GPU backend in this build or machine")
    g = _graph(500, 8)
    scale = _scenarios(64, 9)
    monkeypatch.setenv("CTA_GPU_FAULT", "1")

    asked = _solve(g, scale, device=name)
    monkeypatch.delenv("CTA_GPU_FAULT")
    cpu = _solve(g, scale, device="cpu")

    assert asked["device"] == "cpu"
    assert "fault" in asked["device_note"]
    np.testing.assert_array_equal(asked["makespan"], cpu["makespan"])


@pytest.mark.parametrize("name", BACKENDS or ["none"])
def test_a_graph_too_big_for_the_gpu_budget_falls_back(name: str) -> None:
    if name == "none":
        pytest.skip("no GPU backend in this build or machine")
    g = _graph(400, 10)
    scale = _scenarios(32, 11)

    asked = _solve(g, scale, device=name, gpu_memory_bytes=1000)

    assert asked["device"] == "cpu"
    assert asked["device_note"]
    np.testing.assert_array_equal(
        asked["makespan"], _solve(g, scale, device="cpu")["makespan"]
    )


def test_run_scenarios_reports_the_device(
    tmp_path: Any,
) -> None:
    from cta.synth import goal

    path = tmp_path / "t.goal"
    goal.write_allreduce(path)
    model = whatif.load_model(path, cache_dir=tmp_path / "c")
    scenarios = [whatif.parse_scenario("comm:2x")]

    cpu = whatif.run_scenarios(model, scenarios, accelerator="cpu")
    asked = whatif.run_scenarios(model, scenarios, accelerator="cuda")

    assert cpu.device == "cpu"
    assert asked.device in ("cpu", "cuda")
    np.testing.assert_allclose(
        asked.table["makespan_ns"].to_numpy(),
        cpu.table["makespan_ns"].to_numpy(),
        rtol=1e-9,
    )
