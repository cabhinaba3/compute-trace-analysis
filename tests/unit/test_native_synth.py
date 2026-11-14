"""Tests of the C++ synthetic generators behind ``cta.synth``.

The golden hashes were taken from the Python generators the C++ code replaced,
so a mismatch means the output no longer matches those traces byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from cta.synth import chakra
from cta.synth import goal
from cta.synth import jtop
from cta.synth import kineto
from cta.synth import ncu
from cta.synth import tensorrt


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_small_outputs_match_the_python_generators_they_replaced(
    tmp_path: pathlib.Path,
) -> None:
    golden = {
        "kineto steps=3": "cad55229823c0662ec83942095f084bcfb4135064a4ee8f07b4f8ca67430b5ad",  # noqa: E501
        "kineto fp8 steps=7": "f03fb628563c7b51860e26f98ee62a7512ef4bb08518d44c108f6ca9f0a5f708",  # noqa: E501
        "goal ring 4x2": "27be72b89dfe7df57c71e3b3b5018485c98b31a6e26fc43f168b5f033b077270",  # noqa: E501
        "goal direct 3x2": "dbebb0cb239727ff810190ecb244196253d985320ab12f74f32e81a8d4325def",  # noqa: E501
        "chakra rank1": "02a0c31077f0cbf0b02e83682064648d5da3ca34c8eb9b4451b013c03ddef8ef",  # noqa: E501
        "chakra host-et": "6e8a9d29894a4445a850d32584a52c95180c4a5b517836e8e5b339f2329986ab",  # noqa: E501
        "ncu default": "6f4d750614dbe2ee8f7cc1701144e73db5b628125c3600771cc641d3b7205744",  # noqa: E501
        "jtop default": "8d8fa1f5201fa5a8c9ded82ec6f4a6311e2ff74c2d54e5243ffb719615649867",  # noqa: E501
        "trtexec timing": "1c7bac513d5ef30f9524d89a7d280e9d07165dd487d1aa28c0e899b57b4b5825",  # noqa: E501
    }
    actual = {
        "kineto steps=3": _sha256(
            kineto.write_kineto_trace(tmp_path / "k.json", steps=3)
        ),
        "kineto fp8 steps=7": _sha256(
            kineto.write_kineto_trace(
                tmp_path / "k8.json", steps=7, kernel_variant="fp8"
            )
        ),
        "goal ring 4x2": _sha256(
            goal.write_goal_allreduce(
                tmp_path / "g.goal", world_size=4, steps=2
            )
        ),
        "goal direct 3x2": _sha256(
            goal.write_goal_allreduce(
                tmp_path / "gd.goal",
                world_size=3,
                steps=2,
                algorithm="direct",
            )
        ),
        "chakra rank1": _sha256(
            chakra.write_chakra_et_run(
                tmp_path / "et",
                world_size=3,
                steps=2,
                straggler_rank=1,
                straggler_factor=2.5,
            )[1]
        ),
        "chakra host-et": _sha256(
            chakra.write_chakra_host_et(tmp_path / "h.json", steps=2)
        ),
        "ncu default": _sha256(ncu.write_ncu_csv(tmp_path / "n.csv")),
        "jtop default": _sha256(jtop.write_jtop_trace(tmp_path / "j.json")),
        "trtexec timing": _sha256(
            tensorrt.write_trtexec_timing(tmp_path / "t.json")
        ),
    }
    assert actual == golden


@pytest.mark.parametrize(
    ("world_size", "steps", "algorithm"),
    [(4, 3, "ring"), (3, 2, "direct"), (1, 3, "ring"), (3, 0, "ring")],
)
def test_goal_binary_equals_compiled_text(
    tmp_path: pathlib.Path, world_size: int, steps: int, algorithm: str
) -> None:
    text = goal.write_goal_allreduce(
        tmp_path / "s.goal",
        world_size=world_size,
        steps=steps,
        algorithm=algorithm,
    )
    compiled = goal.compile_goal_to_bin(text, tmp_path / "compiled.bin")
    direct = goal.write_goal_allreduce_bin(
        tmp_path / "direct.bin",
        world_size=world_size,
        steps=steps,
        algorithm=algorithm,
    )
    assert direct.read_bytes() == compiled.read_bytes()


def test_large_kineto_trace_is_well_formed_and_deterministic(
    tmp_path: pathlib.Path,
) -> None:
    first = tmp_path / "a.json"
    event_count = kineto.write_kineto_trace_large(
        first, target_bytes=2 << 20, seed=7, rank=3, world_size=8
    )
    second = tmp_path / "b.json"
    kineto.write_kineto_trace_large(
        second, target_bytes=2 << 20, seed=7, rank=3, world_size=8
    )
    assert first.read_bytes() == second.read_bytes()

    text = first.read_text()
    trace = json.loads(text)
    events = trace["traceEvents"]
    assert len(events) == event_count
    assert first.stat().st_size >= 2 << 20
    assert trace["distributedInfo"]["rank"] == 3
    assert text.rfind('"distributedInfo"') > text.rfind('"traceEvents"')

    launches = {
        event["args"]["correlation"]
        for event in events
        if event.get("cat") == "cuda_runtime" and "args" in event
    }
    gpu_events = [
        event
        for event in events
        if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    assert gpu_events
    assert all(e["args"]["correlation"] in launches for e in gpu_events)


def test_large_kineto_events_on_a_thread_nest_or_are_disjoint(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "t.json"
    kineto.write_kineto_trace_large(path, target_bytes=1 << 20, seed=2)
    events = [
        event
        for event in json.loads(path.read_text())["traceEvents"]
        if event.get("ph") == "X"
    ]
    by_thread: dict[tuple[int, int], list[dict[str, float]]] = {}
    for event in events:
        by_thread.setdefault((event["pid"], event["tid"]), []).append(event)
    for thread_events in by_thread.values():
        thread_events.sort(key=lambda e: (e["ts"], -e["dur"]))
        open_ends: list[float] = []
        for event in thread_events:
            end = event["ts"] + event["dur"]
            while open_ends and open_ends[-1] <= event["ts"] + 1e-6:
                open_ends.pop()
            assert not open_ends or end <= open_ends[-1] + 1e-6
            open_ends.append(end)


def test_generators_reject_arguments_of_the_wrong_type(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(TypeError):
        jtop.write_jtop_trace(
            tmp_path / "j.json",
            duration_s=2.5,  # type: ignore[arg-type]
        )
