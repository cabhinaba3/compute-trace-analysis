"""A `.cta_graph` cache is built whole or not at all, and a bad one is rebuilt.

The what-if and out-of-core paths used to trust any file at the cache path, so
a build cut short by a crash (or a cache of an older format) failed every later
run until someone deleted it by hand.
"""

from __future__ import annotations

import pathlib

import pytest

from cta import models
from cta.analysis import streaming
from cta.analysis import whatif
from cta.ingest import discovery
from cta.synth import goal


@pytest.fixture
def goal_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "t.goal"
    goal.write_allreduce(path)
    return path


def _artifact(path: pathlib.Path) -> models.TraceArtifact:
    trial = discovery.resolve_trial_group(path).trials[0]
    artifact = streaming.pick_graph_artifact(trial)
    assert artifact is not None
    return artifact


def test_a_cache_cut_short_is_built_again(
    goal_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    cache_dir = tmp_path / "c"
    first = whatif.load_streaming_model(goal_file, cache_dir=cache_dir)
    cache = first.cache_path
    good_size = cache.stat().st_size
    cache.write_bytes(cache.read_bytes()[:128])  # a crash mid-build

    again = whatif.load_streaming_model(goal_file, cache_dir=cache_dir)

    assert again.num_nodes == first.num_nodes
    assert again.num_edges == first.num_edges
    assert cache.stat().st_size == good_size
    assert list(cache.parent.glob("*.tmp.*")) == []


def test_a_cache_that_is_not_a_cache_is_built_again(
    goal_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    artifact = _artifact(goal_file)
    cache = tmp_path / "x" / "g.cta_graph"
    cache.parent.mkdir()
    cache.write_bytes(b"not a graph cache" * 40)

    streaming.ensure_graph_cache(artifact, cache)

    assert (
        whatif.build_streaming_model_from_artifact(
            artifact, cache_path=cache
        ).num_nodes
        > 0
    )


def test_a_failed_build_leaves_no_file_behind(
    goal_file: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    artifact = _artifact(goal_file)
    cache = tmp_path / "g.cta_graph"

    with pytest.raises((ValueError, RuntimeError)):
        streaming.ensure_graph_cache(
            artifact, cache, goal_semantics="no-such-semantics"
        )

    assert not cache.exists()
    assert list(tmp_path.glob("g.cta_graph*")) == []


def test_a_cache_built_without_fail_on_cycle_is_checked_for_cycles(
    tmp_path: pathlib.Path,
) -> None:
    """A cycle is tolerated by a first run and refused by a later strict one."""
    path = tmp_path / "cycle.goal"
    # Two calcs that require each other.
    path.write_text(
        "num_ranks 1\nrank 0 {\n"
        "l1: calc 10\nl2: calc 10\nl1 requires l2\nl2 requires l1\n}\n"
    )
    artifact = _artifact(path)
    cache = tmp_path / "g.cta_graph"

    streaming.ensure_graph_cache(artifact, cache, fail_on_cycle=False)
    with pytest.raises(RuntimeError, match="cycle"):
        streaming.ensure_graph_cache(artifact, cache, fail_on_cycle=True)
