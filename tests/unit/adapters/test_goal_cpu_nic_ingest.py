"""GOAL's `cpu N` and `nic N` survive the streaming ingest.

The simulator reads them from the `nodes` table; the streaming sources used to
write zeros there.
"""

from __future__ import annotations

import pathlib

import polars as pl

from cta.adapters.goal import reader
from cta.ingest import discovery

TEXT = (
    "num_ranks 2\n"
    "rank 0 {\nl1: calc 100 cpu 1\nl2: send 8b to 1 tag 0 cpu 2 nic 1\n}\n"
    "rank 1 {\nl1: recv 8b from 0 tag 0 cpu 3 nic 1\n}\n"
)


def test_cpu_and_nic_reach_the_nodes_table(tmp_path: pathlib.Path) -> None:
    goal = tmp_path / "a.goal"
    goal.write_text(TEXT)
    artifact = discovery.resolve_trial_group(goal).trials[0].artifacts["goal"]

    tables = reader.GoalAdapter().ingest(artifact, tmp_path / "out")

    nodes = pl.read_parquet(tables["nodes"]).sort("rank", "node_id")
    assert nodes["cpu"].to_list() == [1, 2, 3]
    assert nodes["nic"].to_list() == [0, 1, 1]
