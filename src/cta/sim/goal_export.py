"""Writes a simulator graph (lowered or not) as a GOAL text schedule.

With lowering this converts a Chakra trace to GOAL: every collective becomes
its algorithm's sends and recvs, exactly as the simulator runs them, so the
result can be fed to LogGOPSim or compared with another converter's output.
NOPs (a lowered collective's entry and exit) become ``calc 0``. Tags are
renumbered densely (GOAL tags are 32-bit; lowered tags are not), which keeps
which messages match.
"""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl

from cta.sim import graph as graph_lib


def write_goal(
    columns: dict[str, np.ndarray], world_size: int, path: str | pathlib.Path
) -> int:
    """Writes the graph's ops and edges as GOAL text; returns the op count.

    Args:
        columns: Engine arrays (``SimGraph.columns`` or ``engine.lower``'s).
        world_size: Number of ranks to declare.
        path: The .goal file to write.

    Raises:
        ValueError: The graph has collectives (lower it first) or more than
            2^32 distinct tags.
    """
    kind = columns["kind"]
    if np.any(kind == graph_lib.KIND_COLLECTIVE):
        raise ValueError("GOAL has no collectives; lower the graph first")
    _, dense_tags = np.unique(columns["tag"], return_inverse=True)
    if len(dense_tags) and dense_tags.max() >= 2**32:
        raise ValueError("more than 2^32 distinct message tags")
    ops = pl.DataFrame(
        {
            "rank": columns["rank"],
            "kind": kind,
            "cpu": columns["cpu"],
            "nic": columns["nic"],
            "duration": columns["duration"],
            "bytes": columns["bytes"],
            "peer": columns["peer"],
            "tag": dense_tags.astype(np.int64),
        }
    ).with_row_index("op")
    ops = ops.with_columns(
        (pl.int_range(pl.len()).over("rank") + 1).alias("label")
    )
    label = pl.lit("l") + pl.col("label").cast(pl.String)
    line = (
        pl.when(pl.col("kind") == graph_lib.KIND_SEND)
        .then(
            label
            + pl.lit(": send ")
            + pl.col("bytes").cast(pl.String)
            + pl.lit("b to ")
            + pl.col("peer").cast(pl.String)
            + pl.lit(" tag ")
            + pl.col("tag").cast(pl.String)
            + pl.lit(" cpu ")
            + pl.col("cpu").cast(pl.String)
            + pl.lit(" nic ")
            + pl.col("nic").cast(pl.String)
        )
        .when(pl.col("kind") == graph_lib.KIND_RECV)
        .then(
            label
            + pl.lit(": recv ")
            + pl.col("bytes").cast(pl.String)
            + pl.lit("b from ")
            + pl.col("peer").cast(pl.String)
            + pl.lit(" tag ")
            + pl.col("tag").cast(pl.String)
            + pl.lit(" cpu ")
            + pl.col("cpu").cast(pl.String)
            + pl.lit(" nic ")
            + pl.col("nic").cast(pl.String)
        )
        .otherwise(
            label
            + pl.lit(": calc ")
            + pl.when(pl.col("kind") == graph_lib.KIND_NOP)
            .then(0)
            .otherwise(pl.col("duration"))
            .cast(pl.String)
            + pl.lit(" cpu ")
            + pl.col("cpu").cast(pl.String)
        )
        .alias("line")
    )
    op_lines = ops.select("rank", "op", line)
    labels = ops.select("op", "label")
    edges = (
        pl.DataFrame(
            {
                "src": columns["edge_src"].astype(np.int64),
                "dst": columns["edge_dst"].astype(np.int64),
                "irequires": columns["edge_kind"] == 1,
            }
        )
        .join(labels.rename({"op": "src", "label": "src_label"}), on="src")
        .join(labels.rename({"op": "dst", "label": "dst_label"}), on="dst")
        .join(ops.select(pl.col("op").alias("dst"), "rank"), on="dst")
        .select(
            "rank",
            (
                pl.lit("l")
                + pl.col("dst_label").cast(pl.String)
                + pl.when(pl.col("irequires"))
                .then(pl.lit(" irequires l"))
                .otherwise(pl.lit(" requires l"))
                + pl.col("src_label").cast(pl.String)
            ).alias("line"),
        )
    )
    lines_by_rank = dict(
        op_lines.group_by("rank", maintain_order=True)
        .agg(pl.col("line"))
        .iter_rows()
    )
    deps_by_rank = dict(
        edges.group_by("rank", maintain_order=True)
        .agg(pl.col("line"))
        .iter_rows()
    )
    with pathlib.Path(path).open("w") as out:
        out.write(f"num_ranks {world_size}\n")
        for rank in range(world_size):
            out.write(f"\nrank {rank} {{\n")
            for text in lines_by_rank.get(rank, []):
                out.write(text + "\n")
            for text in deps_by_rank.get(rank, []):
                out.write(text + "\n")
            out.write("}\n")
    return ops.height
