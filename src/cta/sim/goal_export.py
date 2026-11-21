"""Writes an engine graph as a GOAL text schedule.

With lowering this converts a Chakra trace to the GOAL schedule the simulator
runs. NOPs become ``calc 0``; tags are renumbered densely, as GOAL tags have
32 bits and lowered tags 64.
"""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl

from cta.sim import graph as graph_lib


def write_goal(
    columns: dict[str, np.ndarray], world_size: int, path: str | pathlib.Path
) -> int:
    """Writes a graph without collectives; returns its op count."""
    if np.any(columns["kind"] == graph_lib.KIND_COLLECTIVE):
        raise ValueError("GOAL has no collectives; lower the graph first")
    ops = (
        pl.DataFrame(
            {
                **{
                    c: columns[c]
                    for c in (
                        "rank",
                        "kind",
                        "cpu",
                        "nic",
                        "duration",
                        "bytes",
                        "peer",
                    )
                },
                "tag": np.unique(columns["tag"], return_inverse=True)[1],
            }
        )
        .with_row_index("op")
        .with_columns(label=pl.int_range(1, pl.len() + 1).over("rank"))
    )
    kind = pl.col("kind")
    ops = ops.with_columns(
        line=pl.when(kind.is_in([graph_lib.KIND_SEND, graph_lib.KIND_RECV]))
        .then(
            pl.format(
                "l{}: {} {}b {} {} tag {} cpu {} nic {}",
                "label",
                pl.when(kind == graph_lib.KIND_SEND)
                .then(pl.lit("send"))
                .otherwise(pl.lit("recv")),
                "bytes",
                pl.when(kind == graph_lib.KIND_SEND)
                .then(pl.lit("to"))
                .otherwise(pl.lit("from")),
                "peer",
                "tag",
                "cpu",
                "nic",
            )
        )
        .otherwise(
            pl.format(
                "l{}: calc {} cpu {}",
                "label",
                pl.when(kind == graph_lib.KIND_NOP)
                .then(0)
                .otherwise("duration"),
                "cpu",
            )
        )
    )
    labels = ops.select("op", "label", "rank")
    deps = (
        pl.DataFrame(
            {
                "src": columns["edge_src"],
                "dst": columns["edge_dst"],
                "kind": columns["edge_kind"],
            }
        )
        .join(
            labels.rename({"op": "src", "label": "from"}).drop("rank"), on="src"
        )
        .join(labels.rename({"op": "dst", "label": "to"}), on="dst")
        .select(
            "rank",
            line=pl.format(
                "l{} {} l{}",
                "to",
                pl.when(pl.col("kind") == 1)
                .then(pl.lit("irequires"))
                .otherwise(pl.lit("requires")),
                "from",
            ),
        )
    )
    lines = dict(
        pl.concat([ops.select("rank", "line"), deps])
        .group_by("rank", maintain_order=True)
        .agg("line")
        .iter_rows()
    )
    with pathlib.Path(path).open("w") as out:
        out.write(f"num_ranks {world_size}\n")
        for rank in range(world_size):
            out.write(
                f"\nrank {rank} {{\n"
                + "".join(f"{line}\n" for line in lines.get(rank, []))
                + "}\n"
            )
    return ops.height
