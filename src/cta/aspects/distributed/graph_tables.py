"""Reads the execution-graph tables that the distributed aspects share."""

from __future__ import annotations

import pathlib

import polars as pl

# Formats that provide an execution graph (nodes, edges and traffic tables),
# in the order ``read_graph_tables`` prefers them.
FORMATS = ("chakra_et", "chakra_host_et", "goal")


def read_graph_tables(
    table_paths: dict[str, pathlib.Path],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Reads the node, edge and traffic tables of one trial.

    Uses the first format in ``FORMATS`` that has a node table.

    Args:
        table_paths: Mapping of table key to Parquet path for one trial.

    Returns:
        The ``(nodes, edges, traffic)`` tables.

    Raises:
        KeyError: If no distributed format has a node table.
    """
    for trace_format in FORMATS:
        if f"{trace_format}.nodes" in table_paths:
            return (
                pl.read_parquet(table_paths[f"{trace_format}.nodes"]),
                pl.read_parquet(table_paths[f"{trace_format}.edges"]),
                pl.read_parquet(table_paths[f"{trace_format}.traffic"]),
            )
    raise KeyError("No distributed graph tables found in trial bundle")
