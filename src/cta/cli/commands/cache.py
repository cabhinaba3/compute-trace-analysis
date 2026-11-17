"""The ``cache`` command group: inspect and clear the trace cache."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from cta.cli import output

cache_app = typer.Typer(
    help="Manage the content-addressed Parquet trace cache."
)


# The two commands below have no docstring on purpose: Typer would show it as
# their --help text, which is currently empty.


@cache_app.command("status")
def cache_status(
    cache_dir: Annotated[
        pathlib.Path, typer.Option("--cache-dir")
    ] = pathlib.Path(".cta_cache"),
) -> None:
    """Shows how many trace entries the cache holds and how large it is."""
    from cta.ingest import cache

    status = cache.CacheStore(cache_dir).status()
    output.console.print(
        f"Cache entries: [bold]{status['entries']}[/bold], "
        f"Size: [bold]{status['bytes'] / 1e6:.2f} MB[/bold]"
    )


@cache_app.command("clear")
def cache_clear(
    cache_dir: Annotated[
        pathlib.Path, typer.Option("--cache-dir")
    ] = pathlib.Path(".cta_cache"),
) -> None:
    """Deletes every cached trace entry."""
    from cta.ingest import cache

    removed = cache.CacheStore(cache_dir).clear()
    output.console.print(
        f"Cleared [bold]{removed}[/bold] cached trace entries."
    )
