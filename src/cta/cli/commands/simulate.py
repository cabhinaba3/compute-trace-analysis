"""The ``simulate`` command group: run, compare and study GOAL/Chakra graphs."""

from __future__ import annotations

import json
import pathlib
from typing import Annotated, Any, TYPE_CHECKING

from rich import markup
from rich import table as rich_table
import typer

from cta.cli import output

if TYPE_CHECKING:
    import polars as pl

    from cta.sim import config as config_lib

simulate_app = typer.Typer(
    help="Simulate GOAL and Chakra graphs on a model of the machine.",
    no_args_is_help=True,
)

Model = Annotated[
    str | None,
    typer.Option("--model", "-m", help="loggops, alpha_beta or flow."),
]
Preset = Annotated[
    str | None,
    typer.Option("--preset", help="Machine preset (cta simulate settings)."),
]
ConfigFile = Annotated[
    pathlib.Path | None,
    typer.Option("--config", "-c", help="TOML settings file."),
]
Sets = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="key=value, e.g. loggops.L=2us."),
]
Level = Annotated[
    str | None,
    typer.Option("--level", help="Collectives: lowered or analytical."),
]
Groups = Annotated[
    pathlib.Path | None,
    typer.Option("--groups", help="JSON {pg_name: [ranks]}."),
]
CacheDir = Annotated[pathlib.Path, typer.Option("--cache-dir")]
OutDir = Annotated[
    pathlib.Path | None,
    typer.Option("--out-dir", "-o", help="Write tables here."),
]
Format = Annotated[str, typer.Option("--format", "-f", help=output.FORMAT_HELP)]
Top = Annotated[int, typer.Option("--top", help="Rows per table.")]


def _settings(
    model: str | None,
    preset: str | None,
    config_file: pathlib.Path | None,
    sets: list[str] | None,
    groups: pathlib.Path | None = None,
) -> config_lib.SimConfig:
    from cta.sim import config as config_lib

    extra = {"collectives.groups": str(groups)} if groups else {}
    return config_lib.load_config(
        config_file, preset=preset, overrides=sets or [], model=model
    ).with_overrides(extra)


def _print(frame: pl.DataFrame, title: str) -> None:
    """Prints a table, escaping names such as "x [lowered]" (not markup)."""
    table = rich_table.Table(title=markup.escape(title))
    for name, dtype in frame.schema.items():
        table.add_column(
            name, justify="right" if dtype.is_numeric() else "left"
        )
    for row in frame.iter_rows():
        table.add_row(
            *(
                "-"
                if v is None
                else f"{v:,.3f}"
                if isinstance(v, float)
                else markup.escape(str(v))
                for v in row
            )
        )
    output.console.print(table)


def _print_issues(issues: pl.DataFrame) -> None:
    for row in issues.iter_rows(named=True):
        if row["severity"] != "info":
            where = f"{row['run']}: " if "run" in row else ""
            message = f"{where}{row['check']} ({row['count']}): {row['detail']}"
            (
                output.print_error
                if row["severity"] == "error"
                else output.print_warning
            )(message)


def _guard(work: Any) -> Any:
    try:
        return work()
    except (ValueError, FileNotFoundError) as error:
        output.print_error(error)
        raise typer.Exit(code=2) from None


@simulate_app.command("run")
def run_command(
    source: Annotated[str, typer.Argument(help="GOAL or Chakra trace.")],
    model: Model = None,
    preset: Preset = None,
    config_file: ConfigFile = None,
    sets: Sets = None,
    level: Level = None,
    groups: Groups = None,
    trace_format: Annotated[str | None, typer.Option("--trace-format")] = None,
    out_dir: OutDir = None,
    trace: Annotated[
        pathlib.Path | None,
        typer.Option("--trace", help="Write Chrome trace JSON."),
    ] = None,
    top: Top = 10,
    output_format: Format = "rich",
    cache_dir: CacheDir = pathlib.Path(".cta_cache"),
) -> None:
    """Simulates a GOAL or Chakra trace and shows where the time went."""
    import polars as pl

    from cta.sim import engine
    from cta.sim import trace_export

    result = _guard(
        lambda: engine.simulate(
            source,
            _settings(model, preset, config_file, sets, groups),
            level=level,
            trace_format=trace_format,
            cache_dir=cache_dir,
        )
    )
    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    "run": result.name,
                    "summary": result.summary(),
                    "stats": result.stats,
                },
                default=str,
            )
        )
    else:
        stats = result.stats
        output.console.print(
            f"[bold]{markup.escape(result.name)}[/bold]: "
            f"{result.config['model']} model, {stats['ranks']} ranks, "
            f"{stats['ops']:,} ops, makespan "
            f"[bold]{result.makespan_ns / 1e6:,.3f} ms[/bold]"
            + ("" if result.completed else " [red](did not finish)[/red]")
        )
        _print(
            result.breakdown()
            .filter(pl.col("ns") > 0)
            .with_columns(ms=pl.col("ns") / 1e6)
            .drop("ns"),
            "Critical path",
        )
        _print(
            result.critical_path.group_by("name", "segment")
            .agg(ms=pl.col("duration_ns").sum() / 1e6)
            .sort("ms", descending=True)
            .head(top),
            "Heaviest ops on the critical path",
        )
        _print(
            result.ranks.sort("finish_ns", descending=True).head(top),
            "Slowest ranks",
        )
        if not result.links.is_empty():
            _print(
                result.links.sort("utilization", descending=True).head(top),
                "Busiest links",
            )
        _print_issues(result.issues)
    if out_dir:
        result.write(out_dir)
    if trace:
        trace_export.write_chrome_trace(result, trace)
    if not result.completed:
        raise typer.Exit(code=2)


@simulate_app.command("compare")
def compare_command(
    sources: Annotated[
        list[str], typer.Argument(help="Traces; the first is the baseline.")
    ],
    labels: Annotated[
        str | None, typer.Option("--labels", help="Comma-separated names.")
    ] = None,
    levels: Annotated[
        str, typer.Option("--levels", help="auto, both, lowered or analytical.")
    ] = "auto",
    model: Model = None,
    preset: Preset = None,
    config_file: ConfigFile = None,
    sets: Sets = None,
    groups: Groups = None,
    top: Top = 10,
    out_dir: OutDir = None,
    export: Annotated[
        str,
        typer.Option("--export", help="With --out-dir: markdown, json, ..."),
    ] = "",
    plot: Annotated[
        str, typer.Option("--plot", help="With --out-dir: sim_breakdown, ...")
    ] = "",
    output_format: Format = "rich",
    cache_dir: CacheDir = pathlib.Path(".cta_cache"),
) -> None:
    """Simulates GOAL and Chakra traces and compares them with the first."""
    import polars as pl

    from cta.export import writers
    from cta.sim import compare

    if len(sources) < 2:
        output.print_error("compare needs two or more traces")
        raise typer.Exit(code=2)
    if export:
        output.check_export_formats(out_dir, export)
    comparison = _guard(
        lambda: compare.compare(
            sources,
            _settings(model, preset, config_file, sets, groups),
            labels=output.split_comma_separated(labels),
            levels=levels,
            cache_dir=cache_dir,
        )
    )
    if output_format == "json":
        payload = {
            name: comparison.tables()[name].to_dicts()
            for name in ("overview", "attribution", "issues")
        }
        typer.echo(
            json.dumps(
                payload
                | {"traffic_mismatch_pct": comparison.traffic_mismatch()},
                default=str,
            )
        )
    else:
        _print(
            comparison.overview.select(
                "run",
                "level",
                "makespan_ms",
                "delta_ms",
                "delta_pct",
                "cp_compute_pct",
                "cp_comm_pct",
                "cp_contention_pct",
                "messages",
                "wire_gb",
                "completed",
            ),  # fmt: skip
            f"Simulated runs ({comparison.results[0].config['model']} model)",
        )
        _print(
            comparison.attribution.filter(pl.col("delta_ns") != 0),
            "Makespan gap by critical-path segment",
        )
        if mismatch := comparison.traffic_mismatch():
            _print(
                pl.DataFrame(
                    {
                        "run": list(mismatch),
                        "traffic_mismatch_pct": list(mismatch.values()),
                    }
                ),
                "Traffic",
            )
        if not comparison.phases.is_empty():
            _print(
                comparison.phases.sort(
                    pl.col("span_delta_ns").abs(), descending=True
                ).head(top),
                "Most divergent phases",
            )
        _print_issues(comparison.issues)
    if out_dir:
        comparison.write(out_dir)
        report = comparison.to_report()
        if export:
            writers.write_report(
                report, out_dir, output.split_comma_separated(export) or []
            )
        for kind in output.split_comma_separated(plot) or []:
            report.plot(kind=kind, path=out_dir / f"{kind}.png")


@simulate_app.command("sweep")
def sweep_command(
    source: Annotated[str, typer.Argument(help="GOAL or Chakra trace.")],
    param: Annotated[
        list[str] | None, typer.Option("--param", "-p", help="key=v1,v2,...")
    ] = None,
    trials: Annotated[
        int, typer.Option("--trials", help="Monte Carlo over this many seeds.")
    ] = 0,
    model: Model = None,
    preset: Preset = None,
    config_file: ConfigFile = None,
    sets: Sets = None,
    level: Level = None,
    groups: Groups = None,
    out_dir: OutDir = None,
    cache_dir: CacheDir = pathlib.Path(".cta_cache"),
) -> None:
    """Varies settings, or seeds, and tabulates the simulated makespan."""
    from cta.sim import graph as graph_lib
    from cta.sim import sweep

    def work() -> tuple[Any, Any]:
        if not param and trials <= 0:
            raise ValueError("give --param key=v1,v2,... or --trials N")
        settings = _settings(model, preset, config_file, sets, groups)
        graph = graph_lib.load_graph(
            source,
            groups=settings["collectives.groups"],
            world_size=settings["world_size"],
            cache_dir=cache_dir,
        )
        points = (
            sweep.sweep(graph, settings, sweep.parse_grid(param), level=level)
            if param
            else None
        )
        return points, sweep.monte_carlo(
            graph, settings, trials, level=level
        ) if trials > 0 else None

    points, monte = _guard(work)
    if points is not None:
        _print(points, "Sweep")
    if monte is not None:
        output.console.print(
            f"Monte Carlo over {trials} seeds: "
            + ", ".join(f"{k} {v:,.3f}" for k, v in monte[1].items())
        )
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        if points is not None:
            points.write_csv(out_dir / "sweep.csv")
        if monte is not None:
            monte[0].write_csv(out_dir / "monte_carlo.csv")


@simulate_app.command("collective")
def collective_command(
    coll_type: Annotated[str, typer.Argument(help="e.g. ALL_REDUCE.")],
    ranks: Annotated[int, typer.Option("--ranks", "-n")] = 8,
    size: Annotated[
        str, typer.Option("--bytes", "-b", help="e.g. 64MiB.")
    ] = "64MiB",
    algorithms: Annotated[
        str | None, typer.Option("--algorithms", "-a")
    ] = None,
    node_size: Annotated[
        int, typer.Option("--node-size", help="For hierarchical.")
    ] = 1,
    schedule: Annotated[
        str | None,
        typer.Option("--schedule", help="Show this algorithm's actions."),
    ] = None,
    model: Model = None,
    preset: Preset = None,
    config_file: ConfigFile = None,
    sets: Sets = None,
) -> None:
    """Compares collective algorithms for one collective, contention-free."""
    import polars as pl

    from cta.sim import collectives
    from cta.sim import units

    def work() -> tuple[Any, Any]:
        settings = _settings(model, preset, config_file, sets)
        nbytes = units.size_bytes(size)
        table = collectives.compare_algorithms(
            coll_type,
            ranks,
            nbytes,
            settings,
            algorithms=output.split_comma_separated(algorithms),
            node_size=node_size,
        )
        steps = schedule and collectives.schedule(
            coll_type, schedule, ranks, nbytes, node_size=node_size
        )
        return table, steps

    table, steps = _guard(work)
    _print(
        table.with_columns(time_us=pl.col("time_ns") / 1e3).drop("time_ns"),
        f"{coll_type} of {size} on {ranks} ranks",
    )
    if steps is not None:
        _print(
            steps.filter(pl.col("member") == 0).with_columns(
                pl.col("deps").cast(pl.List(pl.String)).list.join(",")
            ),
            f"{schedule}, member 0",
        )


@simulate_app.command("lower")
def lower_command(
    source: Annotated[str, typer.Argument(help="Chakra trace.")],
    destination: Annotated[
        pathlib.Path, typer.Argument(help="The .goal file to write.")
    ],
    model: Model = None,
    preset: Preset = None,
    config_file: ConfigFile = None,
    sets: Sets = None,
    groups: Groups = None,
    cache_dir: CacheDir = pathlib.Path(".cta_cache"),
) -> None:
    """Writes a trace as the GOAL schedule the simulator runs for it."""
    from cta.sim import engine
    from cta.sim import goal_export
    from cta.sim import graph as graph_lib

    def work() -> tuple[int, int]:
        settings = _settings(model, preset, config_file, sets, groups)
        graph = graph_lib.load_graph(
            source, groups=settings["collectives.groups"], cache_dir=cache_dir
        )
        columns = (
            engine.lower(graph, settings)[0]
            if graph.has_collectives
            else engine.prepared_columns(graph, settings)
        )
        return goal_export.write_goal(
            columns, graph.world_size, destination
        ), graph.world_size

    count, world = _guard(work)
    output.console.print(
        f"wrote {count:,} ops on {world} ranks to {destination}"
    )


@simulate_app.command("settings")
def settings_command() -> None:
    """Lists every setting and the machine presets."""
    import polars as pl

    from cta.sim import config as config_lib
    from cta.sim import presets

    _print(
        pl.DataFrame(
            [
                {"key": k, "default": repr(s.default), "meaning": s.help}
                for k, s in config_lib.SETTINGS.items()
            ]
        ),
        "Settings",
    )
    _print(
        pl.DataFrame(
            [
                {"preset": name, "description": p.description}
                for name, p in presets.PRESETS.items()
            ]
        ),
        "Presets",
    )


def register(app: typer.Typer) -> None:
    """Adds ``cta simulate`` to the application."""
    app.add_typer(simulate_app, name="simulate")
