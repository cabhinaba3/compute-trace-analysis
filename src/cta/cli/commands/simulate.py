"""The ``simulate`` command group: run, compare and study GOAL/Chakra graphs.

``cta simulate run`` simulates one trace, ``compare`` simulates several (GOAL
and Chakra in any mix) and compares them with the first, ``sweep`` varies
settings, ``collective`` sets collective algorithms side by side, ``lower``
writes a Chakra trace as the GOAL schedule the simulator runs, and
``presets`` and ``settings`` list what can be configured.
"""

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
    from cta.sim import result as result_lib

simulate_app = typer.Typer(
    help="Simulate GOAL and Chakra execution graphs on a model of the "
    "machine, and compare them.",
    no_args_is_help=True,
)

ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model", "-m", help="Network model: loggops, alpha_beta or flow."
    ),
]
PresetOption = Annotated[
    str | None,
    typer.Option("--preset", help="Machine preset (cta simulate presets)."),
]
ConfigOption = Annotated[
    pathlib.Path | None,
    typer.Option("--config", "-c", help="TOML file of settings."),
]
SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="Override a setting, e.g. loggops.L=2us (cta simulate settings).",
    ),
]
LevelOption = Annotated[
    str | None,
    typer.Option(
        "--level", help="Collectives: lowered (messages) or analytical."
    ),
]
GroupsOption = Annotated[
    pathlib.Path | None,
    typer.Option("--groups", help="JSON file {pg_name: [ranks]}."),
]
CacheOption = Annotated[pathlib.Path, typer.Option("--cache-dir")]
OutDirOption = Annotated[
    pathlib.Path | None,
    typer.Option("--out-dir", "-o", help="Write tables (Parquet) here."),
]
FormatOption = Annotated[
    str, typer.Option("--format", "-f", help=output.FORMAT_HELP)
]


def _settings(
    model: str | None,
    preset: str | None,
    config_file: pathlib.Path | None,
    sets: list[str] | None,
    groups: pathlib.Path | None,
) -> config_lib.SimConfig:
    from cta.sim import config as config_lib

    settings = config_lib.load_config(
        config_file, preset=preset, overrides=sets or [], model=model
    )
    if groups is not None:
        settings = settings.with_overrides({"collectives.groups": str(groups)})
    return settings


def _fail(error: Exception) -> typer.Exit:
    output.print_error(error)
    return typer.Exit(code=2)


def _ms(ns: float | None) -> str:
    return "-" if ns is None else f"{ns / 1e6:,.3f}"


def _print_frame(
    frame: pl.DataFrame, title: str, formats: dict[str, str] | None = None
) -> None:
    """Prints a DataFrame as a Rich table, numbers right-aligned."""
    formats = formats or {}
    # Run names such as "chakra [lowered]" would otherwise read as markup.
    table = rich_table.Table(title=markup.escape(title))
    for name, dtype in frame.schema.items():
        table.add_column(
            name, justify="right" if dtype.is_numeric() else "left"
        )
    for row in frame.iter_rows(named=True):
        cells = []
        for name, value in row.items():
            if value is None:
                cells.append("-")
            elif name in formats:
                cells.append(format(value, formats[name]))
            elif isinstance(value, float):
                cells.append(f"{value:,.3f}")
            else:
                cells.append(markup.escape(str(value)))
        table.add_row(*cells)
    output.console.print(table)


def _print_issues(issues: pl.DataFrame) -> None:
    import polars as pl

    shown = issues.filter(pl.col("severity") != "info")
    for row in shown.iter_rows(named=True):
        message = f"{row['check']} ({row['count']}): {row['detail']}"
        if "run" in row:
            message = f"{row['run']}: {message}"
        if row["severity"] == "error":
            output.print_error(message)
        else:
            output.print_warning(message)


def _print_run(result: result_lib.SimResult, top: int) -> None:
    import polars as pl

    stats = result.stats
    output.console.print(
        f"[bold]{markup.escape(result.name)}[/bold]  {result.trace_format}, "
        f"{result.config.model} model, {stats['ranks']} ranks, "
        f"{stats['ops']:,} ops, {stats['messages']:,} messages, "
        f"{stats['events']:,} events in {stats['wall_seconds']:.2f} s"
    )
    output.console.print(
        f"makespan [bold]{_ms(result.makespan_ns)} ms[/bold]"
        + (
            ""
            if result.completed
            else "  [bold red](did not finish)[/bold red]"
        )
    )
    breakdown = result.breakdown().filter(pl.col("ns") > 0)
    _print_frame(
        breakdown.select(
            "segment",
            (pl.col("ns") / 1e6).alias("ms"),
            pl.col("pct").alias("%"),
        ),
        "Critical path",
        {"%": ".1f"},
    )
    heavy = (
        result.critical_path.group_by("name", "segment")
        .agg(pl.col("duration_ns").sum(), pl.len().alias("segments"))
        .sort("duration_ns", descending=True)
        .head(top)
        .select(
            "name",
            "segment",
            "segments",
            (pl.col("duration_ns") / 1e6).alias("ms"),
        )
    )
    _print_frame(heavy, f"Heaviest ops on the critical path (top {top})")
    slowest = result.ranks.sort("finish_ns", descending=True).head(min(top, 10))
    _print_frame(
        slowest.select(
            "rank",
            (pl.col("finish_ns") / 1e6).alias("finish_ms"),
            (pl.col("compute_ns") / 1e6).alias("compute_ms"),
            (pl.col("overhead_ns") / 1e6).alias("overhead_ms"),
            (pl.col("wait_ns") / 1e6).alias("wait_ms"),
            (pl.col("sync_ns") / 1e6).alias("sync_ms"),
            pl.col("utilization"),
        ),
        "Slowest ranks",
    )
    if not result.links.is_empty():
        busiest = result.links.sort("utilization", descending=True).head(
            min(top, 10)
        )
        _print_frame(
            busiest.select(
                "link",
                "kind",
                "src",
                "dst",
                "capacity_gbs",
                "bytes",
                "utilization",
                "max_flows",
            ),
            "Busiest links",
        )


@simulate_app.command("run")
def run_command(
    source: Annotated[str, typer.Argument(help="GOAL or Chakra trace.")],
    model: ModelOption = None,
    preset: PresetOption = None,
    config_file: ConfigOption = None,
    sets: SetOption = None,
    level: LevelOption = None,
    groups: GroupsOption = None,
    trace_format: Annotated[
        str | None,
        typer.Option("--trace-format", help="goal, chakra or chakra_host."),
    ] = None,
    label: Annotated[str | None, typer.Option("--label")] = None,
    out_dir: OutDirOption = None,
    csv: Annotated[bool, typer.Option("--csv", help="Write CSV.")] = False,
    trace: Annotated[
        pathlib.Path | None,
        typer.Option("--trace", help="Write a Chrome/Perfetto trace JSON."),
    ] = None,
    top: Annotated[int, typer.Option("--top", help="Rows per table.")] = 10,
    output_format: FormatOption = "rich",
    cache_dir: CacheOption = pathlib.Path(".cta_cache"),
) -> None:
    """Simulates one GOAL or Chakra trace and shows where the time went."""
    from cta.sim import engine
    from cta.sim import trace_export

    try:
        settings = _settings(model, preset, config_file, sets, groups)
        result = engine.simulate(
            source,
            settings,
            level=level,
            trace_format=trace_format,
            label=label,
            cache_dir=cache_dir,
        )
    except (ValueError, FileNotFoundError) as error:
        raise _fail(error) from None
    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    "run": result.name,
                    "summary": result.summary(),
                    "stats": result.stats,
                    "issues": result.issues.to_dicts(),
                },
                indent=2,
                default=str,
            )
        )
    else:
        _print_run(result, top)
        _print_issues(result.issues)
    if out_dir is not None:
        result.write(out_dir, csv=csv)
    if trace is not None:
        trace_export.write_chrome_trace(result, trace)
    if not result.completed:
        raise typer.Exit(code=2)


@simulate_app.command("compare")
def compare_command(
    sources: Annotated[
        list[str],
        typer.Argument(help="Two or more traces; the first is the baseline."),
    ],
    labels: Annotated[
        str | None, typer.Option("--labels", help="Comma-separated names.")
    ] = None,
    levels: Annotated[
        str,
        typer.Option(
            "--levels",
            help="auto (Chakra both lowered and analytical when compared with "
            "GOAL), both, lowered or analytical.",
        ),
    ] = "auto",
    model: ModelOption = None,
    preset: PresetOption = None,
    config_file: ConfigOption = None,
    sets: SetOption = None,
    groups: GroupsOption = None,
    top: Annotated[int, typer.Option("--top", help="Rows per table.")] = 10,
    out_dir: OutDirOption = None,
    export: Annotated[
        str,
        typer.Option(
            "--export", help="With --out-dir: markdown, latex, csv, json, ..."
        ),
    ] = "",
    plot: Annotated[
        str,
        typer.Option(
            "--plot",
            help="With --out-dir: plot kinds, e.g. sim_breakdown,sim_ranks.",
        ),
    ] = "",
    output_format: FormatOption = "rich",
    cache_dir: CacheOption = pathlib.Path(".cta_cache"),
) -> None:
    """Simulates traces and compares them: GOAL, Chakra, or both."""
    import polars as pl

    from cta.export import writers
    from cta.sim import compare

    if len(sources) < 2:
        raise _fail(ValueError("compare needs at least two traces"))
    if export:
        output.check_export_formats(out_dir, export)
    try:
        settings = _settings(model, preset, config_file, sets, groups)
        comparison = compare.compare(
            sources,
            settings,
            labels=output.split_comma_separated(labels),
            levels=levels,
            cache_dir=cache_dir,
            top=top,
        )
    except (ValueError, FileNotFoundError) as error:
        raise _fail(error) from None
    if output_format == "json":
        typer.echo(
            json.dumps(compare.summarize(comparison), indent=2, default=str)
        )
    else:
        _print_frame(
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
            ),
            f"Simulated runs ({comparison.baseline.config.model} model; "
            f"baseline {comparison.baseline.name})",
            {"delta_pct": "+.2f"},
        )
        if not comparison.attribution.is_empty():
            _print_frame(
                comparison.attribution.filter(pl.col("delta_ns") != 0).select(
                    "run",
                    "relation",
                    "segment",
                    (pl.col("baseline_ns") / 1e6).alias("baseline_ms"),
                    (pl.col("run_ns") / 1e6).alias("run_ms"),
                    (pl.col("delta_ns") / 1e6).alias("delta_ms"),
                    pl.col("share_of_gap_pct").alias("% of gap"),
                ),
                "Makespan gap by critical-path segment",
                {"% of gap": "+.1f"},
            )
        mismatch = compare.traffic_mismatch(comparison)
        if mismatch:
            _print_frame(
                pl.DataFrame(
                    {
                        "run": list(mismatch),
                        "traffic_mismatch_pct": list(mismatch.values()),
                    }
                ),
                "Bytes per (src, dst) pair differing from the baseline",
            )
        if not comparison.phases.is_empty():
            summary = comparison.phases.group_by(
                "run", maintain_order=True
            ).agg(
                pl.len().alias("phases"),
                (pl.col("span_delta_ns").abs().mean() / 1e3).alias(
                    "mean_|span_delta|_us"
                ),
                (pl.col("span_delta_ns").sum() / 1e6).alias(
                    "sum_span_delta_ms"
                ),
                (pl.col("bytes_delta") != 0)
                .sum()
                .alias("phases_with_byte_delta"),
            )
            _print_frame(summary, "Phases (compute anchor to compute anchor)")
            worst = (
                comparison.phases.sort(
                    pl.col("span_delta_ns").abs(), descending=True
                )
                .head(top)
                .select(
                    "run",
                    "rank",
                    "phase",
                    "anchor",
                    (pl.col("baseline_span_ns") / 1e3).alias("base_us"),
                    (pl.col("run_span_ns") / 1e3).alias("run_us"),
                    (pl.col("span_delta_ns") / 1e3).alias("delta_us"),
                    "bytes_delta",
                    "baseline_collectives",
                    "run_collectives",
                )
            )
            _print_frame(worst, f"Most divergent phases (top {top})")
        _print_issues(comparison.issues)
    if out_dir is not None:
        comparison.write(out_dir)
        report = comparison.to_report()
        if export:
            writers.write_report(
                report, out_dir, output.split_comma_separated(export) or []
            )
        for kind in output.split_comma_separated(plot) or []:
            try:
                report.plot(kind=kind, path=out_dir / f"{kind}.png")
            except ValueError as error:
                output.print_warning(f"plot {kind}: {error}")


@simulate_app.command("sweep")
def sweep_command(
    source: Annotated[str, typer.Argument(help="GOAL or Chakra trace.")],
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            "-p",
            help="Setting and values to sweep, e.g. loggops.G=0.01,0.02,0.04.",
        ),
    ] = None,
    trials: Annotated[
        int,
        typer.Option(
            "--trials", help="Monte Carlo: rerun with this many seeds."
        ),
    ] = 0,
    model: ModelOption = None,
    preset: PresetOption = None,
    config_file: ConfigOption = None,
    sets: SetOption = None,
    level: LevelOption = None,
    groups: GroupsOption = None,
    out_dir: OutDirOption = None,
    output_format: FormatOption = "rich",
    cache_dir: CacheOption = pathlib.Path(".cta_cache"),
) -> None:
    """Varies settings (and seeds) and tabulates the simulated makespan."""
    from cta.sim import graph as graph_lib
    from cta.sim import sweep

    if not param and trials <= 0:
        raise _fail(ValueError("give --param key=v1,v2,... or --trials N"))
    try:
        settings = _settings(model, preset, config_file, sets, groups)
        graph = graph_lib.load_graph(
            source,
            groups=settings.collectives.groups,
            world_size=settings.world_size,
            cache_dir=cache_dir,
        )
        points = (
            sweep.sweep(graph, settings, sweep.parse_grid(param), level=level)
            if param
            else None
        )
        monte = (
            sweep.monte_carlo(graph, settings, trials, level=level)
            if trials > 0
            else None
        )
    except (ValueError, FileNotFoundError) as error:
        raise _fail(error) from None
    payload: dict[str, Any] = {}
    if points is not None:
        payload["sweep"] = points.to_dicts()
    if monte is not None:
        payload["monte_carlo"] = {
            "trials": monte[0].to_dicts(),
            "makespan": monte[1],
        }
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        if points is not None:
            _print_frame(points, f"Sweep of {graph.label}")
        if monte is not None:
            output.console.print(
                f"Monte Carlo over {trials} seeds: "
                + ", ".join(f"{k} {v:,.3f}" for k, v in monte[1].items())
            )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if points is not None:
            points.write_csv(out_dir / "sweep.csv")
        if monte is not None:
            monte[0].write_csv(out_dir / "monte_carlo.csv")
            (out_dir / "monte_carlo.json").write_text(
                json.dumps(monte[1], indent=2)
            )


@simulate_app.command("collective")
def collective_command(
    coll_type: Annotated[
        str, typer.Argument(help="e.g. ALL_REDUCE, ALL_TO_ALL.")
    ],
    ranks: Annotated[
        int, typer.Option("--ranks", "-n", help="Group size.")
    ] = 8,
    size: Annotated[
        str, typer.Option("--bytes", "-b", help="Size, e.g. 64MiB.")
    ] = "64MiB",
    algorithms: Annotated[
        str | None,
        typer.Option(
            "--algorithms", "-a", help="Comma-separated; default all."
        ),
    ] = None,
    node_size: Annotated[
        int, typer.Option("--node-size", help="Ranks per node (hierarchical).")
    ] = 1,
    schedule: Annotated[
        str | None,
        typer.Option(
            "--schedule", help="Also print this algorithm's schedule."
        ),
    ] = None,
    member: Annotated[
        int, typer.Option("--member", help="Member to show.")
    ] = 0,
    model: ModelOption = None,
    preset: PresetOption = None,
    config_file: ConfigOption = None,
    sets: SetOption = None,
    output_format: FormatOption = "rich",
) -> None:
    """Compares collective algorithms analytically for one collective."""
    import polars as pl

    from cta.sim import collectives
    from cta.sim import units

    try:
        settings = _settings(model, preset, config_file, sets, None)
        nbytes = units.size_bytes(size)
        table = collectives.compare_algorithms(
            coll_type,
            ranks,
            nbytes,
            settings,
            algorithms=output.split_comma_separated(algorithms),
            node_size=node_size,
        )
        steps = (
            collectives.schedule(
                coll_type, schedule, ranks, nbytes, node_size=node_size
            )
            if schedule
            else None
        )
    except ValueError as error:
        raise _fail(error) from None
    if output_format == "json":
        typer.echo(json.dumps(table.to_dicts(), indent=2))
        return
    _print_frame(
        table.with_columns((pl.col("time_ns") / 1e3).alias("time_us")).drop(
            "time_ns"
        ),
        f"{coll_type.upper()} of {nbytes:,} bytes on {ranks} ranks "
        f"({settings.model} model)",
    )
    if steps is not None:
        _print_frame(
            steps.filter(pl.col("member") == member).with_columns(
                pl.col("deps")
                .list.eval(pl.element().cast(pl.String))
                .list.join(",")
            ),
            f"{schedule}: actions of member {member}",
        )


@simulate_app.command("lower")
def lower_command(
    source: Annotated[str, typer.Argument(help="Chakra (or GOAL) trace.")],
    destination: Annotated[
        pathlib.Path, typer.Argument(help="The .goal file to write.")
    ],
    model: ModelOption = None,
    preset: PresetOption = None,
    config_file: ConfigOption = None,
    sets: SetOption = None,
    groups: GroupsOption = None,
    cache_dir: CacheOption = pathlib.Path(".cta_cache"),
) -> None:
    """Writes a trace as the GOAL schedule the simulator runs (lowered)."""
    from cta.sim import engine
    from cta.sim import goal_export
    from cta.sim import graph as graph_lib

    try:
        settings = _settings(model, preset, config_file, sets, groups)
        graph = graph_lib.load_graph(
            source,
            groups=settings.collectives.groups,
            world_size=settings.world_size,
            cache_dir=cache_dir,
        )
        columns = (
            engine.lower(graph, settings)[0]
            if graph.has_collectives
            else engine.prepared_columns(graph, settings)
        )
        count = goal_export.write_goal(columns, graph.world_size, destination)
    except (ValueError, FileNotFoundError) as error:
        raise _fail(error) from None
    output.console.print(
        f"wrote {count:,} ops on {graph.world_size} ranks to {destination}"
    )


@simulate_app.command("presets")
def presets_command() -> None:
    """Lists the machine presets."""
    from cta.sim import presets

    table = rich_table.Table(title="Simulation presets")
    for column in (
        "name",
        "GB/s",
        "latency",
        "ranks/node",
        "intra GB/s",
        "description",
    ):
        table.add_column(column)
    for p in presets.PRESETS.values():
        table.add_row(
            p.name,
            f"{p.bandwidth:g}",
            f"{p.latency:g} ns",
            str(p.ranks_per_node),
            f"{p.intra_bandwidth:g}" if p.intra_bandwidth else "-",
            p.description,
        )
    output.console.print(table)


@simulate_app.command("settings")
def settings_command() -> None:
    """Lists every setting with its default (for --set and TOML files)."""
    from cta.sim import config as config_lib

    table = rich_table.Table(title="Simulation settings")
    for column in ("key", "default", "meaning"):
        table.add_column(column)
    for key, default, help_text in config_lib.describe():
        table.add_row(key, default, help_text)
    output.console.print(table)


def register(app: typer.Typer) -> None:
    """Adds ``cta simulate`` to the application."""
    app.add_typer(simulate_app, name="simulate")
