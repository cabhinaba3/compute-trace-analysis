"""The ``plot`` command."""

from __future__ import annotations

import pathlib
from typing import Annotated, TYPE_CHECKING

from rich import table
import typer

from cta.cli import output

if TYPE_CHECKING:
    from cta.viz import spec as viz_spec


def _parse_options(pairs: list[str] | None) -> dict[str, str]:
    """Parses repeated ``--opt key=value`` values into a dict.

    Args:
        pairs: The raw ``key=value`` strings; ``None`` means none were given.

    Returns:
        The values by stripped key; later pairs override earlier ones.

    Raises:
        typer.BadParameter: If a pair has no ``=`` or an empty key.
    """
    options: dict[str, str] = {}
    for pair in pairs or []:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            raise typer.BadParameter(
                f"expected key=value, got {pair!r}", param_hint="--opt"
            )
        options[key.strip()] = value
    return options


def _print_kinds() -> None:
    """Prints the table of plot kinds with their data needs and options."""
    from cta.viz import spec as viz_spec

    kinds_table = table.Table(
        title=(
            "Plot kinds (cta plot --kind NAME; options via --opt key=value or "
            "a --spec file)"
        ),
        show_lines=False,
    )
    for column in ("kind", "group", "needs", "options"):
        kinds_table.add_column(column, overflow="fold")
    for kind_info in viz_spec.kind_catalog():
        alias = (
            f" ({', '.join(kind_info.aliases)})" if kind_info.aliases else ""
        )
        option_defaults = ", ".join(
            f"{option_name}={option.default}"
            if option.default is not None
            else option_name
            for option_name, option in kind_info.options.items()
        )
        kinds_table.add_row(
            kind_info.name + alias,
            kind_info.group,
            kind_info.needs,
            option_defaults,
        )
    output.console.print(kinds_table)
    output.console.print(
        "[dim]Every kind also accepts: "
        + ", ".join(viz_spec.COMMON_OPTIONS)
        + "[/dim]"
    )


def _with_shared_options(
    plot_spec: viz_spec.PlotSpec, shared_options: dict[str, str]
) -> viz_spec.PlotSpec:
    """Returns a copy of a plot spec with the shared options it accepts.

    Args:
        plot_spec: The plot spec to extend.
        shared_options: Options given once for every plot (``--opt``,
            ``--theme``, ...); those the spec's kind does not declare are
            left out.

    Returns:
        A new spec whose own options override the shared ones.
    """
    from cta.viz import spec as viz_spec

    declared = viz_spec.get_kind(plot_spec.kind).all_options()
    accepted_shared_options = {
        key: value for key, value in shared_options.items() if key in declared
    }
    return viz_spec.PlotSpec(
        plot_spec.kind,
        {**accepted_shared_options, **plot_spec.options},
        plot_spec.name,
    )


def plot_command(
    traces: Annotated[
        list[str] | None,
        typer.Argument(
            help="One (single-trace) or more (comparative) trace files/bundles."
        ),
    ] = None,
    manifest: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--manifest",
            "-m",
            help=(
                "Declarative YAML/TOML sweep manifest (its `plots:` block is "
                "used when no --kind/--spec is given)."
            ),
        ),
    ] = None,
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            "-k",
            help=(
                "Comma-separated plot kinds, or 'all' for every applicable "
                "kind (see --list-kinds). Default: breakdown."
            ),
        ),
    ] = None,
    spec: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--spec",
            help="YAML file of plot specs (a list, or a `plots:` list).",
        ),
    ] = None,
    sql: Annotated[
        str | None,
        typer.Option(
            "--sql",
            help=(
                "DuckDB SQL over the named tables (see --list-sources); drawn "
                "with --mark/--x/--y/--color/--facet."
            ),
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            help=(
                "Raw table (kernels, steps, ...) or "
                "report:<aspect>[.<table>] for a custom plot."
            ),
        ),
    ] = None,
    mark: Annotated[
        str | None,
        typer.Option(
            "--mark",
            help=(
                "Custom plot mark: line, step, bar, barh, scatter, box, hist, "
                "heatmap."
            ),
        ),
    ] = None,
    x: Annotated[
        str | None, typer.Option("--x", help="Custom plot: column on x.")
    ] = None,
    y: Annotated[
        str | None, typer.Option("--y", help="Custom plot: column on y.")
    ] = None,
    color: Annotated[
        str | None,
        typer.Option(
            "--color", help="Custom plot: column that splits coloured series."
        ),
    ] = None,
    facet: Annotated[
        str | None,
        typer.Option("--facet", help="Custom plot: column that splits panels."),
    ] = None,
    opt: Annotated[
        list[str] | None,
        typer.Option(
            "--opt",
            "-O",
            help=(
                "Plot option key=value, repeatable (e.g. -O top_k=20 "
                "-O steps=2:10 -O where=\"category='comm'\")."
            ),
        ),
    ] = None,
    out_dir: Annotated[
        pathlib.Path,
        typer.Option(
            "--out-dir", "-o", help="Output directory for generated figures."
        ),
    ] = pathlib.Path("figures"),
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Figure file extension: pdf, svg, png (static) or html (one "
                "interactive file per plot)."
            ),
        ),
    ] = "pdf",
    html: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--html",
            help=(
                "Write ONE interactive dashboard (tabs, group/rank/step "
                "filters) to this file; needs cta[viz]."
            ),
        ),
    ] = None,
    serve: Annotated[
        bool,
        typer.Option(
            "--serve",
            help=(
                "Open the dashboard LIVE on a local server: every filter "
                "(group, rank, GPU device, step range) recomputes the "
                "analysis. Ctrl-C stops it."
            ),
        ),
    ] = False,
    port: Annotated[
        int,
        typer.Option("--port", help="Port for --serve (0 = pick a free one)."),
    ] = 0,
    no_open: Annotated[
        bool,
        typer.Option(
            "--no-open",
            help="With --serve, print the URL instead of opening a browser.",
        ),
    ] = False,
    cdn: Annotated[
        bool,
        typer.Option(
            "--cdn",
            help=(
                "Dashboard/HTML loads plotly.js from a CDN instead of "
                "embedding it (smaller file, needs internet)."
            ),
        ),
    ] = False,
    history: Annotated[
        str | None,
        typer.Option(
            "--history",
            help=(
                "Glob of saved report.json files for --kind history, e.g. "
                "'reports/*/report.json'."
            ),
        ),
    ] = None,
    metric: Annotated[
        str | None,
        typer.Option(
            "--metric",
            help="Shortcut for -O metric=... (history, sweep, scaling).",
        ),
    ] = None,
    theme: Annotated[
        str | None, typer.Option("--theme", help="light or dark.")
    ] = None,
    palette: Annotated[
        str | None,
        typer.Option(
            "--palette",
            help="okabe_ito, tableau, pastel, or comma-separated colours.",
        ),
    ] = None,
    title: Annotated[
        str | None, typer.Option("--title", help="Override the figure title.")
    ] = None,
    style: Annotated[
        str,
        typer.Option(
            "--style",
            help=(
                "Static size preset (paper_single_col, paper_double_col, "
                "presentation, web)."
            ),
        ),
    ] = "paper_single_col",
    dpi: Annotated[
        int, typer.Option("--dpi", help="Raster resolution for png.")
    ] = 300,
    warmup_steps: Annotated[
        int,
        typer.Option(
            "--warmup-steps",
            help=(
                "Skip initial warmup steps/iterations (ignored with "
                "--manifest)."
            ),
        ),
    ] = 0,
    max_steps: Annotated[
        int | None,
        typer.Option(
            "--max-steps",
            help="Max steps/iterations to analyze (ignored with --manifest).",
        ),
    ] = None,
    cache_dir: Annotated[
        pathlib.Path, typer.Option("--cache-dir")
    ] = pathlib.Path(".cta_cache"),
    list_kinds: Annotated[
        bool,
        typer.Option(
            "--list-kinds",
            help=(
                "List every plot kind with its data needs and options, then "
                "exit."
            ),
        ),
    ] = False,
    list_sources: Annotated[
        bool,
        typer.Option(
            "--list-sources",
            help=(
                "List the tables (and columns) usable in --sql/--source for "
                "these traces, then exit."
            ),
        ),
    ] = False,
) -> None:
    """Render plots: static PDF/SVG/PNG, per-plot interactive HTML, or one interactive dashboard (--html)."""  # noqa: E501
    from cta.core import engine
    from cta.core import manifest as core_manifest
    from cta.viz import data
    from cta.viz import spec as viz_spec

    if list_kinds:
        _print_kinds()
        return

    manifest_plots: list[dict[str, object]] = []
    report = None
    try:
        if manifest is not None:
            groups, config = core_manifest.load_manifest(manifest)
            manifest_plots = config.plots
            report = engine.run_comparison(groups, config)
        elif traces:
            if len(traces) == 1:
                report = engine.analyze(
                    traces[0],
                    warmup_steps=warmup_steps,
                    max_steps=max_steps,
                    cache_dir=cache_dir,
                )
            else:
                report = engine.compare(
                    traces[0],
                    traces[1],
                    *traces[2:],
                    warmup_steps=warmup_steps,
                    max_steps=max_steps,
                    cache_dir=cache_dir,
                )
        elif not history:
            raise typer.BadParameter(
                "Provide at least one trace path/directory, --manifest, or "
                "--history GLOB."
            )
        ctx = data.DataContext(report, history=history)
        if list_sources:
            for name, columns in ctx.describe().items():
                output.console.print(
                    f"[bold]{name}[/bold]: {', '.join(columns)}"
                )
            return

        # Options given once for every plot: the shortcut flags, then --opt.
        shortcut_options = {
            "metric": metric,
            "theme": theme,
            "palette": palette,
            "title": title,
        }
        shared_options: dict[str, str] = {
            key: value
            for key, value in shortcut_options.items()
            if value is not None
        }
        shared_options.update(_parse_options(opt))
        if history:
            shared_options.setdefault("history", history)

        specs: list[viz_spec.PlotSpec] = []
        has_picked_plots = any(
            value is not None
            for value in (spec, kind, sql, source, mark, x, y, color, facet)
        ) or bool(manifest_plots)
        if spec is not None:
            specs += viz_spec.load_specs(spec)
        has_custom_plot = any(
            value is not None
            for value in (sql, source, mark, x, y, color, facet)
        )
        if has_custom_plot:
            custom_fields = {
                "sql": sql,
                "source": source,
                "mark": mark or "line",
                "x": x,
                "y": y,
                "color": color,
                "facet": facet,
            }
            custom_options = {
                key: value
                for key, value in custom_fields.items()
                if value is not None
            }
            specs.append(
                viz_spec.PlotSpec("custom", custom_options, name="custom")
            )
        kinds = [
            name.strip() for name in (kind or "").split(",") if name.strip()
        ]
        is_all_requested = "all" in kinds
        if not is_all_requested:
            specs += [viz_spec.PlotSpec(name) for name in kinds]
        if not specs and not is_all_requested:
            default_kind = "breakdown" if not history else "history"
            specs = [
                viz_spec.PlotSpec.from_dict(plot) for plot in manifest_plots
            ] or [viz_spec.PlotSpec(default_kind)]

        unused_options = [
            key
            for key in shared_options
            if specs
            and not any(
                key in viz_spec.get_kind(plot_spec.kind).all_options()
                for plot_spec in specs
            )
        ]
        if unused_options:
            raise ValueError(
                f"option(s) {unused_options} are not accepted by any selected "
                "plot kind (see --list-kinds)"
            )
        specs = [
            _with_shared_options(plot_spec, shared_options)
            for plot_spec in specs
        ]
        # Without an explicit choice, None makes the dashboard draw its
        # default plots for the data.
        dashboard_specs = specs if has_picked_plots and specs else None
        page_title = title or "CTA report"

        if serve:
            from cta.viz import server

            if report is None:
                raise ValueError("--serve needs traces or --manifest")

            def announce_url(url: str) -> None:
                output.console.print(
                    f"[green]✓ Live dashboard:[/green] {url}  (Ctrl-C to stop)"
                )

            server.serve(
                report,
                dashboard_specs,
                port=port,
                title=page_title,
                theme=shared_options.get("theme", "light"),
                palette=shared_options.get("palette"),
                open_browser=not no_open,
                on_ready=announce_url,
            )
            return

        if html is not None:
            from cta.viz import dashboard

            result = dashboard.build_dashboard(
                ctx,
                dashboard_specs,
                html,
                title=page_title,
                cdn=cdn,
                theme=shared_options.get("theme", "light"),
                palette=shared_options.get("palette"),
            )
            output.console.print(
                f"[green]✓ Dashboard:[/green] {result.path} "
                f"({len(result.drawn)} plots)"
            )
            for kind_name, reason in result.skipped.items():
                output.console.print(
                    f"[dim]  skipped {kind_name}: {reason}[/dim]"
                )
            return

        from cta.viz import dashboard
        from cta.viz import plots

        is_explicit = bool(specs)
        if specs:
            specs_to_draw = specs
        elif is_all_requested:
            specs_to_draw = [
                _with_shared_options(plot_spec, shared_options)
                for plot_spec in dashboard.default_specs(ctx)
            ]
        else:
            specs_to_draw = []
        extension = fmt.lstrip(".").lower()
        drawn_count = 0
        for plot_spec in specs_to_draw:
            figure_path = out_dir / f"{plot_spec.stem}.{extension}"
            try:
                plots.render_spec(
                    ctx, plot_spec, figure_path, style=style, dpi=dpi
                )
            except viz_spec.NotApplicableError as exc:
                if is_explicit:
                    raise
                output.console.print(
                    f"[dim]  skipped {plot_spec.stem}: {exc}[/dim]"
                )
                continue
            drawn_count += 1
            output.console.print(
                f"[green]✓ Saved figure:[/green] {figure_path}"
            )
        if not drawn_count:
            raise ValueError("no plot could be drawn for these traces")
    except (ValueError, ImportError, FileNotFoundError) as exc:
        output.err_console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc
