"""One interactive HTML dashboard per report: tabs per use case, shared filters.

Needs ``pip install cta[viz]``. Two modes share one page:

* **Static file** (``build_dashboard``): fully offline, plotly.js embedded
  once (``cdn=True`` loads it instead). The filters hide or restyle what is
  already drawn: group (also category-axis bars and heatmap columns), rank,
  and a step-range zoom for step-axis plots. Aggregates are not recomputed,
  because that needs Python.
* **Live** (``cta plot ... --serve``, see ``cta.viz.server``): every filter
  change asks the local server to re-run the analysis for that slice (labels,
  ranks, devices, step range), so all aggregates are exact.
"""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
import dataclasses
import html
import json
import pathlib
from typing import Any

from cta.viz import data
from cta.viz import plots
from cta.viz import spec as viz_spec

# Plot group -> tab title.
_TAB_TITLES = {
    "overview": "Overview",
    "timeline": "Timeline",
    "steps": "Steps",
    "compare": "A/B",
    "sweep": "Sweeps",
    "distributed": "Distributed",
    "kernels": "Kernels",
    "edge": "Edge / energy",
    "drift": "Drift",
    "custom": "Custom",
}

# Page CSS after the theme-dependent ``body`` rule.
_STYLE = (
    "header{padding:12px 20px;border-bottom:1px solid #8884}"
    "h1{font-size:18px;margin:0 0 8px}\n"
    ".filters{display:flex;gap:18px;flex-wrap:wrap;align-items:center;"
    "font-size:13px}"
    ".filters label{margin-right:8px;white-space:nowrap}\n"
    ".status{opacity:.6}.status.busy{opacity:1;color:#D55E00}"
    ".err{color:#D55E00;padding:12px;font-size:13px}\n"
    "nav{display:flex;gap:4px;padding:8px 20px;"
    "border-bottom:1px solid #8884;flex-wrap:wrap}\n"
    ".tab{border:1px solid #8886;background:transparent;color:inherit;"
    "padding:6px 12px;border-radius:6px;cursor:pointer}"
    ".tab span{opacity:.6}\n"
    ".tab.on{background:#0072B2;color:#fff;border-color:#0072B2}"
    ".pane{display:none;padding:12px 20px}.pane.on{display:block}\n"
    ".card{margin:0 0 18px;padding:8px 12px;border:1px solid #8883;"
    "border-radius:8px}"
    ".card h3{margin:4px 0;font-size:15px}"
    ".card small{opacity:.5;font-weight:normal;margin-left:8px}\n"
    ".note{margin:4px 0 0;font-size:12px;opacity:.75}"
    "details{padding:8px 20px;font-size:12px;opacity:.8}"
)


def plotly_js_version() -> str:
    """Returns the version of the plotly.js bundled with the Plotly package."""
    from plotly import offline

    return str(offline.get_plotlyjs_version())


@dataclasses.dataclass(slots=True)
class DashboardResult:
    """What :func:`build_dashboard` wrote.

    Attributes:
        path: The HTML file.
        drawn: Stems of the drawn plots, in spec order.
        skipped: Stem -> reason, for the plots that could not be drawn.
    """

    path: pathlib.Path
    drawn: list[str]
    skipped: dict[str, str]


@dataclasses.dataclass(slots=True)
class Card:
    """One plot of the dashboard.

    Attributes:
        index: Index of the card among the drawn plots (spec order); names the
            plot's page elements and is the plot index the live server answers
            for.
        stem: The spec's output stem.
        group: Plot group, which selects the tab.
        title: Chart title.
        notes: Chart caption.
        spec: The plot spec.
        fig_json: Plotly figure JSON of the full data, safe to embed in a
            ``<script>`` element (``</`` is escaped).
    """

    index: int
    stem: str
    group: str
    title: str
    notes: str
    spec: viz_spec.PlotSpec
    fig_json: str


def default_specs(ctx: data.DataContext) -> list[viz_spec.PlotSpec]:
    """Returns a spec for every registered kind that needs no user input.

    Leaves out ``custom`` (it needs a query) and ``history`` unless the
    context has a history glob.

    Args:
        ctx: Data the plots draw from.

    Returns:
        One spec with default options per kind, in catalog order.
    """
    return [
        viz_spec.PlotSpec(kind=kind.name)
        for kind in viz_spec.kind_catalog()
        if not (
            kind.name == "custom"
            or (kind.name == "history" and not ctx.history)
        )
    ]


def _require_plotly() -> None:
    """Raises ImportError with an install hint if Plotly is missing."""
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the interactive dashboard needs plotly: pip install 'cta[viz]'"
        ) from exc


def figure_json(
    ctx: data.DataContext,
    spec: viz_spec.PlotSpec,
    theme: str = "light",
    palette: str | None = None,
) -> tuple[str, str, str]:
    """Builds one spec's Plotly figure as JSON.

    Args:
        ctx: Data the plot draws from.
        spec: The plot spec; its own ``theme``/``palette`` options win.
        theme: Page theme, ``"light"`` or ``"dark"``.
        palette: Palette name or colour list; None keeps the default.

    Returns:
        The figure JSON (with ``</`` escaped for embedding in HTML), the
        chart title and the chart notes.

    Raises:
        viz_spec.NotApplicableError: If the plot cannot be drawn from this data.
        ValueError: If the kind or an option is invalid.
    """
    import plotly.io as pio

    options = {
        "theme": theme,
        **({"palette": palette} if palette else {}),
        **spec.options,
    }
    chart, resolved = plots.build_chart(ctx, spec.kind, options)
    fig_json = pio.to_json(plots.to_plotly(chart, resolved))
    return fig_json.replace("</", "<\\/"), chart.title, chart.notes


def plan(
    ctx: data.DataContext,
    specs: Sequence[viz_spec.PlotSpec] | None,
    theme: str = "light",
    palette: str | None = None,
) -> tuple[list[Card], dict[str, str]]:
    """Builds every plot once on the full data.

    Plots that cannot be drawn are reported, not silently dropped.

    Args:
        ctx: Data the plots draw from.
        specs: The plots; None uses :func:`default_specs`.
        theme: Page theme, ``"light"`` or ``"dark"``.
        palette: Palette name or colour list; None keeps the default.

    Returns:
        The cards of the drawn plots, and stem -> reason for the skipped
        ones.

    Raises:
        ImportError: If Plotly is not installed.
        viz_spec.NotApplicableError: If no plot could be drawn.
    """
    _require_plotly()
    cards: list[Card] = []
    skipped: dict[str, str] = {}
    for spec in specs if specs is not None else default_specs(ctx):
        try:
            fig_json, title, notes = figure_json(ctx, spec, theme, palette)
        except (viz_spec.NotApplicableError, ValueError) as exc:
            skipped[spec.stem] = str(exc)
            continue
        cards.append(
            Card(
                len(cards),
                spec.stem,
                viz_spec.get_kind(spec.kind).group,
                title,
                notes,
                spec,
                fig_json,
            )
        )
    if not cards:
        first_reasons = list(skipped.items())[:5]
        raise viz_spec.NotApplicableError(
            "no plot could be drawn for this report: "
            + "; ".join(f"{stem}: {reason}" for stem, reason in first_reasons)
        )
    return cards, skipped


def filter_choices(ctx: data.DataContext) -> dict[str, Any]:
    """Returns the values the filter bar offers.

    Args:
        ctx: Data the plots draw from.

    Returns:
        ``labels``, ``ranks`` and ``devices`` (lists) and ``nsteps``, the
        largest number of steps of one rank in one trial. Ranks and steps
        stay empty/0 without a steps table, devices without a kernels table.
    """
    choices: dict[str, Any] = {
        "labels": ctx.labels,
        "ranks": [],
        "devices": [],
        "nsteps": 0,
    }
    with contextlib.suppress(viz_spec.NotApplicableError):
        steps = ctx.frame("steps")
        choices["ranks"] = sorted(
            int(rank) for rank in steps["rank"].unique().to_list()
        )
        steps_per_rank = steps.group_by("label", "trial", "rank").len()["len"]
        choices["nsteps"] = int(
            steps_per_rank.max() or 0  # type: ignore[arg-type]
        )
    with contextlib.suppress(viz_spec.NotApplicableError):
        devices = ctx.frame("kernels")["device"].unique().to_list()
        choices["devices"] = sorted(int(device) for device in devices)
    return choices


def build_dashboard(
    ctx: data.DataContext,
    specs: Sequence[viz_spec.PlotSpec] | None,
    path: pathlib.Path,
    *,
    title: str = "CTA report",
    cdn: bool = False,
    theme: str = "light",
    palette: str | None = None,
) -> DashboardResult:
    """Writes the static dashboard of a report.

    Args:
        ctx: Data the plots draw from.
        specs: The plots; None uses :func:`default_specs`.
        path: Output HTML file; missing parent directories are created.
        title: Page title.
        cdn: If True, the page loads plotly.js from the CDN instead of
            embedding it.
        theme: Page theme, ``"light"`` or ``"dark"``.
        palette: Palette name or colour list; None keeps the default.

    Returns:
        The written file and the drawn and skipped plots.

    Raises:
        ImportError: If Plotly is not installed.
        viz_spec.NotApplicableError: If no plot could be drawn.
    """
    from plotly import offline

    cards, skipped = plan(ctx, specs, theme, palette)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        page(
            title,
            cards,
            filter_choices(ctx),
            offline.get_plotlyjs() if not cdn else None,
            theme,
            skipped,
            live=False,
        ),
        encoding="utf-8",
    )
    return DashboardResult(path, [card.stem for card in cards], skipped)


def page(
    title: str,
    cards: list[Card],
    choices: dict[str, Any],
    plotly_js: str | None,
    theme: str,
    skipped: dict[str, str],
    *,
    live: bool,
) -> str:
    """Returns the dashboard HTML page.

    Args:
        title: Page title.
        cards: The drawn plots.
        choices: Filter values, from :func:`filter_choices`.
        plotly_js: The plotly.js source to embed; None loads it from the CDN.
        theme: ``"dark"`` uses dark page colours, anything else light ones.
        skipped: Stem -> reason, for the plots that could not be drawn.
        live: If True, the page asks the live server for every plot (and
            offers the device filter); else it embeds the figures and filters
            them in the browser.

    Returns:
        The HTML document.
    """
    groups = [
        group
        for group in viz_spec.GROUPS
        if any(card.group == group for card in cards)
    ]
    nav = "".join(_tab_button(group, cards) for group in groups)
    body = "".join(_tab_pane(group, cards, live=live) for group in groups)
    filters = _filter_bar(choices, live=live)
    if plotly_js:
        script_tag = f"<script>{plotly_js}</script>"
    else:
        plotly_js_url = (
            f"https://cdn.plot.ly/plotly-{plotly_js_version()}.min.js"
        )
        script_tag = f'<script src="{plotly_js_url}"></script>'
    skipped_items = "".join(
        f"<li><b>{html.escape(stem)}</b>: {html.escape(reason)}</li>"
        for stem, reason in skipped.items()
    )
    skipped_details = (
        f"<details><summary>{len(skipped)} plots not applicable</summary>"
        f"<ul>{skipped_items}</ul></details>"
        if skipped
        else ""
    )
    page_js = (
        _LIVE_JS
        if live
        else _STATIC_JS.replace("__LABELS__", json.dumps(choices["labels"]))
    )
    is_dark = theme == "dark"
    background = "#111418" if is_dark else "#fafafa"
    foreground = "#d7dae0" if is_dark else "#222"
    escaped_title = html.escape(title)
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{escaped_title}</title><style>\n"
        "body{font:14px system-ui,sans-serif;margin:0;"
        f"background:{background};color:{foreground}}}\n"
        f"{_STYLE}</style>{script_tag}</head><body>\n"
        f"<header><h1>{escaped_title}</h1>"
        f'<div class="filters">{filters}</div></header>\n'
        f"<nav>{nav}</nav>{body}{skipped_details}\n"
        f"<script>{page_js}</script></body></html>"
    )


def _tab_button(group: str, cards: list[Card]) -> str:
    """Returns the nav button of a group's tab, with its plot count."""
    count = sum(card.group == group for card in cards)
    return (
        f'<button class="tab" data-tab="{group}">'
        f"{_TAB_TITLES[group]} <span>{count}</span></button>"
    )


def _tab_pane(group: str, cards: list[Card], *, live: bool) -> str:
    """Returns the tab pane holding a group's cards."""
    group_cards = "".join(
        _card_html(card, live=live) for card in cards if card.group == group
    )
    return f'<section id="tab-{group}" class="pane">{group_cards}</section>'


def _card_html(card: Card, *, live: bool) -> str:
    """Returns one plot's card.

    A static card embeds its figure JSON; a live card fetches the figure
    and always has a notes element for the server's notes.
    """
    payload = (
        ""
        if live
        else f'<script type="application/json" id="fig-{card.index}">'
        f"{card.fig_json}</script>"
    )
    note = (
        f'<p class="note" id="note-{card.index}">{html.escape(card.notes)}</p>'
        if (card.notes or live)
        else ""
    )
    card_title = html.escape(card.title)
    stem = html.escape(card.spec.stem)
    return (
        f'<div class="card"><h3>{card_title} <small>{stem}</small></h3>'
        f'<div class="plot" id="plot-{card.index}"></div>{note}{payload}</div>'
    )


def _checkboxes(css_class: str, values: list[Any], prefix: str = "") -> str:
    """Returns one checked checkbox per value, labelled ``prefix + value``."""
    return "".join(
        f'<label><input type="checkbox" class="{css_class}" '
        f'value="{html.escape(str(value))}" checked> '
        f"{prefix}{html.escape(str(value))}</label>"
        for value in values
    )


def _filter_bar(choices: dict[str, Any], *, live: bool) -> str:
    """Returns the filter bar contents.

    Group checkboxes always; rank checkboxes with more than one rank; device
    checkboxes on a live page with more than one device; a step range with
    more than one step; and the live status line.
    """
    label_boxes = _checkboxes("f-label", choices["labels"])
    rank_boxes = (
        _checkboxes("f-rank", choices["ranks"], "r")
        if len(choices["ranks"]) > 1
        else ""
    )
    device_boxes = (
        _checkboxes("f-device", choices["devices"], "gpu")
        if live and len(choices["devices"]) > 1
        else ""
    )
    nsteps = choices["nsteps"]
    step_range = (
        '<label>steps <input id="s-lo" type="number" min="0" value="0" '
        'style="width:4em"> to <input id="s-hi" type="number" min="1" '
        f'value="{nsteps}" style="width:4em"></label>'
        if nsteps > 1
        else ""
    )
    status = (
        '<span id="status" class="status">'
        "live: every filter recomputes the analysis</span>"
        if live
        else ""
    )
    return (
        f"<span>{label_boxes}</span>"
        f"<span>{rank_boxes}</span>"
        f"<span>{device_boxes}</span>" + step_range + status
    )


# The page scripts stay verbatim JavaScript, so their long lines are kept.
_COMMON_JS = """
const done=new Set();
function show(tab){document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('on',b.dataset.tab===tab));
 document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('on',p.id==='tab-'+tab));
 document.querySelectorAll('#tab-'+tab+' .plot').forEach(d=>{const i=d.id.split('-')[1];
  if(!done.has(i)){done.add(i);draw(d,i);}else Plotly.Plots.resize(d);});}
function checked(c){return new Set([...document.querySelectorAll(c+':checked')].map(x=>x.value));}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>show(b.dataset.tab)));
"""  # noqa: E501

_STATIC_JS = (
    """const LABELS=new Set(__LABELS__);const figs={};
function figOf(i){return figs[i]||(figs[i]=JSON.parse(document.getElementById('fig-'+i).textContent));}
function draw(d,i){const f=figOf(i);
 Plotly.newPlot(d,structuredClone(f.data),structuredClone(f.layout),{responsive:true,displaylogo:false}).then(()=>apply(d,i));}
function apply(d,i){const labs=checked('.f-label'),rk=checked('.f-rank'),f=figOf(i);
 const vis=f.data.map(t=>{const m=t.meta||{};if(m.label!==undefined&&!labs.has(String(m.label)))return false;
  if(m.rank!==undefined&&rk.size&&!rk.has(String(m.rank)))return false;return true;});
 Plotly.restyle(d,{visible:vis});
 f.data.forEach((t,k)=>{ // plots with the group on a category axis: drop the unchecked groups' bars / heatmap columns
  const cat=t.type==='bar'?(t.orientation==='h'?t.y:t.x):(t.type==='heatmap'?t.x:null);
  if(!cat||!cat.length||!cat.every(c=>LABELS.has(String(c))))return;
  const keep=cat.map(c=>labs.has(String(c)));const pick=a=>a.filter((_,j)=>keep[j]);
  if(t.type==='heatmap')Plotly.restyle(d,{x:[pick(t.x)],z:[t.z.map(pick)]},[k]);
  else if(t.orientation==='h')Plotly.restyle(d,{y:[pick(t.y)],x:[pick(t.x)]},[k]);
  else Plotly.restyle(d,{x:[pick(t.x)],y:[pick(t.y)]},[k]);});
 const lo=document.getElementById('s-lo'),hi=document.getElementById('s-hi');
 const axes=(f.layout.meta&&f.layout.meta.step_axes)||[];
 if(lo&&axes.length){const u={};axes.forEach(a=>u[a+'.range']=[+lo.value-0.5,+hi.value-0.5]);Plotly.relayout(d,u);}}
function applyAll(){done.forEach(i=>apply(document.getElementById('plot-'+i),i));}
document.querySelectorAll('.filters input').forEach(x=>x.addEventListener('change',applyAll));
"""  # noqa: E501
    + _COMMON_JS
    + "show(document.querySelector('.tab').dataset.tab);"
)

_LIVE_JS = (
    """function query(){const q=new URLSearchParams();
 const add=(k,c)=>{const all=document.querySelectorAll(c),on=[...document.querySelectorAll(c+':checked')];
  if(all.length&&on.length!==all.length)q.set(k,on.map(x=>x.value).join(',')||'none');};
 add('labels','.f-label');add('ranks','.f-rank');add('devices','.f-device');
 const lo=document.getElementById('s-lo'),hi=document.getElementById('s-hi');if(lo)q.set('steps',lo.value+':'+hi.value);return q.toString();}
let token=0;
async function draw(d,i){const t=++token;const st=document.getElementById('status');st.classList.add('busy');st.textContent='recomputing...';
 try{const r=await fetch('/api/plot?id='+i+'&'+query());const j=await r.json();
  d.querySelectorAll('.err').forEach(e=>e.remove());
  if(j.error){Plotly.purge(d);d.innerHTML='<div class="err">'+j.error.replace(/</g,'&lt;')+'</div>';}
  else{const f=JSON.parse(j.figure);await Plotly.react(d,f.data,f.layout,{responsive:true,displaylogo:false});}
  document.getElementById('note-'+i).textContent=j.notes||'';}
 catch(e){d.innerHTML='<div class="err">'+e+'</div>';}
 if(t===token){st.classList.remove('busy');st.textContent='live: every filter recomputes the analysis';}}
let timer=null;
function refresh(){clearTimeout(timer);timer=setTimeout(()=>done.forEach(i=>draw(document.getElementById('plot-'+i),i)),300);}
document.querySelectorAll('.filters input').forEach(x=>x.addEventListener('change',refresh));
"""  # noqa: E501
    + _COMMON_JS
    + "show(document.querySelector('.tab').dataset.tab);"
)
