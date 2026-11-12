"""One interactive HTML dashboard per report: tabs per use case and shared filters.

Needs ``pip install cta[viz]``. Two modes share one page:

* **Static file** (``build_dashboard``): fully offline, plotly.js embedded once (``cdn=True`` loads it instead). The
  filters hide or restyle what is already drawn: group (also category-axis bars and heatmap columns), rank, and a
  step-range zoom for step-axis plots. Aggregates are not recomputed, because that needs Python.
* **Live** (``cta plot ... --serve``, see ``cta.viz.server``): every filter change asks the local server to re-run the
  analysis for that slice (labels, ranks, devices, step range), so all aggregates are exact.
"""

from __future__ import annotations

import contextlib
import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cta.viz.data import DataContext
from cta.viz.plots import build_chart, to_plotly
from cta.viz.spec import GROUPS, NotApplicable, PlotSpec, get_kind, kind_catalog

_TITLES = {
    "overview": "Overview", "timeline": "Timeline", "steps": "Steps", "compare": "A/B", "sweep": "Sweeps",
    "distributed": "Distributed", "kernels": "Kernels", "edge": "Edge / energy", "drift": "Drift", "custom": "Custom",
}  # fmt: skip


def plotly_js_version() -> str:
    from plotly.offline import get_plotlyjs_version

    return str(get_plotlyjs_version())


@dataclass(slots=True)
class DashboardResult:
    path: Path
    drawn: list[str]
    skipped: dict[str, str]


@dataclass(slots=True)
class Card:
    id: int
    stem: str
    group: str
    title: str
    notes: str
    spec: PlotSpec
    fig_json: str


def default_specs(ctx: DataContext) -> list[PlotSpec]:
    """Every registered kind except the ones that need user input (custom, and history without a glob)."""
    return [PlotSpec(kind=k.name) for k in kind_catalog() if not (k.name == "custom" or (k.name == "history" and not ctx.history))]


def _require_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError("the interactive dashboard needs plotly: pip install 'cta[viz]'") from exc


def figure_json(ctx: DataContext, spec: PlotSpec, theme: str = "light", palette: str | None = None) -> tuple[str, str, str]:
    """(plotly figure JSON, chart title, notes) for one spec; raises NotApplicable/ValueError."""
    import plotly.io as pio

    opts = {"theme": theme, **({"palette": palette} if palette else {}), **spec.options}
    chart, resolved = build_chart(ctx, spec.kind, opts)
    return pio.to_json(to_plotly(chart, resolved)).replace("</", "<\\/"), chart.title, chart.notes


def plan(ctx: DataContext, specs: Sequence[PlotSpec] | None, theme: str = "light", palette: str | None = None) -> tuple[list[Card], dict[str, str]]:
    """Build every plot once on the full data; the ones that cannot be drawn are reported, not silently dropped."""
    _require_plotly()
    cards: list[Card] = []
    skipped: dict[str, str] = {}
    for spec in specs if specs is not None else default_specs(ctx):
        try:
            fj, title, notes = figure_json(ctx, spec, theme, palette)
        except (NotApplicable, ValueError) as exc:
            skipped[spec.stem] = str(exc)
            continue
        cards.append(Card(len(cards), spec.stem, get_kind(spec.kind).group, title, notes, spec, fj))
    if not cards:
        raise NotApplicable("no plot could be drawn for this report: " + "; ".join(f"{k}: {v}" for k, v in list(skipped.items())[:5]))
    return cards, skipped


def filter_choices(ctx: DataContext) -> dict[str, Any]:
    """Values the filter bar offers: labels, ranks, GPU devices, and the largest number of steps."""
    out: dict[str, Any] = {"labels": ctx.labels, "ranks": [], "devices": [], "nsteps": 0}
    with contextlib.suppress(NotApplicable):
        steps = ctx.frame("steps")
        out["ranks"] = sorted(int(r) for r in steps["rank"].unique().to_list())
        out["nsteps"] = int(steps.group_by("label", "trial", "rank").len()["len"].max() or 0)  # type: ignore[call-overload,arg-type]
    with contextlib.suppress(NotApplicable):
        out["devices"] = sorted(int(d) for d in ctx.frame("kernels")["device"].unique().to_list())
    return out


def build_dashboard(
    ctx: DataContext, specs: Sequence[PlotSpec] | None, path: Path, *, title: str = "CTA report", cdn: bool = False,
    theme: str = "light", palette: str | None = None,
) -> DashboardResult:  # fmt: skip
    from plotly.offline import get_plotlyjs

    cards, skipped = plan(ctx, specs, theme, palette)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page(title, cards, filter_choices(ctx), get_plotlyjs() if not cdn else None, theme, skipped, live=False), encoding="utf-8")
    return DashboardResult(path, [c.stem for c in cards], skipped)


def page(title: str, cards: list[Card], choices: dict[str, Any], js: str | None, theme: str, skipped: dict[str, str], *, live: bool) -> str:
    groups = [g for g in GROUPS if any(c.group == g for c in cards)]
    nav = "".join(f'<button class="tab" data-tab="{g}">{_TITLES[g]} <span>{sum(c.group == g for c in cards)}</span></button>' for g in groups)
    body = ""
    for g in groups:
        body += f'<section id="tab-{g}" class="pane">'
        for c in cards:
            if c.group != g:
                continue
            payload = "" if live else f'<script type="application/json" id="fig-{c.id}">{c.fig_json}</script>'
            note = f'<p class="note" id="note-{c.id}">{html.escape(c.notes)}</p>' if (c.notes or live) else ""
            body += f'<div class="card"><h3>{html.escape(c.title)} <small>{html.escape(c.spec.stem)}</small></h3><div class="plot" id="plot-{c.id}"></div>{note}{payload}</div>'
        body += "</section>"

    def boxes(cls: str, values: list[Any], prefix: str = "") -> str:
        return "".join(f'<label><input type="checkbox" class="{cls}" value="{html.escape(str(v))}" checked> {prefix}{html.escape(str(v))}</label>' for v in values)

    nsteps = choices["nsteps"]
    filters = (
        f'<span>{boxes("f-label", choices["labels"])}</span>'
        f'<span>{boxes("f-rank", choices["ranks"], "r") if len(choices["ranks"]) > 1 else ""}</span>'
        f'<span>{boxes("f-device", choices["devices"], "gpu") if live and len(choices["devices"]) > 1 else ""}</span>'
        + (f'<label>steps <input id="s-lo" type="number" min="0" value="0" style="width:4em"> to <input id="s-hi" type="number" min="1" value="{nsteps}" style="width:4em"></label>' if nsteps > 1 else "")
        + ('<span id="status" class="status">live: every filter recomputes the analysis</span>' if live else "")
    )
    script_tag = f"<script>{js}</script>" if js else f'<script src="https://cdn.plot.ly/plotly-{plotly_js_version()}.min.js"></script>'
    skipped_html = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in skipped.items())
    dark = theme == "dark"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>
body{{font:14px system-ui,sans-serif;margin:0;background:{"#111418" if dark else "#fafafa"};color:{"#d7dae0" if dark else "#222"}}}
header{{padding:12px 20px;border-bottom:1px solid #8884}}h1{{font-size:18px;margin:0 0 8px}}
.filters{{display:flex;gap:18px;flex-wrap:wrap;align-items:center;font-size:13px}}.filters label{{margin-right:8px;white-space:nowrap}}
.status{{opacity:.6}}.status.busy{{opacity:1;color:#D55E00}}.err{{color:#D55E00;padding:12px;font-size:13px}}
nav{{display:flex;gap:4px;padding:8px 20px;border-bottom:1px solid #8884;flex-wrap:wrap}}
.tab{{border:1px solid #8886;background:transparent;color:inherit;padding:6px 12px;border-radius:6px;cursor:pointer}}.tab span{{opacity:.6}}
.tab.on{{background:#0072B2;color:#fff;border-color:#0072B2}}.pane{{display:none;padding:12px 20px}}.pane.on{{display:block}}
.card{{margin:0 0 18px;padding:8px 12px;border:1px solid #8883;border-radius:8px}}.card h3{{margin:4px 0;font-size:15px}}.card small{{opacity:.5;font-weight:normal;margin-left:8px}}
.note{{margin:4px 0 0;font-size:12px;opacity:.75}}details{{padding:8px 20px;font-size:12px;opacity:.8}}</style>{script_tag}</head><body>
<header><h1>{html.escape(title)}</h1><div class="filters">{filters}</div></header>
<nav>{nav}</nav>{body}{f"<details><summary>{len(skipped)} plots not applicable</summary><ul>{skipped_html}</ul></details>" if skipped else ""}
<script>{_LIVE_JS if live else _STATIC_JS.replace("__LABELS__", json.dumps(choices["labels"]))}</script></body></html>"""


_COMMON_JS = """
const done=new Set();
function show(tab){document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('on',b.dataset.tab===tab));
 document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('on',p.id==='tab-'+tab));
 document.querySelectorAll('#tab-'+tab+' .plot').forEach(d=>{const i=d.id.split('-')[1];
  if(!done.has(i)){done.add(i);draw(d,i);}else Plotly.Plots.resize(d);});}
function checked(c){return new Set([...document.querySelectorAll(c+':checked')].map(x=>x.value));}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>show(b.dataset.tab)));
"""

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
"""
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
"""
    + _COMMON_JS
    + "show(document.querySelector('.tab').dataset.tab);"
)
