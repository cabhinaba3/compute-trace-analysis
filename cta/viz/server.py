"""Live dashboard: a loopback HTTP server that recomputes the analysis for the filters the page sends.

Each request re-runs ``run_comparison`` on the selected labels with ``ranks``/``devices`` and a step window
(``warmup_steps`` and ``max_steps`` shifted by the window start), then rebuilds the plot with the ordinary plot
code. Cache hits make that cheap, and because it is the same code path every aggregate is exact. Stdlib only.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from cta.core.engine import run_comparison
from cta.core.models import CompareConfig, ComparisonReport
from cta.viz.dashboard import Card, figure_json, filter_choices, page, plan
from cta.viz.data import DataContext
from cta.viz.spec import PlotSpec


@dataclass(frozen=True, slots=True)
class Filters:
    labels: tuple[str, ...] | None = None
    ranks: tuple[int, ...] | None = None
    devices: tuple[int, ...] | None = None
    steps: tuple[int, int] | None = None  # window over the already warm-up-sliced steps

    @classmethod
    def from_query(cls, q: dict[str, list[str]]) -> Filters:
        def ints(key: str) -> tuple[int, ...] | None:
            raw = q.get(key, [""])[0]
            return None if not raw else tuple(int(v) for v in raw.split(",") if v not in ("", "none"))

        raw_labels = q.get("labels", [""])[0]
        steps = q.get("steps", [""])[0]
        lo, _, hi = steps.partition(":")
        return cls(
            labels=tuple(raw_labels.split(",")) if raw_labels and raw_labels != "none" else (() if raw_labels == "none" else None),
            ranks=ints("ranks"), devices=ints("devices"),
            steps=(int(lo or 0), int(hi)) if steps and hi else None,
        )  # fmt: skip


class LiveSession:
    """The report being explored plus a small cache of recomputed reports keyed by filter."""

    def __init__(self, report: ComparisonReport, specs: list[PlotSpec] | None, *, theme: str = "light", palette: str | None = None) -> None:
        self.base = report
        self.theme, self.palette = theme, palette
        self.base_ctx = DataContext(report)
        self.cards, self.skipped = plan(self.base_ctx, specs, theme, palette)
        self.choices = filter_choices(self.base_ctx)
        self._cache: OrderedDict[Filters, ComparisonReport] = OrderedDict()
        self._lock = threading.Lock()

    def report_for(self, f: Filters) -> ComparisonReport:
        cfg = self.base.config or CompareConfig()
        full = (f.labels is None or len(f.labels) == len(self.base.groups)) and f.ranks is None and f.devices is None
        lo, hi = f.steps or (0, self.choices["nsteps"])
        full = full and lo == 0 and (hi >= self.choices["nsteps"] or not self.choices["nsteps"])
        if full:
            return self.base
        with self._lock:
            if f in self._cache:
                self._cache.move_to_end(f)
                return self._cache[f]
            groups = tuple(g for g in self.base.groups if f.labels is None or g.label in f.labels)
            if not groups:
                raise ValueError("no group selected")
            n = self.choices["nsteps"]
            new_cfg = replace(
                cfg, ranks=f.ranks if f.ranks is not None else cfg.ranks, devices=f.devices if f.devices is not None else cfg.devices,
                warmup_steps=cfg.warmup_steps + lo, max_steps=(hi - lo) if hi < n else cfg.max_steps,
            )  # fmt: skip
            rep = run_comparison(groups, new_cfg)
            self._cache[f] = rep
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
            return rep

    def plot(self, card: Card, f: Filters) -> dict[str, Any]:
        try:
            rep = self.report_for(f)
            fj, _, notes = figure_json(DataContext(rep), card.spec, self.theme, self.palette)
        except Exception as exc:  # shown in the card: an empty slice or an inapplicable plot must not kill the page
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"figure": fj, "notes": notes}

    def html(self, title: str, js: str | None) -> str:
        return page(title, self.cards, self.choices, js, self.theme, self.skipped, live=True)


def make_server(report: ComparisonReport, specs: list[PlotSpec] | None, *, port: int = 0, title: str = "CTA report", theme: str = "light",
                palette: str | None = None) -> tuple[ThreadingHTTPServer, LiveSession]:  # fmt: skip
    from plotly.offline import get_plotlyjs

    session = LiveSession(report, specs, theme=theme, palette=palette)
    index = session.html(title, get_plotlyjs()).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:  # quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(index, "text/html; charset=utf-8")
            elif url.path == "/api/meta":
                self._send(json.dumps(session.choices).encode(), "application/json")
            elif url.path == "/api/plot":
                q = parse_qs(url.query)
                try:
                    out = session.plot(session.cards[int(q["id"][0])], Filters.from_query(q))
                except (KeyError, IndexError, ValueError) as exc:
                    out = {"error": f"bad request: {exc}"}
                self._send(json.dumps(out).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler), session  # loopback only: the page can re-run analyses on this machine


def serve(report: ComparisonReport, specs: list[PlotSpec] | None, *, port: int = 0, title: str = "CTA report", theme: str = "light",
          palette: str | None = None, open_browser: bool = True, on_ready: Any = None) -> None:  # fmt: skip
    httpd, _ = make_server(report, specs, port=port, title=title, theme=theme, palette=palette)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    if on_ready:
        on_ready(url)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
