"""Live dashboard: a loopback HTTP server that re-runs the analysis per filter.

Each request re-runs ``run_comparison`` on the selected labels with
``ranks``/``devices`` and a step window (``warmup_steps`` and ``max_steps``
shifted by the window start), then rebuilds the plot with the ordinary plot
code. Cache hits make that cheap, and because it is the same code path every
aggregate is exact. Stdlib only.
"""

from __future__ import annotations

import collections
import dataclasses
from http import server
import json
import threading
from typing import Any
from urllib import parse
import webbrowser

from cta.core import engine
from cta.core import models
from cta.viz import dashboard
from cta.viz import data
from cta.viz import spec

# Recomputed reports kept per session, least recently used evicted first.
_MAX_CACHED_REPORTS = 8


@dataclasses.dataclass(frozen=True, slots=True)
class Filters:
    """A filter selection sent by the live page; None means "not filtered".

    Attributes:
        labels: Selected group labels; an empty tuple selects no group.
        ranks: Selected ranks.
        devices: Selected GPU devices.
        steps: Step window ``(start, end)`` over the already warm-up-sliced
            steps.
    """

    labels: tuple[str, ...] | None = None
    ranks: tuple[int, ...] | None = None
    devices: tuple[int, ...] | None = None
    steps: tuple[int, int] | None = None

    @classmethod
    def from_query(cls, query: dict[str, list[str]]) -> Filters:
        """Parses the filters from a page request's query parameters.

        Args:
            query: Parameters as returned by ``urllib.parse.parse_qs``:
                comma-separated ``labels``, ``ranks`` and ``devices`` (``none``
                selects nothing) and ``steps`` as ``start:end``.

        Returns:
            The filters; absent or empty parameters give None.

        Raises:
            ValueError: If a rank, device or step bound is not an integer.
        """

        def int_tuple(key: str) -> tuple[int, ...] | None:
            raw_value = query.get(key, [""])[0]
            if not raw_value:
                return None
            return tuple(
                int(item)
                for item in raw_value.split(",")
                if item not in ("", "none")
            )

        raw_labels = query.get("labels", [""])[0]
        raw_steps = query.get("steps", [""])[0]
        step_start, _, step_end = raw_steps.partition(":")
        labels: tuple[str, ...] | None
        if raw_labels == "none":
            labels = ()
        elif raw_labels:
            labels = tuple(raw_labels.split(","))
        else:
            labels = None
        ranks = int_tuple("ranks")
        devices = int_tuple("devices")
        steps = (
            (int(step_start or 0), int(step_end))
            if raw_steps and step_end
            else None
        )
        return cls(labels=labels, ranks=ranks, devices=devices, steps=steps)


class LiveSession:
    """The report being explored plus a small cache of recomputed reports.

    Attributes:
        base: The report on the full data.
        theme: Plot theme.
        palette: Palette name or colour list; None for the default.
        base_ctx: Data context of ``base``.
        cards: The plots, built once on the full data.
        skipped: Stem -> reason, for the plots that could not be drawn.
        choices: Filter values the page offers (see
            ``dashboard.filter_choices``).
    """

    def __init__(
        self,
        report: models.ComparisonReport,
        specs: list[spec.PlotSpec] | None,
        *,
        theme: str = "light",
        palette: str | None = None,
    ) -> None:
        """Builds every plot once on the full report.

        Args:
            report: The report to explore.
            specs: The plots; None uses ``dashboard.default_specs``.
            theme: Plot theme, ``"light"`` or ``"dark"``.
            palette: Palette name or colour list; None keeps the default.

        Raises:
            ImportError: If Plotly is not installed.
            spec.NotApplicableError: If no plot could be drawn.
        """
        self.base = report
        self.theme, self.palette = theme, palette
        self.base_ctx = data.DataContext(report)
        self.cards, self.skipped = dashboard.plan(
            self.base_ctx, specs, theme, palette
        )
        self.choices = dashboard.filter_choices(self.base_ctx)
        self._report_cache: collections.OrderedDict[
            Filters, models.ComparisonReport
        ] = collections.OrderedDict()
        self._lock = threading.Lock()

    def report_for(self, filters: Filters) -> models.ComparisonReport:
        """Returns the report for a filter selection, recomputing if needed.

        A selection that covers everything returns the base report.
        Recomputed reports are cached per selection.

        Args:
            filters: The filter selection.

        Returns:
            The report of the selected groups, ranks, devices and steps.

        Raises:
            ValueError: If no group is selected.
        """
        config = self.base.config or models.CompareConfig()
        nsteps = self.choices["nsteps"]
        is_unfiltered = (
            (
                filters.labels is None
                or len(filters.labels) == len(self.base.groups)
            )
            and filters.ranks is None
            and filters.devices is None
        )
        step_start, step_end = filters.steps or (0, nsteps)
        is_unfiltered = (
            is_unfiltered
            and step_start == 0
            and (step_end >= nsteps or not nsteps)
        )
        if is_unfiltered:
            return self.base
        with self._lock:
            if filters in self._report_cache:
                self._report_cache.move_to_end(filters)
                return self._report_cache[filters]
        groups = tuple(
            group
            for group in self.base.groups
            if filters.labels is None or group.label in filters.labels
        )
        if not groups:
            raise ValueError("no group selected")
        sliced_config = dataclasses.replace(
            config,
            ranks=filters.ranks if filters.ranks is not None else config.ranks,
            devices=filters.devices
            if filters.devices is not None
            else config.devices,
            warmup_steps=config.warmup_steps + step_start,
            max_steps=(
                (step_end - step_start)
                if step_end < nsteps
                else config.max_steps
            ),
        )
        # Heavy: runs outside the lock so other cards are not serialized.
        report = engine.run_comparison(groups, sliced_config)
        with self._lock:
            self._report_cache[filters] = report
            while len(self._report_cache) > _MAX_CACHED_REPORTS:
                self._report_cache.popitem(last=False)
        return report

    def plot(self, card: dashboard.Card, filters: Filters) -> dict[str, Any]:
        """Builds one card's figure for a filter selection.

        Args:
            card: The card to draw.
            filters: The filter selection.

        Returns:
            ``{"figure": <figure JSON>, "notes": <notes>}``, or
            ``{"error": <message>}`` if the slice cannot be analysed or the
            plot cannot be drawn from it.
        """
        try:
            report = self.report_for(filters)
            fig_json, _, notes = dashboard.figure_json(
                data.DataContext(report), card.spec, self.theme, self.palette
            )
        except Exception as exc:
            # Shown in the card: an empty slice or an inapplicable plot must
            # not kill the page.
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"figure": fig_json, "notes": notes}

    def html(self, title: str, js: str | None) -> str:
        """Returns the live dashboard page.

        Args:
            title: Page title.
            js: The plotly.js source to embed; None loads it from the CDN.

        Returns:
            The HTML document.
        """
        return dashboard.page(
            title,
            self.cards,
            self.choices,
            js,
            self.theme,
            self.skipped,
            live=True,
        )


def make_server(
    report: models.ComparisonReport,
    specs: list[spec.PlotSpec] | None,
    *,
    port: int = 0,
    title: str = "CTA report",
    theme: str = "light",
    palette: str | None = None,
) -> tuple[server.ThreadingHTTPServer, LiveSession]:
    """Creates the live dashboard server without starting it.

    The server answers ``/`` (the page), ``/api/meta`` (the filter values)
    and ``/api/plot?id=<card>&<filters>`` (one figure for a selection).

    Args:
        report: The report to explore.
        specs: The plots; None uses ``dashboard.default_specs``.
        port: Port to listen on; 0 picks a free one.
        title: Page title.
        theme: Plot theme, ``"light"`` or ``"dark"``.
        palette: Palette name or colour list; None keeps the default.

    Returns:
        The server, bound to 127.0.0.1, and its session.

    Raises:
        ImportError: If Plotly is not installed.
        spec.NotApplicableError: If no plot could be drawn.
    """
    from plotly import offline

    session = LiveSession(report, specs, theme=theme, palette=palette)
    index = session.html(title, offline.get_plotlyjs()).encode("utf-8")

    class Handler(server.BaseHTTPRequestHandler):
        """Serves the page, the filter values and per-selection figures."""

        def log_message(self, *_args: Any) -> None:
            """Keeps the server quiet: no per-request log lines."""

        def _send(
            self, body: bytes, content_type: str, status: int = 200
        ) -> None:
            """Sends a complete response that browsers must not cache."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            """Answers a GET request; unknown paths get a 404."""
            url = parse.urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(index, "text/html; charset=utf-8")
            elif url.path == "/api/meta":
                self._send(
                    json.dumps(session.choices).encode(), "application/json"
                )
            elif url.path == "/api/plot":
                query = parse.parse_qs(url.query)
                try:
                    card = session.cards[int(query["id"][0])]
                    response = session.plot(card, Filters.from_query(query))
                except (KeyError, IndexError, ValueError) as exc:
                    response = {"error": f"bad request: {exc}"}
                self._send(json.dumps(response).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

    # Loopback only: the page can re-run analyses on this machine.
    httpd = server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return httpd, session


def serve(
    report: models.ComparisonReport,
    specs: list[spec.PlotSpec] | None,
    *,
    port: int = 0,
    title: str = "CTA report",
    theme: str = "light",
    palette: str | None = None,
    open_browser: bool = True,
    on_ready: Any = None,
) -> None:
    """Runs the live dashboard until interrupted with Ctrl-C.

    Args:
        report: The report to explore.
        specs: The plots; None uses ``dashboard.default_specs``.
        port: Port to listen on; 0 picks a free one.
        title: Page title.
        theme: Plot theme, ``"light"`` or ``"dark"``.
        palette: Palette name or colour list; None keeps the default.
        open_browser: If True, opens the page in the default browser.
        on_ready: Called with the page URL once the server is bound.

    Raises:
        ImportError: If Plotly is not installed.
        spec.NotApplicableError: If no plot could be drawn.
    """
    httpd, _ = make_server(
        report, specs, port=port, title=title, theme=theme, palette=palette
    )
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
