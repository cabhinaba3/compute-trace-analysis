"""Visualisation: plot kinds over report tables and raw traces, static (Matplotlib) and interactive (Plotly) output.

Importing this package does not import Matplotlib or Plotly; they load when a plot is drawn.
"""

from cta.viz.plots import build_chart, render_chart, render_plot, render_spec
from cta.viz.spec import PlotSpec, kind_catalog, load_specs, register_plot

__all__ = ["PlotSpec", "build_chart", "kind_catalog", "load_specs", "register_plot", "render_chart", "render_plot", "render_spec"]
