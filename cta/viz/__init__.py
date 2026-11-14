"""Visualisation: plot kinds over report tables and raw trace tables.

Plots render as static (Matplotlib) or interactive (Plotly) output. Importing
this package does not import Matplotlib or Plotly; they load when a plot is
drawn.
"""

from cta.viz.plots import build_chart
from cta.viz.plots import render_chart
from cta.viz.plots import render_plot
from cta.viz.plots import render_spec
from cta.viz.spec import kind_catalog
from cta.viz.spec import load_specs
from cta.viz.spec import PlotSpec
from cta.viz.spec import register_plot

__all__ = [
    "PlotSpec",
    "build_chart",
    "kind_catalog",
    "load_specs",
    "register_plot",
    "render_chart",
    "render_plot",
    "render_spec",
]
