"""Palettes, size presets and Matplotlib styling for static figures."""

from __future__ import annotations

from typing import Any

OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#333333",
)
TABLEAU = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)
PASTEL = (
    "#8DA0CB",
    "#FC8D62",
    "#66C2A5",
    "#E78AC3",
    "#A6D854",
    "#FFD92F",
    "#E5C494",
    "#B3B3B3",
)
PALETTES: dict[str, tuple[str, ...]] = {
    "okabe_ito": OKABE_ITO,
    "tableau": TABLEAU,
    "pastel": PASTEL,
}

GOOD = "#009E73"  # Improvement.
BAD = "#D55E00"  # Regression.
NEUTRAL = "#888888"  # Reference.

STYLE_SIZES: dict[str, tuple[float, float]] = {
    "paper_single_col": (3.5, 2.4),
    "paper_double_col": (7.16, 3.0),
    "presentation": (6.4, 4.0),
    "web": (8.0, 4.5),
}

PAPER_RC: dict[str, Any] = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
}

DARK_RC: dict[str, Any] = {
    "figure.facecolor": "#111418",
    "axes.facecolor": "#111418",
    "savefig.facecolor": "#111418",
    "axes.edgecolor": "#8a9099",
    "axes.labelcolor": "#d7dae0",
    "text.color": "#d7dae0",
    "xtick.color": "#b0b5bd",
    "ytick.color": "#b0b5bd",
    "grid.color": "#8a9099",
}

# Matplotlib colormap name -> Plotly colorscale name.
PLOTLY_SCALES = {
    "viridis": "Viridis",
    "magma": "Magma",
    "plasma": "Plasma",
    "cividis": "Cividis",
    "inferno": "Inferno",
    "Blues": "Blues",
    "Greens": "Greens",
    "Oranges": "Oranges",
    "Reds": "Reds",
    "Greys": "Greys",
    "RdBu": "RdBu",
    "RdBu_r": "RdBu_r",
    "coolwarm": "RdBu_r",
    "YlOrRd": "YlOrRd",
}


def palette(name: str | None) -> tuple[str, ...]:
    """Returns the colours of a named palette or of an explicit colour list.

    Args:
        name: A key of ``PALETTES``, a comma-separated list of colours (or a
            single ``#rrggbb`` colour), or None for the Okabe-Ito default.

    Returns:
        The palette colours, in drawing order.

    Raises:
        ValueError: If ``name`` is neither a known palette nor a colour list.
    """
    if name is None:
        return OKABE_ITO
    if name in PALETTES:
        return PALETTES[name]
    if name.startswith("#") or "," in name:  # An explicit colour list.
        return tuple(
            color.strip() for color in name.split(",") if color.strip()
        )
    raise ValueError(
        f"Unknown palette '{name}'. Use one of {sorted(PALETTES)} or a "
        "comma-separated colour list"
    )


def rc_for(theme: str) -> dict[str, Any]:
    """Returns the Matplotlib rcParams for a theme.

    Args:
        theme: ``"light"`` (paper style) or ``"dark"`` (paper style on a dark
            background).

    Returns:
        A new dict of rcParams; callers may modify it.

    Raises:
        ValueError: If ``theme`` is neither ``"light"`` nor ``"dark"``.
    """
    if theme == "light":
        return dict(PAPER_RC)
    if theme == "dark":
        return {**PAPER_RC, **DARK_RC}
    raise ValueError(f"Unknown theme '{theme}'. Use 'light' or 'dark'")
