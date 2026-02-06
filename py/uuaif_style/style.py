"""
UU Algorithms in Finance - Matplotlib Style Functions

Main entry point for applying the UU-AIF visual style to matplotlib figures.
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

from . import colors

# Path to the .mplstyle file
_STYLE_FILE = Path(__file__).parent / "uu_aif.mplstyle"


def apply_style(dark_mode: bool = False) -> None:
    """
    Apply the UU-AIF matplotlib style for content slides.

    Parameters
    ----------
    dark_mode : bool, default False
        If True, use dark background (#1A1A2E). If False (default), use
        cream background (#FAF9F6) for content slides.

    Examples
    --------
    >>> from uuaif_style import apply_style
    >>> apply_style()  # Cream background (default)
    >>> apply_style(dark_mode=True)  # Dark background

    Note: For the signature dark slide aesthetic matching uu-aif-beamer.sty,
    use slide_mode() instead.
    """
    # Apply base style from .mplstyle file
    plt.style.use(str(_STYLE_FILE))

    if dark_mode:
        # Override for dark mode (section divider slides)
        plt.rcParams.update({
            "figure.facecolor": colors.DARKBG,
            "figure.edgecolor": colors.DARKBG,
            "axes.facecolor": colors.DARKBG,
            "axes.edgecolor": colors.UUYELLOW,
            "axes.labelcolor": colors.UUWHITE,
            "xtick.color": colors.UUWHITE,
            "ytick.color": colors.UUWHITE,
            "grid.color": colors.UUWHITE,
            "grid.alpha": 0.2,
            "text.color": colors.UUWHITE,
            "legend.facecolor": colors.DARKBG,
            "legend.edgecolor": colors.UUYELLOW,
            "savefig.facecolor": colors.DARKBG,
        })


def slide_mode() -> None:
    """
    Apply the signature UU-AIF dark slide style matching uu-aif-beamer.sty.

    This creates figures that match the dark background aesthetic of the
    Beamer template with:
    - Dark background (#1A1A2E)
    - Yellow titles and accents (#FFCD00)
    - Light color palette optimized for dark backgrounds
    - Clean, modern styling

    Examples
    --------
    >>> from uuaif_style import slide_mode, colors
    >>> slide_mode()
    >>> plt.plot(x, y)  # Uses yellow as first color
    >>> plt.savefig('figure.pdf')
    """
    # Start with base style
    plt.style.use(str(_STYLE_FILE))

    # Apply dark slide aesthetic
    plt.rcParams.update({
        # Background
        "figure.facecolor": colors.DARKBG,
        "figure.edgecolor": colors.DARKBG,
        "axes.facecolor": colors.DARKBG,

        # Borders and spines
        "axes.edgecolor": colors.UUYELLOW,
        "axes.linewidth": 1.2,

        # Text colors
        "axes.labelcolor": colors.UUWHITE,
        "text.color": colors.UUWHITE,

        # Ticks
        "xtick.color": colors.UUWHITE,
        "ytick.color": colors.UUWHITE,

        # Grid - subtle
        "axes.grid": True,
        "grid.color": colors.UUWHITE,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,

        # Color cycle for slide mode (light colors on dark bg)
        "axes.prop_cycle": plt.cycler('color', colors.COLOR_CYCLE_SLIDE),

        # Legend
        "legend.facecolor": colors.DARKBG,
        "legend.edgecolor": colors.UUYELLOW,

        # Save settings
        "savefig.facecolor": colors.DARKBG,
        "savefig.edgecolor": "none",
    })


def set_transparent_background() -> None:
    """
    Set transparent background for LaTeX embedding.

    Use this after apply_style() when you want the figure to have no
    background color (transparent) for clean embedding in LaTeX slides.

    Examples
    --------
    >>> from uuaif_style import apply_style, set_transparent_background
    >>> apply_style()
    >>> set_transparent_background()
    >>> # ... create plot ...
    >>> plt.savefig('figure.pdf', transparent=True)
    """
    plt.rcParams.update({
        "figure.facecolor": "none",
        "figure.edgecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
    })


def configure_for_beamer(aspect_ratio: str = "16:9") -> None:
    """
    Configure figure size for Beamer slides.

    Parameters
    ----------
    aspect_ratio : str, default "16:9"
        Target aspect ratio. Options: "16:9", "4:3", "16:10"

    Examples
    --------
    >>> from uuaif_style import apply_style, configure_for_beamer
    >>> apply_style()
    >>> configure_for_beamer("16:9")
    """
    aspect_ratios = {
        "16:9": (10, 5.625),   # Standard widescreen
        "4:3": (8, 6),         # Classic
        "16:10": (10, 6.25),   # Widescreen variant
    }

    if aspect_ratio not in aspect_ratios:
        raise ValueError(
            f"Unknown aspect ratio '{aspect_ratio}'. "
            f"Choose from: {list(aspect_ratios.keys())}"
        )

    plt.rcParams["figure.figsize"] = aspect_ratios[aspect_ratio]


def get_color_cycle() -> list:
    """
    Return the UU-AIF color cycle for manual use.

    Returns
    -------
    list
        List of hex color strings in cycle order.

    Examples
    --------
    >>> from uuaif_style import get_color_cycle
    >>> colors = get_color_cycle()
    >>> for i, color in enumerate(colors):
    ...     plt.plot(x, y[i], color=color)
    """
    return colors.COLOR_CYCLE.copy()


def reset_style() -> None:
    """
    Reset matplotlib to default style.

    Useful for returning to defaults after using UU-AIF style.
    """
    plt.style.use("default")
