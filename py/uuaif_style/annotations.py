"""
UU Algorithms in Finance - Annotation Helpers

Functions for adding styled annotations, boxes, and highlights to matplotlib
figures, matching the LaTeX tcolorbox styles in uu-aif-beamer.sty.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from typing import Tuple, Optional, Union

from . import colors


def annotation_box(
    ax: Axes,
    text: str,
    xy: Tuple[float, float],
    xytext: Tuple[float, float],
    color: str = "softyellow",
    fontsize: int = 10,
    fontweight: str = "normal",
    arrowprops: Optional[dict] = None,
    **kwargs
) -> plt.Annotation:
    """
    Add an annotation with a colored box background.

    Matches the LaTeX `infobox` style with rounded corners and soft colors.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes to annotate.
    text : str
        The annotation text.
    xy : tuple
        The point (x, y) to annotate.
    xytext : tuple
        The position (x, y) for the text.
    color : str, default "softyellow"
        Box background color. Can be:
        - A key from colors.SOFT: "yellow", "purple", "blue", "coral", "green"
        - Any hex color string
    fontsize : int, default 10
        Font size for the annotation text.
    fontweight : str, default "normal"
        Font weight ("normal", "bold").
    arrowprops : dict, optional
        Arrow properties. If None, uses default styled arrow.
    **kwargs
        Additional arguments passed to ax.annotate().

    Returns
    -------
    Annotation
        The matplotlib Annotation object.

    Examples
    --------
    >>> annotation_box(ax, "Key insight", xy=(5, 10), xytext=(7, 15))
    >>> annotation_box(ax, "Warning!", xy=(5, 10), xytext=(7, 15), color="coral")
    """
    # Resolve color name to hex
    if color in colors.SOFT:
        bgcolor = colors.SOFT[color]
    elif color.startswith("soft"):
        # Handle "softyellow" -> "yellow" mapping
        color_key = color.replace("soft", "")
        bgcolor = colors.SOFT.get(color_key, color)
    else:
        bgcolor = color

    # Determine border color (darker version or matching UU color)
    border_colors = {
        colors.SOFTYELLOW: colors.UUYELLOW,
        colors.SOFTPURPLE: colors.UUPURPLE,
        colors.SOFTBLUE: colors.UUBLUE,
        colors.SOFTCORAL: colors.UURED,
        colors.SOFTGREEN: colors.UUGREEN,
    }
    bordercolor = border_colors.get(bgcolor, colors.DARKBG)

    # Default arrow style
    if arrowprops is None:
        arrowprops = dict(
            arrowstyle="->",
            color=bordercolor,
            lw=1.5,
            connectionstyle="arc3,rad=0.1"
        )

    return ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        fontweight=fontweight,
        color=colors.DARKBG,
        bbox=dict(
            boxstyle="round,pad=0.4,rounding_size=0.2",
            facecolor=bgcolor,
            edgecolor=bordercolor,
            linewidth=1.5,
        ),
        arrowprops=arrowprops,
        **kwargs
    )


def highlight_region(
    ax: Axes,
    x_start: float,
    x_end: float,
    color: str = "softcoral",
    alpha: float = 0.3,
    label: Optional[str] = None,
    **kwargs
) -> mpatches.Polygon:
    """
    Add a vertical highlighted region (axvspan).

    Useful for marking specific ranges, VaR regions, confidence intervals, etc.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    x_start : float
        Start of the highlighted region (x-axis).
    x_end : float
        End of the highlighted region (x-axis).
    color : str, default "softcoral"
        Fill color. Can be a SOFT color key or hex string.
    alpha : float, default 0.3
        Transparency (0-1).
    label : str, optional
        Legend label for the region.
    **kwargs
        Additional arguments passed to ax.axvspan().

    Returns
    -------
    Polygon
        The matplotlib Polygon patch.

    Examples
    --------
    >>> highlight_region(ax, 100, 200, color="coral", label="VaR Region")
    """
    # Resolve color
    if color in colors.SOFT:
        fillcolor = colors.SOFT[color]
    elif color.startswith("soft"):
        color_key = color.replace("soft", "")
        fillcolor = colors.SOFT.get(color_key, color)
    else:
        fillcolor = color

    return ax.axvspan(
        x_start, x_end,
        color=fillcolor,
        alpha=alpha,
        label=label,
        **kwargs
    )


def formula_box(
    ax: Axes,
    text: str,
    position: Tuple[float, float],
    transform: str = "axes",
    fontsize: int = 12,
    **kwargs
) -> plt.Annotation:
    """
    Add a formula box matching LaTeX `formulabox` style.

    Yellow background with yellow border, suitable for mathematical formulas.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    text : str
        The formula/text to display (can include LaTeX math).
    position : tuple
        Position (x, y) for the box.
    transform : str, default "axes"
        Coordinate system: "axes" (0-1), "data", or "figure".
    fontsize : int, default 12
        Font size.
    **kwargs
        Additional arguments passed to ax.annotate().

    Returns
    -------
    Annotation
        The matplotlib Annotation object.

    Examples
    --------
    >>> formula_box(ax, r"$\\sigma = \\sqrt{\\frac{1}{n}\\sum x_i^2}$", (0.5, 0.9))
    """
    transforms = {
        "axes": ax.transAxes,
        "data": ax.transData,
        "figure": ax.figure.transFigure,
    }

    if transform not in transforms:
        raise ValueError(f"Unknown transform '{transform}'. Use: {list(transforms.keys())}")

    return ax.annotate(
        text,
        xy=position,
        xycoords=transforms[transform],
        fontsize=fontsize,
        color=colors.DARKBG,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.6,rounding_size=0.3",
            facecolor=colors.SOFTYELLOW,
            edgecolor=colors.UUYELLOW,
            linewidth=1.5,
        ),
        **kwargs
    )


def add_stat_callout(
    ax: Axes,
    stat: str,
    position: Tuple[float, float],
    label: Optional[str] = None,
    color: str = "softgreen",
    fontsize: int = 24,
    **kwargs
) -> plt.Annotation:
    """
    Add a large statistic callout (AI Now "$1T" style).

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    stat : str
        The statistic to display prominently.
    position : tuple
        Position (x, y) in axes coordinates (0-1).
    label : str, optional
        Small label text below the stat.
    color : str, default "softgreen"
        Background color for the box.
    fontsize : int, default 24
        Font size for the stat.
    **kwargs
        Additional arguments.

    Returns
    -------
    Annotation
        The matplotlib Annotation object.

    Examples
    --------
    >>> add_stat_callout(ax, "$329K", (0.8, 0.8), label="95% VaR")
    """
    # Resolve color
    if color in colors.SOFT:
        bgcolor = colors.SOFT[color]
    else:
        bgcolor = color

    # Main stat
    ann = ax.annotate(
        stat,
        xy=position,
        xycoords="axes fraction",
        fontsize=fontsize,
        fontweight="bold",
        color=colors.DARKBG,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.8,rounding_size=0.4",
            facecolor=bgcolor,
            edgecolor=bgcolor,  # No visible border
            linewidth=0,
        ),
        **kwargs
    )

    # Add label if provided
    if label:
        label_y = position[1] - 0.08
        ax.annotate(
            label,
            xy=(position[0], label_y),
            xycoords="axes fraction",
            fontsize=10,
            color=colors.DARKBG,
            ha="center",
            va="top",
        )

    return ann
