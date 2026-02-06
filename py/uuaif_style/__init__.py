"""
UU Algorithms in Finance - Matplotlib Style Package

A Python styling package that mirrors uu-aif-beamer.sty for visual
consistency between LaTeX lecture slides and Python-generated figures.

Usage
-----
>>> from uuaif_style import apply_style, colors
>>> apply_style()
>>> plt.plot(x, y, color=colors.UUBLUE)
>>> plt.savefig('assets/graphs/figure.pdf')

For dark backgrounds (section divider slides):
>>> apply_style(dark_mode=True)

For transparent backgrounds (LaTeX embedding):
>>> apply_style()
>>> set_transparent_background()

Available Colors
----------------
Primary: UUYELLOW, UURED, UUBLACK, UUWHITE
Secondary: UUBLUE, UUGREEN, UUPURPLE, UUORANGE, UUBURGUNDY, UUBROWN, UUDARKBLUE
Backgrounds: DARKBG, CREAM
Soft pastels: SOFTYELLOW, SOFTBLUE, SOFTGREEN, SOFTPURPLE, SOFTCORAL
"""

from .colors import (
    # Primary
    UUYELLOW,
    UUBLACK,
    UUWHITE,
    UURED,
    # Secondary
    UUPURPLE,
    UUBLUE,
    UUGREEN,
    UUORANGE,
    UUBURGUNDY,
    UUBROWN,
    UUDARKBLUE,
    UUCREAM,
    # Backgrounds
    DARKBG,
    CREAM,
    # Soft pastels
    SOFTYELLOW,
    SOFTPURPLE,
    SOFTBLUE,
    SOFTCORAL,
    SOFTGREEN,
    # Circuit accent
    CHIPTURQ,
    # Light colors for dark backgrounds
    LIGHT_CORAL,
    LIGHT_TEAL,
    LIGHT_GREEN,
    LIGHT_BLUE,
    LIGHT_PURPLE,
    # Cycles
    COLOR_CYCLE,
    COLOR_CYCLE_EXTENDED,
    COLOR_CYCLE_SLIDE,
    # Groupings
    PRIMARY,
    SECONDARY,
    BACKGROUNDS,
    SOFT,
)

from .style import (
    apply_style,
    slide_mode,
    set_transparent_background,
    configure_for_beamer,
    get_color_cycle,
    reset_style,
)

from .annotations import (
    annotation_box,
    highlight_region,
    formula_box,
    add_stat_callout,
)

# Also expose the colors module for attribute access
from . import colors

__version__ = "1.0.0"
__all__ = [
    # Style functions
    "apply_style",
    "slide_mode",
    "set_transparent_background",
    "configure_for_beamer",
    "get_color_cycle",
    "reset_style",
    # Annotation functions
    "annotation_box",
    "highlight_region",
    "formula_box",
    "add_stat_callout",
    # Colors module
    "colors",
    # Individual colors (for convenience)
    "UUYELLOW",
    "UUBLACK",
    "UUWHITE",
    "UURED",
    "UUPURPLE",
    "UUBLUE",
    "UUGREEN",
    "UUORANGE",
    "UUBURGUNDY",
    "UUBROWN",
    "UUDARKBLUE",
    "UUCREAM",
    "DARKBG",
    "CREAM",
    "SOFTYELLOW",
    "SOFTPURPLE",
    "SOFTBLUE",
    "SOFTCORAL",
    "SOFTGREEN",
    "CHIPTURQ",
    "LIGHT_CORAL",
    "LIGHT_TEAL",
    "LIGHT_GREEN",
    "LIGHT_BLUE",
    "LIGHT_PURPLE",
    "COLOR_CYCLE",
    "COLOR_CYCLE_EXTENDED",
    "COLOR_CYCLE_SLIDE",
    "PRIMARY",
    "SECONDARY",
    "BACKGROUNDS",
    "SOFT",
]
