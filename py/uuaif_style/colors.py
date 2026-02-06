"""
UU Algorithms in Finance - Color Definitions

Color palette matching uu-aif-beamer.sty (lines 44-74)
All hex codes are exact matches to the LaTeX package.
"""

# =============================================================================
# PRIMARY UU COLORS
# =============================================================================

UUYELLOW = "#FFCD00"   # Signature color (brand identity)
UUBLACK = "#000000"    # Primary text
UUWHITE = "#FFFFFF"    # Background/contrast
UURED = "#C00A35"      # Accent color (use sparingly)

# =============================================================================
# SECONDARY UU COLORS (for visualizations)
# =============================================================================

UUPURPLE = "#5B2182"
UUBLUE = "#5287C6"
UUGREEN = "#24A793"
UUORANGE = "#F3965E"
UUBURGUNDY = "#AA1555"
UUBROWN = "#6E3B23"
UUDARKBLUE = "#001240"
UUCREAM = "#FFE6AB"     # Brand cream (different from background cream)

# =============================================================================
# BACKGROUND COLORS
# =============================================================================

DARKBG = "#1A1A2E"      # Dark slides background (title, section dividers)
CREAM = "#FAF9F6"       # Content slides background

# =============================================================================
# SOFT PASTELS (for content boxes, highlights)
# =============================================================================

SOFTYELLOW = "#FFF5CC"
SOFTPURPLE = "#E8D4F0"
SOFTBLUE = "#D4E5F7"
SOFTCORAL = "#F5D4D0"
SOFTGREEN = "#D4F0E8"

# =============================================================================
# CIRCUIT BOARD ACCENT
# =============================================================================

CHIPTURQ = "#5CCCC9"

# =============================================================================
# COLOR CYCLE FOR MULTI-LINE PLOTS
# =============================================================================
# Order: blue -> orange -> green -> purple -> red
# Yellow excluded due to low contrast on cream background

COLOR_CYCLE = [
    UUBLUE,
    UUORANGE,
    UUGREEN,
    UUPURPLE,
    UURED,
]

# Extended cycle including more colors
COLOR_CYCLE_EXTENDED = [
    UUBLUE,
    UUORANGE,
    UUGREEN,
    UUPURPLE,
    UURED,
    UUBURGUNDY,
    UUBROWN,
    UUDARKBLUE,
    CHIPTURQ,
]

# =============================================================================
# SLIDE MODE COLORS (light colors for dark backgrounds)
# =============================================================================
# These colors are designed to pop on the dark background (#1A1A2E)

LIGHT_CORAL = "#F5A6A0"     # Soft coral/pink
LIGHT_TEAL = "#7EEAE5"      # Light teal (brighter than CHIPTURQ)
LIGHT_GREEN = "#8FE8B0"     # Soft mint green
LIGHT_BLUE = "#9ECFFF"      # Soft sky blue
LIGHT_PURPLE = "#CBA6F7"    # Soft lavender

# Slide mode color cycle: yellow -> teal -> coral -> green -> blue -> purple
COLOR_CYCLE_SLIDE = [
    UUYELLOW,       # Primary accent
    LIGHT_TEAL,     # Circuit board aesthetic
    LIGHT_CORAL,    # Warm accent
    LIGHT_GREEN,    # Cool accent
    LIGHT_BLUE,     # Secondary cool
    LIGHT_PURPLE,   # Tertiary
]

# =============================================================================
# CONVENIENCE GROUPINGS
# =============================================================================

# All primary colors
PRIMARY = {
    "yellow": UUYELLOW,
    "black": UUBLACK,
    "white": UUWHITE,
    "red": UURED,
}

# All secondary colors
SECONDARY = {
    "purple": UUPURPLE,
    "blue": UUBLUE,
    "green": UUGREEN,
    "orange": UUORANGE,
    "burgundy": UUBURGUNDY,
    "brown": UUBROWN,
    "darkblue": UUDARKBLUE,
    "cream": UUCREAM,
}

# Background colors
BACKGROUNDS = {
    "dark": DARKBG,
    "cream": CREAM,
}

# Soft pastels for boxes/highlights
SOFT = {
    "yellow": SOFTYELLOW,
    "purple": SOFTPURPLE,
    "blue": SOFTBLUE,
    "coral": SOFTCORAL,
    "green": SOFTGREEN,
}
