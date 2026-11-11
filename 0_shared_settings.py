"""
0_shared_settings.py  --  the few constants and helpers that more than one file needs
======================================================================================

Read this file first.  Every other file imports these names instead of defining its own copy,
so a number is written down in exactly one place.

    N0           reference network size of all scaling laws:  s(N) = s0 * (N / N0) ** p
    SQRT2        sqrt(2), the saturation value of the error E_rms (two unrelated unit vectors are sqrt(2) apart)
    E_CUT        points with a simulated E_rms above this value are "on the plateau" (E_rms -> sqrt(2));
                 fit metrics are also reported without them, and the validation excludes them from its metrics
    fmt(v)       number -> short text for labels,              0.5 -> '0.5'
    tag(v)       number -> text that is safe in file names,    0.5 -> '0p5'
    set_paper_style()   matplotlib look of the manuscript figures (serif fonts, inward ticks)
    label_panel(ax, letter)   bold panel letter "(a)" inside an axes
"""
import matplotlib
import numpy as np

N0 = 100
SQRT2 = np.sqrt(2.0)
E_CUT = 1.2


def fmt(v):
    """Number -> short text for labels, e.g. 0.5 -> '0.5'."""
    return f"{v:g}"


def tag(v):
    """Number -> text usable in file names, e.g. 0.5 -> '0p5'."""
    return f"{v:g}".replace(".", "p")


def set_paper_style():
    """Look of the manuscript figures: serif fonts, ticks pointing inwards on all four sides, legends without frame."""
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                                "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.7,
                                "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                                "legend.frameon": False})


def label_panel(ax, letter, x=0.03, ha="left"):
    """Write the panel letter, e.g. (a), in bold into the top corner of the axes (x = 0.03 left, x = 0.97 with ha = 'right')."""
    ax.text(x, 0.95, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha=ha)
