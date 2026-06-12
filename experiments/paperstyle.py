# Paper-style matplotlib config shared by all experiments.
# Rules: serif fonts, inward ticks, no top/right spines, grayscale plus one accent,
#        no in-figure titles (captions live in the README as Figure N text), 300 dpi.

import matplotlib as mpl

INK = "#1a1a1a"       # body black
GRAY = "#9a9a9a"      # secondary gray
ACCENT = "#3a5fa0"    # accent (navy)
ACCENT2 = "#a04040"   # secondary accent (brick)
LIGHT = "#dfe6f0"     # shaded regions


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
    })


def panel_label(ax, label):
    """Put an (a)/(b) panel label at the top-left of an axes."""
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="bottom", ha="right")
