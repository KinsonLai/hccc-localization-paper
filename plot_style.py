"""Shared matplotlib style for every HCCC figure.

Importing this module (and calling ``setup()``) makes all plots use the same
serif font, the same qualitative colour palette, light grids, and consistent
legend/tick sizing, so the figures look like one coherent set in the paper.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent qualitative palette -- the SAME meaning everywhere:
HCCC = "#1e7d34"   # HCCC / cooperative estimate  -> always GREEN
BASE = "#c0392b"   # baseline / solo PDR / flat coop -> always RED
GREY = "#7f8c8d"   # ablated / neutral variants
GT = "#222222"     # ground truth trajectory


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.9,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.4,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })


def grid(ax):
    ax.grid(True, which="major", ls=":", lw=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)


def legend(ax, loc="best"):
    ax.legend(loc=loc, fontsize=7.5, framealpha=0.95, edgecolor="#888888")
