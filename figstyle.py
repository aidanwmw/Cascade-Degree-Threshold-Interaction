import os
import matplotlib.pyplot as plt

# Shared matplotlib style for the paper figures (mirrors figure 1).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
})

# figure-1 colour family, extended to six for the multi-curve panels.
COL = ['#6B2FFF', '#1A8FE3', '#E8406B', '#3D9A3D', '#F5A623', '#00ACC1']
PALETTE = COL
LW = 2.0
REF_LW = 1.2
FIGW_SINGLE = 3.35   # one column in revtex4-2 twocolumn (in)
FIGW_DOUBLE = 7.3    # full text width (in)

OUT = "Paper Figures"
os.makedirs(OUT, exist_ok=True)


def panel_label(ax, letter):
    # matches figure 1: inside the axes, top-left, base font
    ax.text(0.04, 0.94, f"({letter})", transform=ax.transAxes,
            fontweight="bold", va="top", ha="left")


def save(fig, stem, **kw):
    kw.setdefault("bbox_inches", "tight")
    fig.savefig(os.path.join(OUT, stem + ".pdf"), **kw)
    fig.savefig(os.path.join(OUT, stem + ".png"), **kw)
