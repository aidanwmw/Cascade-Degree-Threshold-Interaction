import os
import matplotlib.pyplot as plt

# Shared matplotlib style for the paper figures.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 500,
})

PALETTE = ['#7b00ff', '#2196F3', '#E91E63', '#4CAF50', '#F57C00', '#C62828']
LW = 1.8
REF_LW = 1.2
FIGW_SINGLE = 3.35   # one column in revtex4-2 twocolumn (in)
FIGW_DOUBLE = 6.93   # full text width, \textwidth (in)

OUT = "Paper Figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, stem, **kw):
    fig.savefig(os.path.join(OUT, stem + ".pdf"), **kw)
    fig.savefig(os.path.join(OUT, stem + ".png"), **kw)
