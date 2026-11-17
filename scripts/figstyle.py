"""Common figure style: Times-metric serif (Nimbus Roman, the URW clone of Times),
black ticks and labels, 1000 dpi, PDF and PNG output."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIF = "Nimbus Roman"
plt.rcParams.update({
    "font.family": "serif", "font.serif": [SERIF, "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": SERIF, "mathtext.it": f"{SERIF}:italic",
    "mathtext.bf": f"{SERIF}:bold", "mathtext.cal": SERIF,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "lines.linewidth": 1.0,
    "savefig.dpi": 1000, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# colour-blind-safe palette (Okabe and Ito)
C = {"bulk": "#D55E00", "ecr": "#0072B2", "neutral": "#009E73", "data": "black",
     "grey": "#7f7f7f", "fit": "#CC79A7", "orange": "#E69F00"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def label(ax, s, x=-0.18, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
