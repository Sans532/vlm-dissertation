"""LaTeX-ready variant of vfig02 (binary-task recall plane).

Same data and geometry as vfig2() in make_figures_video.py, but rebuilt for
inclusion in a thesis: serif type at the document's body size, no baked-in
title or footnote (those belong in \\caption{}), sized to a standard
textwidth, and written as a vector PDF with Type-42 fonts embedded.

    python figures-video/make_vfig02_latex.py
    -> figures-video/vfig02_binary_discrimination_latex.pdf (+ .png proof)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures_video import (AQUA, CHANCE, GRAY, INK, INK2, MODEL_COLOR,
                                MODEL_NAME, VIDEO_MODELS, load, style)

OUT = os.path.dirname(os.path.abspath(__file__))
NAME = "vfig02_binary_discrimination_latex"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed real fonts, not paths
})


def build(df):
    b = df[(df.prompt == "binary") & df.valid & df.model.isin(VIDEO_MODELS)]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))          # fits \textwidth
    ax.fill_between([0, 100], [100, 0], 100, color=AQUA, alpha=0.06, zorder=0)
    ax.plot([0, 100], [100, 0], color=CHANCE, linestyle="--", linewidth=1.2, zorder=2)

    for m in VIDEO_MODELS:
        for a, mk in (("climbing", "o"), ("dance", "^")):
            for v in ("exo", "ego"):
                g = b[(b.model == m) & (b.activity == a) & (b.view == v)]
                if not len(g):
                    continue
                rn = 100 * (g[g["gt"] == "Novice"]["pred"] == "Novice").mean()
                re = 100 * (g[g["gt"] == "Expert"]["pred"] == "Expert").mean()
                ax.scatter(rn, re, s=70, marker=mk, color=MODEL_COLOR[m], alpha=0.9,
                           edgecolors="white", linewidths=1.0, zorder=4)
                ax.annotate(v, xy=(rn, re), xytext=(4, -9), textcoords="offset points",
                            fontsize=6.5, color=INK2)

    ax.scatter([], [], s=70, marker="o", color=GRAY, label="climbing")
    ax.scatter([], [], s=70, marker="^", color=GRAY, label="dance")
    for m in VIDEO_MODELS:
        ax.scatter([], [], s=70, marker="s", color=MODEL_COLOR[m], label=MODEL_NAME[m])

    ax.set_xlim(-3, 108)
    ax.set_ylim(-8, 108)
    ax.annotate("always answers “Novice”\n(accuracy = 50%, no skill)", xy=(99.5, 1),
                xytext=(24, 15), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(52, 92, "better than chance", fontsize=7.5, color=INK2, style="italic")
    ax.text(10, 32, "worse than chance", fontsize=7.5, color=INK2, style="italic")
    ax.text(4, 46, "chance line\n(accuracy = 50%)", fontsize=6.5, color=CHANCE, rotation=-38)
    style(ax, ylabel="Recall on Expert clips (%)", xlabel="Recall on Novice clips (%)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2,
              handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    return fig


if __name__ == "__main__":
    fig = build(load())
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{NAME}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", NAME)
