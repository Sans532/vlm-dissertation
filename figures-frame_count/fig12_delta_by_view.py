"""fig12: entire->trimmed accuracy delta, split by view (ego/exo).

Diverging bar chart, one row per activity x model x prompt x frame-count
condition that has a valid entire AND trimmed run on BOTH views. Two bars
per row (ego delta above, exo delta below), colored by sign, so opposite-
direction shifts between views are visible at a glance.

Reads figures-frame_count/master_summary.csv. Writes PNG (300dpi) + PDF.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

UP = "#1baf7a"      # improvement on trimming
DOWN = "#d1495b"    # regression on trimming
INK = "#0b0b0b"
INK2 = "#52514e"
GRAY = "#9aa0a6"
MODEL_NAME = {"qwen": "Qwen2.5-VL", "videollava": "Video-LLaVA"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main():
    df = pd.read_csv(os.path.join(OUT, "master_summary.csv"))
    df["frames"] = df["frames"].astype(float)
    idx = {}
    for _, r in df.iterrows():
        idx[(r.activity, r.model, r.prompt, r.frames, r.condition, r.view)] = r.acc

    combos = sorted(set(
        (r.activity, r.model, r.prompt, r.frames)
        for _, r in df.iterrows()
    ))

    rows = []
    for act, model, prompt, frames in combos:
        if (act, model, prompt, frames) == ("climbing", "videollava", "binary", 16.0):
            continue
        aee = idx.get((act, model, prompt, frames, "entire", "ego"))
        aex = idx.get((act, model, prompt, frames, "entire", "exo"))
        ate = idx.get((act, model, prompt, frames, "trimmed", "ego"))
        atx = idx.get((act, model, prompt, frames, "trimmed", "exo"))
        if pd.isna(aee) or pd.isna(aex) or pd.isna(ate) or pd.isna(atx):
            continue
        d_ego, d_exo = ate - aee, atx - aex
        if abs(d_ego) < 0.01 and abs(d_exo) < 0.01:
            continue
        label = f"{act[:4]}/{MODEL_NAME[model]}/{prompt}/{int(frames)}f"
        rows.append((label, d_ego, d_exo))

    # sort by max abs delta so the largest swings anchor top/bottom
    rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])))

    fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(rows) + 1.6))
    y = list(range(len(rows)))
    bar_h = 0.34

    for i, (label, d_ego, d_exo) in enumerate(rows):
        ax.barh(i + bar_h / 2 + 0.02, d_ego, height=bar_h,
                color=UP if d_ego >= 0 else DOWN, edgecolor="white", linewidth=0.5, zorder=3)
        ax.barh(i - bar_h / 2 - 0.02, d_exo, height=bar_h,
                color=UP if d_exo >= 0 else DOWN, alpha=0.55, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(d_ego + (0.4 if d_ego >= 0 else -0.4), i + bar_h / 2 + 0.02, f"{d_ego:+.1f}",
                fontsize=7.5, va="center", ha="left" if d_ego >= 0 else "right", color=INK2)
        ax.text(d_exo + (0.4 if d_exo >= 0 else -0.4), i - bar_h / 2 - 0.02, f"{d_exo:+.1f}",
                fontsize=7.5, va="center", ha="left" if d_exo >= 0 else "right", color=INK2)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.axvline(0, color=INK2, linewidth=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)
    xmax = max(abs(min(min(r[1], r[2]) for r in rows)), abs(max(max(r[1], r[2]) for r in rows))) + 4
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("Accuracy change, trimmed − entire (percentage points)")

    handles = [
        Patch(facecolor=UP, label="improves on trimming"),
        Patch(facecolor=DOWN, label="regresses on trimming"),
        Patch(facecolor=INK2, label="top bar = ego, bottom bar (faint) = exo"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -1.4 / len(rows)), ncol=1, fontsize=8.5, handlelength=1.2)
    ax.set_title("Entire→trimmed accuracy shift often reverses sign between ego and exo", pad=14, fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig12_delta_by_view")


if __name__ == "__main__":
    main()
