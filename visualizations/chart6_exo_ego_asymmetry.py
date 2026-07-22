"""Chart 6: Exo vs. ego view asymmetry, binary task, entire, n16 (n8 for VL climbing entire n16 view gap).

Paired bars per model/activity. Flags cases where exo and ego diverge sharply
(e.g. Video-LLaVA climbing n16: 76% exo vs 32% ego) versus the common case
where both views collapse together near baseline.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import AMBER, NAVY, TEAL, accuracy, load, style_ax

FILES = {
    "Qwen — Climbing": "dissertation_v2/results/qwen/qwen_climbing_entire_n16_binary.csv",
    "Qwen — Dance": "diss_dance/results/qwen/qwen_dance_entire_n16_binary.csv",
    "VL — Climbing": "dissertation_v2/results/videollava/vl_climbing_entire_n16_binary.csv",
    "VL — Dance": "diss_dance/results/videollava/vl_dance_entire_n16_binary.csv",
}


def exo_ego_accuracy(relpath):
    df = load(relpath)
    exo = accuracy(df, "exo_correct")[0] if "exo_correct" in df.columns else None
    ego = accuracy(df, "ego_correct")[0] if "ego_correct" in df.columns else None
    return exo, ego


def main():
    labels = list(FILES.keys())
    exo_vals, ego_vals = [], []
    for path in FILES.values():
        exo, ego = exo_ego_accuracy(path)
        exo_vals.append(exo)
        ego_vals.append(ego)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars1 = ax.bar(x - width / 2, exo_vals, width, label="Exo view", color=NAVY)
    bars2 = ax.bar(x + width / 2, ego_vals, width, label="Ego view", color=TEAL)

    ax.axhline(50, color=AMBER, linestyle="--", linewidth=2, label="Random chance (50%)")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%",
                     ha="center", va="bottom", fontsize=9, fontfamily="sans-serif")

    # Flag the large asymmetry case
    for i, label in enumerate(labels):
        gap = abs(exo_vals[i] - ego_vals[i])
        if gap > 20:
            ax.annotate(f"Δ={gap:.0f}pts", xy=(x[i], max(exo_vals[i], ego_vals[i]) + 8),
                        ha="center", fontsize=9, fontweight="bold", color="#B0413E",
                        fontfamily="sans-serif")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontfamily="sans-serif")
    ax.set_ylim(0, 100)
    ax.set_title("Exo vs. Ego View Accuracy — Binary Task (Entire, n16)",
                 fontsize=12.5, fontfamily="sans-serif", fontweight="bold")
    style_ax(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("chart6_exo_ego_asymmetry.png", dpi=300)
    print("Saved chart6_exo_ego_asymmetry.png")
    for label, e, g in zip(labels, exo_vals, ego_vals):
        print(label, f"exo={e:.1f}%", f"ego={g:.1f}%")


if __name__ == "__main__":
    main()
