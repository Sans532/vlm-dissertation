"""Chart 2: Binary accuracy vs. random-chance baseline, both activities, exo/single view."""

import matplotlib.pyplot as plt
import numpy as np

from common import AMBER, NAVY, TEAL, accuracy, load, style_ax, views_present

FILES = {
    ("Qwen2.5-VL", "Climbing"): "dissertation_v2/results/qwen/qwen_climbing_entire_n16_binary.csv",
    ("Qwen2.5-VL", "Dance"): "diss_dance/results/qwen/qwen_dance_entire_n16_binary.csv",
    ("Video-LLaVA", "Climbing"): "dissertation_v2/results/videollava/vl_climbing_entire_n16_binary.csv",
    ("Video-LLaVA", "Dance"): "diss_dance/results/videollava/vl_dance_entire_n16_binary.csv",
    ("Gemini", "Climbing"): "dissertation_v2/results/gemini/gemini_climbing_entire_binary.csv",
    ("Gemini", "Dance"): "diss_dance/results/gemini/gemini_dance_entire_binary.csv",
}

MODELS = ["Qwen2.5-VL", "Video-LLaVA", "Gemini"]
ACTIVITIES = ["Climbing", "Dance"]


def exo_or_single_accuracy(df):
    views = views_present(df)
    for view, _pred, corr in views:
        if view in ("exo", "single"):
            return accuracy(df, corr)[0]
    return None


def main():
    data = {}
    for (model, activity), relpath in FILES.items():
        df = load(relpath)
        data[(model, activity)] = exo_or_single_accuracy(df)

    x = np.arange(len(MODELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    climbing_vals = [data[(m, "Climbing")] for m in MODELS]
    dance_vals = [data[(m, "Dance")] for m in MODELS]

    bars1 = ax.bar(x - width / 2, climbing_vals, width, label="Climbing", color=NAVY)
    bars2 = ax.bar(x + width / 2, dance_vals, width, label="Dance", color=TEAL)

    ax.axhline(50, color=AMBER, linestyle="--", linewidth=2, label="Random chance (50%)")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=9, fontfamily="sans-serif")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10, fontfamily="sans-serif")
    ax.set_ylim(0, 100)
    ax.set_title("Binary Accuracy vs. Random Chance — Climbing vs. Dance (Exo/Single View)",
                 fontsize=12.5, fontfamily="sans-serif", fontweight="bold")
    style_ax(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("chart2_accuracy_vs_chance.png", dpi=300)
    print("Saved chart2_accuracy_vs_chance.png")
    for k, v in data.items():
        print(k, f"{v:.2f}%")


if __name__ == "__main__":
    main()
