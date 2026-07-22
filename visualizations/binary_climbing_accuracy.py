"""Compare binary (Novice vs Expert) accuracy for climbing, exo view, across models."""

import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#1E2761"
TEAL = "#028090"
AMBER = "#EDA100"

RESULTS_DIR = "../dissertation_v2/results"

MODELS = {
    "Qwen2.5-VL": (f"{RESULTS_DIR}/qwen/qwen_climbing_entire_n16_binary.csv", "exo_correct"),
    "Video-LLaVA": (f"{RESULTS_DIR}/videollava/vl_climbing_entire_n16_binary.csv", "exo_correct"),
    "Gemini": (f"{RESULTS_DIR}/gemini/gemini_climbing_entire_binary.csv", "correct"),
}


def compute_accuracy(path, correct_col):
    df = pd.read_csv(path)
    return df[correct_col].astype(bool).mean() * 100


def main():
    accuracies = {model: compute_accuracy(path, col) for model, (path, col) in MODELS.items()}

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(accuracies.keys(), accuracies.values(), color=[NAVY, TEAL, NAVY], width=0.5)

    ax.axhline(50, color=AMBER, linestyle="--", linewidth=2, label="Random chance (50%)")

    for bar, acc in zip(bars, accuracies.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontfamily="sans-serif")

    ax.set_ylabel("Accuracy (%)", fontsize=11, fontfamily="sans-serif")
    ax.set_title("Binary Skill Classification Accuracy — Climbing (Exo View)",
                 fontsize=13, fontfamily="sans-serif", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=False, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("binary_climbing_accuracy.png", dpi=300)
    print("Saved binary_climbing_accuracy.png")
    for model, acc in accuracies.items():
        print(f"{model}: {acc:.2f}%")


if __name__ == "__main__":
    main()
