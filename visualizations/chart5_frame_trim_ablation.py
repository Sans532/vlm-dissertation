"""Chart 5: Frame count / trimming ablation.

Binary accuracy (exo view) for Qwen and Video-LLaVA across four conditions:
entire-n8, entire-n16, trimmed-n8, trimmed-n16. Flat lines across conditions
show accuracy is invariant to these input manipulations.
"""

import matplotlib.pyplot as plt

from common import AMBER, NAVY, TEAL, accuracy, load, style_ax

CONDITIONS = ["Entire n8", "Entire n16", "Trimmed n8", "Trimmed n16"]

FILES = {
    "Qwen2.5-VL — Climbing": [
        "dissertation_v2/results/qwen/qwen_climbing_entire_n8_binary.csv",
        "dissertation_v2/results/qwen/qwen_climbing_entire_n16_binary.csv",
        "dissertation_v2/results/qwen/trimmed/qwen_climbing_trimmed_exo_n8_binary.csv",
        "dissertation_v2/results/qwen/trimmed/qwen_climbing_trimmed_exo_n16_binary.csv",
    ],
    "Qwen2.5-VL — Dance": [
        "diss_dance/results/qwen/qwen_dance_entire_n8_binary.csv",
        "diss_dance/results/qwen/qwen_dance_entire_n16_binary.csv",
        "diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_binary.csv",
        "diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_binary.csv",
    ],
    "Video-LLaVA — Climbing": [
        "dissertation_v2/results/videollava/vl_climbing_entire_n8_binary.csv",
        "dissertation_v2/results/videollava/vl_climbing_entire_n16_binary.csv",
        "dissertation_v2/results/videollava/trimmed/vl_climbing_trimmed_exo_n8_binary.csv",
        "dissertation_v2/results/videollava/trimmed/vl_climbing_trimmed_exo_n16_binary.csv",
    ],
    "Video-LLaVA — Dance": [
        "diss_dance/results/videollava/vl_dance_entire_n8_binary.csv",
        "diss_dance/results/videollava/vl_dance_entire_n16_binary.csv",
        "diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_binary.csv",
        "diss_dance/results/videollava/trimmed/vl_dance_trimmed_n16_binary.csv",
    ],
}

LINE_STYLE = {
    "Qwen2.5-VL — Climbing": dict(color=NAVY, linestyle="-", marker="o"),
    "Qwen2.5-VL — Dance": dict(color=NAVY, linestyle="--", marker="D"),
    "Video-LLaVA — Climbing": dict(color=TEAL, linestyle="-", marker="s"),
    "Video-LLaVA — Dance": dict(color=TEAL, linestyle="--", marker="^"),
}

# Small vertical jitter so overlapping flat-50% lines remain visually distinguishable.
JITTER = {
    "Qwen2.5-VL — Climbing": 0.0,
    "Qwen2.5-VL — Dance": 0.6,
    "Video-LLaVA — Climbing": 0.0,
    "Video-LLaVA — Dance": -0.6,
}


def exo_accuracy(relpath):
    df = load(relpath)
    for candidate in ["exo_correct", "binary_exo_correct", "correct"]:
        if candidate in df.columns:
            return accuracy(df, candidate)[0]
    raise KeyError(f"no correctness column found in {relpath}: {list(df.columns)}")


def main():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = range(len(CONDITIONS))

    results = {}
    for series_name, paths in FILES.items():
        vals = [exo_accuracy(p) for p in paths]
        results[series_name] = vals
        plotted = [v + JITTER[series_name] for v in vals]
        ax.plot(x, plotted, label=series_name, linewidth=2, markersize=7, **LINE_STYLE[series_name])
        for xi, (v, p) in enumerate(zip(vals, plotted)):
            if abs(p - 50) < 3:
                continue
            ax.annotate(f"{v:.0f}%", (xi, p), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8.5, fontfamily="sans-serif")

    ax.axhline(50, color=AMBER, linestyle=":", linewidth=2, label="Random chance (50%)", zorder=0)
    ax.text(0.02, 0.30, "Note: Qwen and VL-Dance lines are jittered ±0.6pts\nfor visibility — all sit exactly at 50% accuracy.",
            transform=ax.transAxes, fontsize=8, fontfamily="sans-serif", color="#666",
            va="bottom", ha="left")

    ax.set_xticks(list(x))
    ax.set_xticklabels(CONDITIONS, fontsize=10, fontfamily="sans-serif")
    ax.set_ylim(0, 100)
    ax.set_title("Binary Accuracy Across Frame Count / Trimming Conditions (Exo View)",
                 fontsize=12.5, fontfamily="sans-serif", fontweight="bold")
    style_ax(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, ncol=1)

    plt.tight_layout()
    plt.savefig("chart5_frame_trim_ablation.png", dpi=300)
    print("Saved chart5_frame_trim_ablation.png")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()
