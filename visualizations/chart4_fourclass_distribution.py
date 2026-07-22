"""Chart 4: Fourclass prediction distribution stacked bars.

Ground truth is uniform (25/25/25/25 across Novice/Early Expert/Intermediate
Expert/Late Expert). A stacked bar dominated by one color is direct visual
proof of label collapse.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import load, pred_distribution, views_present

FILES = {
    ("Qwen2.5-VL", "Climbing"): "dissertation_v2/results/qwen/qwen_climbing_entire_n16_fourclass.csv",
    ("Qwen2.5-VL", "Dance"): "diss_dance/results/qwen/qwen_dance_entire_n16_fourclass.csv",
    ("Video-LLaVA", "Climbing"): "dissertation_v2/results/videollava/vl_climbing_entire_n8_fourclass.csv",
    ("Video-LLaVA", "Dance"): "diss_dance/results/videollava/vl_dance_entire_n8_fourclass.csv",
    ("Gemini", "Climbing"): "dissertation_v2/results/gemini/gemini_climbing_entire_fourclass.csv",
    ("Gemini", "Dance"): "diss_dance/results/gemini/gemini_dance_entire_fourclass.csv",
}

CLASS_ORDER = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]
CLASS_COLORS = {
    "Novice": "#1E2761",
    "Early Expert": "#4C6FAF",
    "Intermediate Expert": "#028090",
    "Late Expert": "#EDA100",
}
OTHER_COLOR = "#B0B0B0"


def classify_label(raw):
    r = raw.lower()
    for c in CLASS_ORDER:
        if c.lower() == r or c.lower().replace(" ", "") == r.replace(" ", "").replace("_", ""):
            return c
    if "early" in r:
        return "Early Expert"
    if "intermediate" in r or "int" in r:
        return "Intermediate Expert"
    if "late" in r:
        return "Late Expert"
    if "novice" in r:
        return "Novice"
    return "Other/Unknown"


def get_distribution(relpath):
    df = load(relpath)
    view, pred_col, _ = views_present(df)[0]
    dist = pred_distribution(df, pred_col)
    grouped = {}
    for raw, pct in dist.items():
        cls = classify_label(str(raw))
        grouped[cls] = grouped.get(cls, 0) + pct
    return grouped


def main():
    labels = [f"{m}\n{a}" for m, a in FILES]
    all_classes = CLASS_ORDER + ["Other/Unknown"]

    dists = [get_distribution(path) for path in FILES.values()]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottoms = np.zeros(len(dists))
    x = np.arange(len(dists))

    for cls in all_classes:
        vals = np.array([d.get(cls, 0) for d in dists])
        color = CLASS_COLORS.get(cls, OTHER_COLOR)
        ax.bar(x, vals, bottom=bottoms, label=cls, color=color, width=0.6)
        for i, v in enumerate(vals):
            if v > 6:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8.5, color="white", fontweight="bold", fontfamily="sans-serif")
        bottoms += vals

    ax.axhline(25, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(len(dists) - 0.3, 26, "uniform GT = 25% each", fontsize=8, fontfamily="sans-serif", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, fontfamily="sans-serif")
    ax.set_ylabel("% of predictions", fontsize=11, fontfamily="sans-serif")
    ax.set_ylim(0, 105)
    ax.set_title("Fourclass Prediction Distribution vs. Uniform Ground Truth",
                 fontsize=12.5, fontfamily="sans-serif", fontweight="bold")
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8.5)

    plt.tight_layout()
    plt.savefig("chart4_fourclass_distribution.png", dpi=300)
    print("Saved chart4_fourclass_distribution.png")
    for label, d in zip(labels, dists):
        print(label.replace("\n", " "), d)


if __name__ == "__main__":
    main()
