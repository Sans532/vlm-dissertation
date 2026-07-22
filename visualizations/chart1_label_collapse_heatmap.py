"""Chart 1: Label collapse heatmap.

For binary tasks (entire, n16 where available), show % of predictions that are
"Novice" for every model x activity x view combination. Values near 100% mean
the model is defaulting to one label regardless of ground truth.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import BASE, load, pred_distribution, views_present

FILES = {
    ("Qwen2.5-VL", "Climbing"): "dissertation_v2/results/qwen/qwen_climbing_entire_n16_binary.csv",
    ("Qwen2.5-VL", "Dance"): "diss_dance/results/qwen/qwen_dance_entire_n16_binary.csv",
    ("Video-LLaVA", "Climbing"): "dissertation_v2/results/videollava/vl_climbing_entire_n16_binary.csv",
    ("Video-LLaVA", "Dance"): "diss_dance/results/videollava/vl_dance_entire_n16_binary.csv",
    ("Gemini", "Climbing"): "dissertation_v2/results/gemini/gemini_climbing_entire_binary.csv",
    ("Gemini", "Dance"): "diss_dance/results/gemini/gemini_dance_entire_binary.csv",
}


def novice_pct(df, pred_col):
    dist = pred_distribution(df, pred_col)
    return next((v for k, v in dist.items() if "novice" in k.lower()), 0.0)


def main():
    rows = []  # (model, activity, view, novice_pct)
    for (model, activity), relpath in FILES.items():
        df = load(relpath)
        for view, pred_col, _corr_col in views_present(df):
            rows.append((model, activity, view, novice_pct(df, pred_col)))

    col_labels = [f"{a}\n({v})" for _, a, v, _ in rows]
    # dedupe columns preserving order of (activity, view)
    seen = []
    for _, a, v, _ in rows:
        key = (a, v)
        if key not in seen:
            seen.append(key)
    models = ["Qwen2.5-VL", "Video-LLaVA", "Gemini"]

    matrix = np.full((len(models), len(seen)), np.nan)
    for model, activity, view, pct in rows:
        i = models.index(model)
        j = seen.index((activity, view))
        matrix[i, j] = pct

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.cm.get_cmap("YlOrRd")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(seen)))
    ax.set_xticklabels([f"{a}\n({v})" for a, v in seen], fontsize=10, fontfamily="sans-serif")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10, fontfamily="sans-serif")

    for i in range(len(models)):
        for j in range(len(seen)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 55 else "#1E2761"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color, fontfamily="sans-serif")

    ax.set_title("Binary Prediction Collapse to \"Novice\" (Entire, n16)",
                 fontsize=13, fontfamily="sans-serif", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% predictions = \"Novice\"", fontsize=10, fontfamily="sans-serif")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(seen), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig("chart1_label_collapse_heatmap.png", dpi=300)
    print("Saved chart1_label_collapse_heatmap.png")
    for model, activity, view, pct in rows:
        print(f"{model:15s} {activity:10s} {view:6s} Novice%={pct:.1f}")


if __name__ == "__main__":
    main()
