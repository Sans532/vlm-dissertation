"""Chart 3: Cross-domain consistency scatter.

Climbing accuracy (x) vs dance accuracy (y) for each model x prompt-type,
using the exo/single view, entire, n16 (or n8 where n16 unavailable).
Baselines: 50% for binary, 25% for fourclass/structured/reasoning.
Points clustering near the diagonal AND near baseline show the failure
pattern is domain-independent.
"""

import matplotlib.pyplot as plt

from common import AMBER, NAVY, TEAL, GREY, accuracy, load, style_ax, views_present

# (model, prompt) -> (climbing_path, dance_path, baseline)
FILES = {
    ("Qwen2.5-VL", "binary"): (
        "dissertation_v2/results/qwen/qwen_climbing_entire_n16_binary.csv",
        "diss_dance/results/qwen/qwen_dance_entire_n16_binary.csv", 50),
    ("Qwen2.5-VL", "fourclass"): (
        "dissertation_v2/results/qwen/qwen_climbing_entire_n16_fourclass.csv",
        "diss_dance/results/qwen/qwen_dance_entire_n16_fourclass.csv", 25),
    ("Qwen2.5-VL", "structured"): (
        "dissertation_v2/results/qwen/qwen_climbing_entire_n16_structured.csv",
        "diss_dance/results/qwen/qwen_dance_entire_n16_structured.csv", 25),
    ("Qwen2.5-VL", "reasoning"): (
        "dissertation_v2/results/qwen/qwen_climbing_entire_n16_reasoning.csv",
        "diss_dance/results/qwen/qwen_dance_entire_n16_reasoning.csv", 25),
    ("Video-LLaVA", "binary"): (
        "dissertation_v2/results/videollava/vl_climbing_entire_n16_binary.csv",
        "diss_dance/results/videollava/vl_dance_entire_n16_binary.csv", 50),
    ("Video-LLaVA", "fourclass"): (
        "dissertation_v2/results/videollava/vl_climbing_entire_n8_fourclass.csv",
        "diss_dance/results/videollava/vl_dance_entire_n8_fourclass.csv", 25),
    ("Video-LLaVA", "structured"): (
        "dissertation_v2/results/videollava/vl_climbing_entire_n8_structured.csv",
        "diss_dance/results/videollava/vl_dance_entire_n8_structured.csv", 25),
    ("Video-LLaVA", "reasoning"): (
        "dissertation_v2/results/videollava/vl_climbing_entire_n8_reasoning.csv",
        "diss_dance/results/videollava/vl_dance_entire_n8_reasoning.csv", 25),
    ("Gemini", "binary"): (
        "dissertation_v2/results/gemini/gemini_climbing_entire_binary.csv",
        "diss_dance/results/gemini/gemini_dance_entire_binary.csv", 50),
    ("Gemini", "fourclass"): (
        "dissertation_v2/results/gemini/gemini_climbing_entire_fourclass.csv",
        "diss_dance/results/gemini/gemini_dance_entire_fourclass.csv", 25),
    ("Gemini", "structured"): (
        "dissertation_v2/results/gemini/gemini_climbing_entire_structured.csv",
        "diss_dance/results/gemini/gemini_dance_entire_structured.csv", 25),
    ("Gemini", "reasoning"): (
        "dissertation_v2/results/gemini/gemini_climbing_entire_reasoning.csv",
        "diss_dance/results/gemini/gemini_dance_entire_reasoning.csv", 25),
}

MODEL_COLOR = {"Qwen2.5-VL": NAVY, "Video-LLaVA": TEAL, "Gemini": AMBER}
MODEL_MARKER = {"Qwen2.5-VL": "o", "Video-LLaVA": "s", "Gemini": "^"}


def exo_or_single_accuracy(df):
    for view, _pred, corr in views_present(df):
        if view in ("exo", "single"):
            return accuracy(df, corr)[0]
    return None


def main():
    fig, ax = plt.subplots(figsize=(9, 9))

    plotted_models = set()
    points = []
    for (model, prompt), (climb_path, dance_path, baseline) in FILES.items():
        climb_acc = exo_or_single_accuracy(load(climb_path))
        dance_acc = exo_or_single_accuracy(load(dance_path))
        if climb_acc is None or dance_acc is None:
            continue
        label = model if model not in plotted_models else None
        plotted_models.add(model)
        ax.scatter(climb_acc, dance_acc, color=MODEL_COLOR[model], marker=MODEL_MARKER[model],
                   s=110, label=label, edgecolor="white", linewidth=0.8, zorder=3)
        points.append((climb_acc, dance_acc, prompt))

    # Spread labels for points that are close together to avoid overlapping text.
    used_offsets = []
    for cx, cy, prompt in points:
        candidates = [(8, 6), (8, -12), (-45, 6), (-45, -12), (8, 18), (8, -22)]
        chosen = candidates[0]
        for cand in candidates:
            lx, ly = cx + cand[0] * 0.4, cy + cand[1] * 0.4
            if all(((lx - ux) ** 2 + (ly - uy) ** 2) ** 0.5 > 4 for ux, uy in used_offsets):
                chosen = cand
                break
        used_offsets.append((cx + chosen[0] * 0.4, cy + chosen[1] * 0.4))
        ax.annotate(prompt, (cx, cy), textcoords="offset points", xytext=chosen,
                    fontsize=8, fontfamily="sans-serif", color="#333")

    ax.plot([0, 100], [0, 100], linestyle="--", color=GREY, linewidth=1.3, zorder=1,
             label="y = x (equal performance)")
    ax.axvline(50, color="#ccc", linewidth=0.8, zorder=0)
    ax.axhline(50, color="#ccc", linewidth=0.8, zorder=0)
    ax.axvline(25, color="#e5e5e5", linewidth=0.8, linestyle=":", zorder=0)
    ax.axhline(25, color="#e5e5e5", linewidth=0.8, linestyle=":", zorder=0)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Climbing accuracy (%)", fontsize=11, fontfamily="sans-serif")
    ax.set_ylabel("Dance accuracy (%)", fontsize=11, fontfamily="sans-serif")
    ax.set_title("Cross-Domain Consistency: Climbing vs. Dance Accuracy\n(each point = one model x prompt-type)",
                 fontsize=12.5, fontfamily="sans-serif", fontweight="bold")
    style_ax(ax, ylabel="Dance accuracy (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("chart3_cross_domain_scatter.png", dpi=300)
    print("Saved chart3_cross_domain_scatter.png")


if __name__ == "__main__":
    main()
