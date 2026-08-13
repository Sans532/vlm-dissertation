"""Generate thesis figures from re-scored VLM proficiency-estimation results.

Reads figures/master_rows.csv (produced by figures/rescore.py) and writes
PNG (300 dpi) + PDF versions of each figure into figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- palette (validated categorical order; ordinal ramp for skill classes) --
QWEN = "#2a78d6"     # slot 1 blue
VL = "#eb6834"       # slot 2 orange
AQUA = "#1baf7a"     # slot 3
GRAY = "#9aa0a6"
INK = "#0b0b0b"
INK2 = "#52514e"
CHANCE = "#52514e"
# ordinal ramp Novice -> Late Expert (single hue, light->dark)
RAMP4 = ["#d7e5f7", "#9cc0ec", "#5d97dd", "#1e5da8"]
CLASSES = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]
CLASS_COLOR = dict(zip(CLASSES, RAMP4))
MODEL_NAME = {"qwen": "Qwen2.5-VL", "videollava": "Video-LLaVA"}
MODEL_COLOR = {"qwen": QWEN, "videollava": VL}
FOURWAY = ["fourclass", "structured", "reasoning"]
PROMPTS = ["binary", "fourclass", "structured", "reasoning"]

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

MIN_VALID = 20  # runs with fewer valid answers are excluded from accuracy plots


def style(ax, ylabel=None, xlabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def load():
    df = pd.read_csv(os.path.join(OUT, "master_rows.csv"))
    df["valid"] = df["pred"].notna() & (df["pred"] != "Unparseable")
    df["correct"] = df["valid"] & (df["pred"] == df["gt"])
    return df


def runs_table(df):
    g = df.groupby(["activity", "model", "condition", "frames", "prompt", "view"], dropna=False)
    t = g.agg(n=("gt", "size"), n_valid=("valid", "sum"), n_correct=("correct", "sum")).reset_index()
    t["acc"] = 100 * t.n_correct / t.n_valid.where(t.n_valid > 0)
    t["invalid_rate"] = 100 * (t.n - t.n_valid) / t.n
    return t


PROMPT_LABEL = {"binary": "binary", "fourclass": "four-class",
                "structured": "structured", "reasoning": "reasoning"}


def chance_of(prompt):
    return 50.0 if prompt == "binary" else 25.0


# ------------------------------------------------------------------- fig 1 --
def fig1(runs):
    """Headline: accuracy above chance, so 2-way and 4-way tasks share one scale."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for i, activity in enumerate(["climbing", "dance"]):
        ax = axes[i]
        sub = runs[(runs.activity == activity) & (runs.n_valid >= MIN_VALID)].copy()
        sub["above"] = sub.acc - sub.prompt.map(chance_of)
        for k, p in enumerate(PROMPTS):
            for model, dx in (("qwen", -0.15), ("videollava", 0.15)):
                pts = sub[(sub.prompt == p) & (sub.model == model)]["above"].values
                if not len(pts):
                    continue
                x = k + dx + rng.uniform(-0.07, 0.07, len(pts))
                ax.scatter(x, pts, s=30, color=MODEL_COLOR[model], alpha=0.8,
                           edgecolors="white", linewidths=0.6, zorder=3)
        ax.axhline(0, color=CHANCE, linewidth=1.4, zorder=2)
        ax.axvline(0.5, color=GRAY, linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels([PROMPT_LABEL[p] for p in PROMPTS], fontsize=9)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-30, 35)
        ax.set_title(f"{activity.capitalize()}", fontsize=11)
        style(ax, ylabel="Accuracy above chance (pp)" if i == 0 else None)
        ax.text(0, 32, "2-way task", fontsize=8, color=INK2, ha="center")
        ax.text(2, 32, "4-way tasks", fontsize=8, color=INK2, ha="center")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=MODEL_COLOR[m],
                          markersize=7, label=MODEL_NAME[m]) for m in ("qwen", "videollava")]
    handles.append(plt.Line2D([], [], color=CHANCE, linewidth=1.4, label="chance (0 pp)"))
    axes[1].legend(handles=handles, frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("No condition beats chance: accuracy expressed relative to each task's own baseline",
                 y=1.0, fontsize=12)
    fig.text(0.5, -0.02,
             "Each dot = one run (frame count × entire/trimmed × exo/ego). Chance is subtracted per task: "
             "50% for the 2-way binary prompt, 25% for the 4-way prompts.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig01_accuracy_above_chance")


# ------------------------------------------------------------------ fig 1b --
def fig1b(runs):
    """Raw accuracy, with the 2-way task kept on its own panel and own scale."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4),
                             gridspec_kw={"width_ratios": [1.3, 2.1]})
    short_model = {"qwen": "Qwen", "videollava": "V-LLaVA"}
    rng = np.random.default_rng(1)
    groups = [(a, m) for a in ("climbing", "dance") for m in ("qwen", "videollava")]

    # -- left: binary only
    ax = axes[0]
    for k, (a, m) in enumerate(groups):
        pts = runs[(runs.activity == a) & (runs.model == m) & (runs.prompt == "binary")
                   & (runs.n_valid >= MIN_VALID)]["acc"].values
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(pts)), pts, s=30, color=MODEL_COLOR[m],
                   alpha=0.8, edgecolors="white", linewidths=0.6, zorder=3)
    ax.axhline(50, color=CHANCE, linestyle="--", linewidth=1.3, zorder=2)
    ax.text(3.45, 51.5, "chance = 50%", fontsize=8.5, color=CHANCE, ha="right")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{a}\n{short_model[m]}" for a, m in groups], fontsize=8.5)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 100)
    ax.set_title("2-way task: Novice vs Expert", fontsize=11)
    style(ax, ylabel="Accuracy (%)")

    # -- right: the three four-class prompts
    ax = axes[1]
    fourway_prompts = ["fourclass", "structured", "reasoning"]
    for k, (a, m) in enumerate(groups):
        for j, p in enumerate(fourway_prompts):
            pts = runs[(runs.activity == a) & (runs.model == m) & (runs.prompt == p)
                       & (runs.n_valid >= MIN_VALID)]["acc"].values
            x = k + (j - 1) * 0.26
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(pts)), pts, s=30,
                       color=MODEL_COLOR[m], alpha=0.8, marker=["o", "s", "^"][j],
                       edgecolors="white", linewidths=0.6, zorder=3)
    ax.axhline(25, color=CHANCE, linestyle="--", linewidth=1.3, zorder=2)
    ax.text(3.45, 26.5, "chance = 25%", fontsize=8.5, color=CHANCE, ha="right")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{a}\n{short_model[m]}" for a, m in groups], fontsize=8.5)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 100)
    ax.set_title("4-way tasks: Novice / Early / Intermediate / Late Expert", fontsize=11)
    style(ax)
    shape_handles = [plt.Line2D([], [], marker=mk, linestyle="", color=INK2, markersize=6,
                                label=PROMPT_LABEL[p]) for mk, p in zip(["o", "s", "^"], fourway_prompts)]
    ax.legend(handles=shape_handles, frameon=False, fontsize=8.5, loc="upper right", ncol=3)

    model_handles = [plt.Line2D([], [], marker="o", linestyle="", color=MODEL_COLOR[m],
                                markersize=7, label=MODEL_NAME[m]) for m in ("qwen", "videollava")]
    fig.legend(handles=model_handles, frameon=False, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Raw accuracy, shown separately for the 2-way and 4-way tasks", y=1.0, fontsize=12)
    fig.text(0.5, -0.13, "The two task types have different chance baselines and are never directly comparable: "
             "50% on the binary task is the same as 25% on the four-class task.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig01b_accuracy_by_task_type")


# ------------------------------------------------------------------- fig 2 --
def fig2(runs):
    """Frame-count effect: slopes 8->16 for 4-way tasks + VL output collapse."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for i, activity in enumerate(["climbing", "dance"]):
        ax = axes[i]
        sub = runs[(runs.activity == activity) & runs.prompt.isin(FOURWAY)
                   & runs.frames.isin([8, 16]) & (runs.n_valid >= MIN_VALID)]
        for model in ["qwen", "videollava"]:
            ms = sub[sub.model == model]
            # thin lines: one per prompt x condition x view where both frame counts exist
            for key, g in ms.groupby(["prompt", "condition", "view"]):
                g = g.sort_values("frames")
                if set(g.frames) == {8, 16}:
                    ax.plot(g.frames, g.acc, color=MODEL_COLOR[model], alpha=0.25, linewidth=1)
            mean = ms.groupby("frames").acc.mean()
            ax.plot(mean.index, mean.values, color=MODEL_COLOR[model], linewidth=2.6,
                    marker="o", markersize=6, label=MODEL_NAME[model], zorder=4)
        ax.axhline(25, color=CHANCE, linestyle="--", linewidth=1.1)
        ax.set_xticks([8, 16])
        ax.set_ylim(0, 60)
        ax.set_title(f"{activity.capitalize()}: four-class accuracy", fontsize=10.5)
        style(ax, ylabel="Accuracy (%)" if i == 0 else None, xlabel="Frames sampled")
        if i == 0:
            ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.text(16.15, 25, "chance", fontsize=8, color=CHANCE, va="center")

    # panel 3: valid-output rate vs frames, climbing
    ax = axes[2]
    sub = runs[runs.activity == "climbing"]
    for model in ["qwen", "videollava"]:
        ms = sub[sub.model == model]
        vr = ms.groupby("frames").apply(lambda g: 100 * g.n_valid.sum() / g.n.sum(), include_groups=False)
        ax.plot(vr.index, vr.values, color=MODEL_COLOR[model], linewidth=2.6,
                marker="o", markersize=6, label=MODEL_NAME[model])
    ax.set_xticks([8, 16])
    ax.set_ylim(-3, 103)
    ax.set_title("Climbing: usable responses", fontsize=10.5)
    style(ax, ylabel="Valid-output rate (%)", xlabel="Frames sampled")
    ax.annotate("Video-LLaVA output\ncollapses at 16 frames", xy=(15.85, 27), xytext=(9.6, 46),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    fig.suptitle("Increasing frame count does not improve accuracy — and breaks Video-LLaVA", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "fig02_frame_count_effect")


# ------------------------------------------------------------------- fig 3 --
def fig3(runs):
    """Entire vs trimmed dumbbells per activity x model x prompt."""
    sub = runs[(runs.n_valid >= MIN_VALID)]
    agg = sub.groupby(["activity", "model", "prompt", "condition"]).apply(
        lambda g: 100 * g.n_correct.sum() / g.n_valid.sum(), include_groups=False).rename("acc").reset_index()
    rows = []
    for activity in ["climbing", "dance"]:
        for model in ["qwen", "videollava"]:
            for p in PROMPTS:
                e = agg[(agg.activity == activity) & (agg.model == model) & (agg.prompt == p) & (agg.condition == "entire")]
                t = agg[(agg.activity == activity) & (agg.model == model) & (agg.prompt == p) & (agg.condition == "trimmed")]
                if len(e) and len(t):
                    rows.append((activity, model, p, e.acc.iloc[0], t.acc.iloc[0]))
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    ys, labels = [], []
    y = 0
    for activity in ["climbing", "dance"]:
        for model in ["qwen", "videollava"]:
            block = [r for r in rows if r[0] == activity and r[1] == model]
            if not block:
                continue
            ax.text(-0.16, y + 0.55, f"{activity.capitalize()} — {MODEL_NAME[model]}",
                    fontsize=9.5, fontweight="bold", ha="left", va="bottom",
                    transform=ax.get_yaxis_transform())
            for (_, _, p, ea, ta) in block:
                ax.plot([ea, ta], [y, y], color=GRAY, linewidth=1.6, zorder=2)
                ax.scatter([ea], [y], s=52, color="white", edgecolors=MODEL_COLOR[model], linewidths=1.8, zorder=3)
                ax.scatter([ta], [y], s=52, color=MODEL_COLOR[model], edgecolors="white", linewidths=0.8, zorder=3)
                ys.append(y); labels.append(p)
                y -= 1
            y -= 0.9
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(25, color=CHANCE, linestyle="--", linewidth=1)
    ax.axvline(50, color=CHANCE, linestyle=":", linewidth=1)
    ax.text(25, 1.005, "25% chance (4-class) ", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=CHANCE)
    ax.text(50, 1.005, " 50% chance (binary)", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=8, color=CHANCE)
    ax.set_xlim(0, 80)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)
    ax.set_xlabel("Accuracy (%)")
    ax.scatter([], [], s=52, color="white", edgecolors=INK2, linewidths=1.8, label="entire video")
    ax.scatter([], [], s=52, color=INK2, label="trimmed to task segment")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=9)
    ax.set_title("Trimming videos to the task segment does not change accuracy", pad=26)
    fig.tight_layout()
    save(fig, "fig03_entire_vs_trimmed")


# ------------------------------------------------------------------- fig 4 --
def fig4(runs, df):
    """View asymmetry: exo vs ego accuracy + Novice-share by view."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    ax = axes[0]
    sub = runs[runs.n_valid >= MIN_VALID]
    piv = sub.pivot_table(index=["activity", "model", "condition", "frames", "prompt"],
                          columns="view", values="acc").dropna().reset_index()
    for model in ["qwen", "videollava"]:
        for activity, marker in (("climbing", "o"), ("dance", "^")):
            m = piv[(piv.model == model) & (piv.activity == activity)]
            ax.scatter(m.exo, m.ego, s=34, color=MODEL_COLOR[model], marker=marker,
                       alpha=0.8, edgecolors="white", linewidths=0.6,
                       label=f"{MODEL_NAME[model]}, {activity}")
    ax.plot([0, 80], [0, 80], color=GRAY, linewidth=1, zorder=1)
    ax.set_xlim(0, 80); ax.set_ylim(0, 80)
    style(ax, ylabel="Egocentric accuracy (%)", xlabel="Exocentric accuracy (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Accuracy: exocentric vs egocentric view", fontsize=10.5)

    ax = axes[1]
    d4 = df[df.prompt.isin(FOURWAY) & df.valid]
    share = d4.groupby(["model", "activity", "view"]).pred.apply(lambda s: 100 * (s == "Novice").mean()).reset_index()
    groups = [(m, a) for m in ["qwen", "videollava"] for a in ["climbing", "dance"]]
    x = np.arange(len(groups))
    w = 0.36
    for off, view, alpha in ((-w/2, "exo", 1.0), (w/2, "ego", 0.45)):
        vals = [share[(share.model == m) & (share.activity == a) & (share.view == view)].pred.iloc[0] for m, a in groups]
        cols = [MODEL_COLOR[m] for m, _ in groups]
        bars = ax.bar(x + off, vals, width=w * 0.94, color=cols, alpha=alpha, label=view + "centric")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(25, color=CHANCE, linestyle="--", linewidth=1)
    ax.text(len(groups) - 0.45, 26.5, "true share (25%)", fontsize=8, color=CHANCE, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_NAME[m]}\n{a}" for m, a in groups], fontsize=8.5)
    ax.set_ylim(0, 100)
    style(ax, ylabel="“Novice” predictions (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Egocentric views amplify the Novice default", fontsize=10.5)
    fig.suptitle("Camera viewpoint: little accuracy difference, strong bias difference", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "fig04_viewpoint_asymmetry")


# ------------------------------------------------------------------- fig 5 --
def fig5(df):
    """Label collapse: predicted-class distribution vs ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    d4 = df[df.prompt.isin(FOURWAY) & df.valid]
    rows = [("Ground truth", None, {c: 25.0 for c in CLASSES})]
    for m in ["qwen", "videollava"]:
        for a in ["climbing", "dance"]:
            s = d4[(d4.model == m) & (d4.activity == a)].pred.value_counts(normalize=True) * 100
            rows.append((f"{MODEL_NAME[m]} — {a}", m, {c: s.get(c, 0.0) for c in CLASSES}))
    ypos = np.arange(len(rows))[::-1]
    for y, (label, _, dist) in zip(ypos, rows):
        left = 0
        for c in CLASSES:
            v = dist[c]
            ax.barh(y, v, left=left, height=0.62, color=CLASS_COLOR[c],
                    edgecolor="white", linewidth=1.2)
            if v > 8:
                ax.text(left + v/2, y, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color=INK if c in CLASSES[:2] else "white")
            left += v
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 100)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlabel("Share of predictions (%)")
    ax.set_title("Four-class prompts: predictions collapse onto 1–2 labels", fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c in CLASSES]
    ax.legend(handles, CLASSES, frameon=False, fontsize=8, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))

    ax = axes[1]
    b = df[(df.prompt == "binary") & df.valid]
    groups = [(m, a) for m in ["qwen", "videollava"] for a in ["climbing", "dance"]]
    vals = [100 * (b[(b.model == m) & (b.activity == a)].pred == "Novice").mean() for m, a in groups]
    cols = [MODEL_COLOR[m] for m, _ in groups]
    bars = ax.bar(np.arange(len(groups)), vals, color=cols, width=0.62)
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width()/2, v - 6, f"{v:.0f}%", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.axhline(50, color=CHANCE, linestyle="--", linewidth=1.1)
    ax.text(-0.4, 52, "true share (50%)", fontsize=8, color=CHANCE)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([a for _, a in groups], fontsize=9)
    ax.set_ylim(0, 108)
    style(ax, ylabel="“Novice” answers (%)")
    handles2 = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m]) for m in ("qwen", "videollava")]
    ax.legend(handles2, [MODEL_NAME[m] for m in ("qwen", "videollava")], frameon=False,
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("Binary prompts: (almost) everyone is a novice", fontsize=10.5)
    fig.suptitle("Label collapse: models default to “Novice” regardless of true skill", y=1.03, fontsize=12)
    fig.tight_layout()
    save(fig, "fig05_label_collapse")


# ------------------------------------------------------------------- fig 6 --
def fig6(df):
    """Row-normalised confusion matrices, pooled four-way tasks."""
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6))
    d4 = df[df.prompt.isin(FOURWAY) & df.valid]
    short = {"Novice": "Nov", "Early Expert": "Early", "Intermediate Expert": "Interm", "Late Expert": "Late"}
    for i, activity in enumerate(["climbing", "dance"]):
        for j, model in enumerate(["qwen", "videollava"]):
            ax = axes[i][j]
            sub = d4[(d4.activity == activity) & (d4.model == model)]
            mat = np.zeros((4, 4))
            for gi, g in enumerate(CLASSES):
                gsub = sub[sub["gt"] == g]
                for pi, p in enumerate(CLASSES):
                    mat[gi, pi] = 100 * (gsub.pred == p).mean() if len(gsub) else np.nan
            im = ax.imshow(mat, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                "blues", ["#ffffff", QWEN]), vmin=0, vmax=100)
            for gi in range(4):
                for pi in range(4):
                    v = mat[gi, pi]
                    ax.text(pi, gi, f"{v:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if v > 55 else INK)
            ax.set_xticks(range(4)); ax.set_xticklabels([short[c] for c in CLASSES], fontsize=8.5)
            ax.set_yticks(range(4)); ax.set_yticklabels([short[c] for c in CLASSES], fontsize=8.5)
            if i == 1:
                ax.set_xlabel("Predicted")
            if j == 0:
                ax.set_ylabel("Ground truth")
            ax.set_title(f"{activity.capitalize()} — {MODEL_NAME[model]}", fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle("Confusion matrices (row-normalised %, four-way tasks pooled, valid answers)", y=0.98, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig06_confusion_matrices")


# ------------------------------------------------------------------- fig 7 --
def fig7(runs):
    """Invalid-output rates by model / activity / frames."""
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    combos = []
    for a in ["climbing", "dance"]:
        for f in [8, 16]:
            combos.append((a, f))
    x = np.arange(len(combos))
    w = 0.36
    for off, model in ((-w/2, "qwen"), (w/2, "videollava")):
        vals, xs = [], []
        for k, (a, f) in enumerate(combos):
            sub = runs[(runs.activity == a) & (runs.model == model) & (runs.frames == f)]
            if not len(sub):
                continue
            vals.append(100 * (sub.n - sub.n_valid).sum() / sub.n.sum())
            xs.append(k + off)
        bars = ax.bar(xs, vals, width=w * 0.92, color=MODEL_COLOR[model], label=MODEL_NAME[model])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{f} frames" for a, f in combos], fontsize=9)
    ax.set_ylim(0, 108)
    style(ax, ylabel="Invalid responses (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Unusable responses (runtime errors, empty or unclassifiable output)", pad=10)
    ax.annotate("9 climbing takes missing on disk\n→ ~9% baseline error rate", xy=(0.65, 12), xytext=(1.0, 40),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    fig.tight_layout()
    save(fig, "fig07_output_validity")


# ------------------------------------------------------------------- fig 8 --
def fig8(df):
    """Impact of re-scoring: stored vs recomputed accuracy per run."""
    s = df[df.stored_pred.notna()].copy()
    s["stored_ok"] = s.stored_pred == s["gt"]
    g = s.groupby(["activity", "model", "condition", "frames", "prompt", "view"], dropna=False)
    t = g.agg(n=("gt", "size"), stored=("stored_ok", "mean"), mine=("correct", "mean")).reset_index()
    t["stored"] *= 100; t["mine"] *= 100
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    for model in ["qwen", "videollava"]:
        m = t[t.model == model]
        ax.scatter(m.stored, m.mine, s=30, alpha=0.75, color=MODEL_COLOR[model],
                   edgecolors="white", linewidths=0.6, label=MODEL_NAME[model])
    lim = max(t.stored.max(), t.mine.max()) + 5
    ax.plot([0, lim], [0, lim], color=GRAY, linewidth=1, zorder=1)
    t["delta"] = (t.mine - t.stored).abs()
    worst = t.nlargest(3, "delta").reset_index(drop=True)
    offsets = [(10, -3), (14, -13), (6, -22)]
    for k, r in worst.iterrows():
        ax.annotate(f"{r.activity} {r.prompt}, {r.condition} n{int(r.frames)} {r.view} ({r.stored:.0f}→{r.mine:.0f})",
                    xy=(r.stored, r.mine), xytext=(r.stored + offsets[k][0], r.mine + offsets[k][1]),
                    fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    style(ax, ylabel="Re-parsed accuracy (%)", xlabel="Originally recorded accuracy (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Effect of re-parsing model answers on measured accuracy", pad=10)
    fig.tight_layout()
    save(fig, "fig08_rescoring_impact")


# ------------------------------------------------------------------- fig 9 --
def fig9(df):
    """Response stereotypy: share of most common opening line, reasoning prompt."""
    r = df[(df.prompt == "reasoning") & (df.raw.str.len() > 10)].copy()
    r["opening"] = r.raw.str.split("\n").str[0].str.strip().str.slice(0, 90)
    rows = []
    for m in ["qwen", "videollava"]:
        for a in ["climbing", "dance"]:
            for v in ["exo", "ego"]:
                sub = r[(r.model == m) & (r.activity == a) & (r.view == v)]
                if len(sub) < 30:
                    continue
                top = sub.opening.value_counts(normalize=True)
                rows.append((m, a, v, 100 * top.iloc[0], top.index[0], len(sub)))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ylab, vals, cols, alphas = [], [], [], []
    for m, a, v, share, line, n in rows:
        ylab.append(f"{MODEL_NAME[m]} — {a}, {v}")
        vals.append(share)
        cols.append(MODEL_COLOR[m])
        alphas.append(1.0 if v == "exo" else 0.45)
    y = np.arange(len(vals))[::-1]
    for yi, v, c, al in zip(y, vals, cols, alphas):
        ax.barh(yi, v, color=c, alpha=al, height=0.62)
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(ylab, fontsize=9)
    ax.set_xlim(0, 105)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of responses with the single most frequent opening line (%)")
    ax.set_title("Reasoning responses are highly stereotyped:\nmany clips receive a near-identical answer", pad=10, fontsize=11)
    fig.tight_layout()
    save(fig, "fig09_response_stereotypy")


def main():
    df = load()
    runs = runs_table(df)
    runs.to_csv(os.path.join(OUT, "master_summary.csv"), index=False)
    fig1(runs)
    fig1b(runs)
    fig2(runs)
    fig3(runs)
    fig4(runs, df)
    fig5(df)
    fig6(df)
    fig7(runs)
    fig8(df)
    fig9(df)


if __name__ == "__main__":
    main()
