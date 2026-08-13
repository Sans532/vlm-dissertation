"""Generate thesis figures for the video-native / adaptive-sampling models.

Reads figures-video/master_rows.csv (from rescore_video.py) and, for the
cross-generation figure, the frame-sampling set's master_rows.csv.
Writes PNG (300 dpi) + PDF into figures-video/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT)
FRAME_DIR = next((os.path.join(BASE, d) for d in ("figures-frame_count", "figures")
                  if os.path.exists(os.path.join(BASE, d, "master_rows.csv"))), None)

# ---- palette: one stable hue per model across both figure sets --------------
BLUE = "#2a78d6"     # Qwen2.5-VL   (slot 1)
ORANGE = "#eb6834"   # Video-LLaVA  (slot 2)
AQUA = "#1baf7a"     # Qwen3-VL     (slot 3)
VIOLET = "#4a3aa7"   # Gemini       (slot 7)
GRAY = "#9aa0a6"
INK = "#0b0b0b"
INK2 = "#52514e"
CHANCE = "#52514e"
RAMP4 = ["#d7e5f7", "#9cc0ec", "#5d97dd", "#1e5da8"]
CLASSES = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]
CLASS_COLOR = dict(zip(CLASSES, RAMP4))
SHORT = {"Novice": "Nov", "Early Expert": "Early",
         "Intermediate Expert": "Interm", "Late Expert": "Late"}

MODEL_NAME = {"gemini": "Gemini", "qwen3vl": "Qwen3-VL",
              "qwen": "Qwen2.5-VL", "videollava": "Video-LLaVA"}
MODEL_COLOR = {"gemini": VIOLET, "qwen3vl": AQUA, "qwen": BLUE, "videollava": ORANGE}
MODEL_SHORT = {"gemini": "Gemini", "qwen3vl": "Qwen3-VL",
               "qwen": "Qwen2.5", "videollava": "V-LLaVA"}
VIDEO_MODELS = ["gemini", "qwen3vl"]
PROMPTS = ["binary", "fourclass", "structured", "reasoning"]
PROMPT_LABEL = {"binary": "binary", "fourclass": "four-class",
                "structured": "structured", "reasoning": "reasoning"}
FOURWAY = ["fourclass", "structured", "reasoning"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "figure.facecolor": "white", "savefig.facecolor": "white",
})

MIN_VALID = 20


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


def chance_of(prompt):
    return 50.0 if prompt == "binary" else 25.0


def classes_for(prompt):
    return ["Novice", "Expert"] if prompt == "binary" else CLASSES


def balanced_acc(g, prompt):
    """Mean per-class recall — immune to a model that just picks one label."""
    recs = []
    for c in classes_for(prompt):
        sub = g[g["gt"] == c]
        if len(sub):
            recs.append((sub["pred"] == c).mean())
    return 100 * np.mean(recs) if recs else np.nan


def load():
    df = pd.read_csv(os.path.join(OUT, "master_rows.csv"))
    df["valid"] = df["pred"].notna() & (df["pred"] != "Unparseable")
    df["correct"] = df["valid"] & (df["pred"] == df["gt"])
    return df


def runs_table(df):
    rows = []
    for key, g in df.groupby(["activity", "model", "sampling", "prompt", "view"], dropna=False):
        v = g[g.valid]
        rows.append(dict(zip(["activity", "model", "sampling", "prompt", "view"], key),
                         n=len(g), n_valid=len(v), n_correct=int(g.correct.sum()),
                         acc=(100 * g.correct.sum() / len(v)) if len(v) else np.nan,
                         bal=balanced_acc(v, key[3]) if len(v) else np.nan,
                         invalid_rate=100 * (len(g) - len(v)) / len(g)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ vfig 1 --
def vfig1(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for i, activity in enumerate(["climbing", "dance"]):
        ax = axes[i]
        sub = runs[(runs.activity == activity) & (runs.n_valid >= MIN_VALID)
                   & runs.model.isin(VIDEO_MODELS)].copy()
        sub["above"] = sub.acc - sub.prompt.map(chance_of)
        for k, p in enumerate(PROMPTS):
            for model, dx in (("gemini", -0.15), ("qwen3vl", 0.15)):
                pts = sub[(sub.prompt == p) & (sub.model == model)]["above"].values
                if not len(pts):
                    continue
                ax.scatter(k + dx + rng.uniform(-0.05, 0.05, len(pts)), pts, s=44,
                           color=MODEL_COLOR[model], alpha=0.85, edgecolors="white",
                           linewidths=0.7, zorder=3)
        ax.axhline(0, color=CHANCE, linewidth=1.4, zorder=2)
        ax.axvline(0.5, color=GRAY, linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels([PROMPT_LABEL[p] for p in PROMPTS], fontsize=9)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-15, 32)
        ax.set_title(activity.capitalize(), fontsize=11)
        style(ax, ylabel="Accuracy above chance (pp)" if i == 0 else None)
        ax.text(0, 29, "2-way task", fontsize=8, color=INK2, ha="center")
        ax.text(2, 29, "4-way tasks", fontsize=8, color=INK2, ha="center")
    axes[0].annotate("Gemini, egocentric", xy=(-0.15, 26), xytext=(0.35, 22.5), fontsize=8,
                     color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=MODEL_COLOR[m],
                          markersize=8, label=MODEL_NAME[m]) for m in VIDEO_MODELS]
    handles.append(plt.Line2D([], [], color=CHANCE, linewidth=1.4, label="chance (0 pp)"))
    axes[1].legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Video-native models: only the 2-way climbing task clears chance", y=1.0, fontsize=12)
    fig.text(0.5, -0.02, "Each dot = one run (viewpoint × prompt). Chance subtracted per task: "
             "50% for the 2-way binary prompt, 25% for the 4-way prompts.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "vfig01_accuracy_above_chance")


# ----------------------------------------------------------------- vfig 1b --
def vfig1b(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 2.1]})
    rng = np.random.default_rng(1)
    groups = [(a, m) for a in ("climbing", "dance") for m in VIDEO_MODELS]

    ax = axes[0]
    for k, (a, m) in enumerate(groups):
        pts = runs[(runs.activity == a) & (runs.model == m) & (runs.prompt == "binary")
                   & (runs.n_valid >= MIN_VALID)]["acc"].values
        ax.scatter(k + rng.uniform(-0.16, 0.16, len(pts)), pts, s=42, color=MODEL_COLOR[m],
                   alpha=0.85, edgecolors="white", linewidths=0.7, zorder=3)
    ax.axhline(50, color=CHANCE, linestyle="--", linewidth=1.3, zorder=2)
    ax.text(3.45, 51.5, "chance = 50%", fontsize=8.5, color=CHANCE, ha="right")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{a}\n{MODEL_SHORT[m]}" for a, m in groups], fontsize=8.5)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 100)
    ax.set_title("2-way task: Novice vs Expert", fontsize=11)
    style(ax, ylabel="Accuracy (%)")

    ax = axes[1]
    for k, (a, m) in enumerate(groups):
        for j, p in enumerate(FOURWAY):
            pts = runs[(runs.activity == a) & (runs.model == m) & (runs.prompt == p)
                       & (runs.n_valid >= MIN_VALID)]["acc"].values
            ax.scatter(k + (j - 1) * 0.26 + rng.uniform(-0.05, 0.05, len(pts)), pts, s=42,
                       color=MODEL_COLOR[m], alpha=0.85, marker=["o", "s", "^"][j],
                       edgecolors="white", linewidths=0.7, zorder=3)
    ax.axhline(25, color=CHANCE, linestyle="--", linewidth=1.3, zorder=2)
    ax.text(3.45, 26.5, "chance = 25%", fontsize=8.5, color=CHANCE, ha="right")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{a}\n{MODEL_SHORT[m]}" for a, m in groups], fontsize=8.5)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 100)
    ax.set_title("4-way tasks: Novice / Early / Intermediate / Late Expert", fontsize=11)
    style(ax)
    ax.legend(handles=[plt.Line2D([], [], marker=mk, linestyle="", color=INK2, markersize=6,
                                  label=PROMPT_LABEL[p]) for mk, p in zip("os^", FOURWAY)],
              frameon=False, fontsize=8.5, loc="upper right", ncol=3)
    fig.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=MODEL_COLOR[m],
                                   markersize=8, label=MODEL_NAME[m]) for m in VIDEO_MODELS],
               frameon=False, fontsize=9, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Raw accuracy, shown separately for the 2-way and 4-way tasks", y=1.0, fontsize=12)
    fig.text(0.5, -0.13, "The two task types have different chance baselines and are never directly "
             "comparable: 50% on the binary task is the same as 25% on the four-class task.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "vfig01b_accuracy_by_task_type")


# ------------------------------------------------------------------ vfig 2 --
def vfig2(df):
    """Binary task: recall on each class, which separates skill from a one-label habit.

    The evaluation set is class-balanced, so overall accuracy is exactly the mean
    of the two recalls — the informative view is where a run sits in the recall
    plane, not its scalar accuracy.
    """
    b = df[(df.prompt == "binary") & df.valid & df.model.isin(VIDEO_MODELS)]
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.fill_between([0, 100], [100, 0], 100, color=AQUA, alpha=0.06, zorder=0)
    ax.plot([0, 100], [100, 0], color=CHANCE, linestyle="--", linewidth=1.3, zorder=2)
    for m in VIDEO_MODELS:
        for a, mk in (("climbing", "o"), ("dance", "^")):
            for v in ("exo", "ego"):
                g = b[(b.model == m) & (b.activity == a) & (b.view == v)]
                if not len(g):
                    continue
                rn = 100 * (g[g["gt"] == "Novice"]["pred"] == "Novice").mean()
                re = 100 * (g[g["gt"] == "Expert"]["pred"] == "Expert").mean()
                ax.scatter(rn, re, s=95, marker=mk, color=MODEL_COLOR[m], alpha=0.9,
                           edgecolors="white", linewidths=1.1, zorder=4)
                ax.annotate(v, xy=(rn, re), xytext=(4, -10), textcoords="offset points",
                            fontsize=7.5, color=INK2)
    ax.scatter([], [], s=95, marker="o", color=GRAY, label="climbing")
    ax.scatter([], [], s=95, marker="^", color=GRAY, label="dance")
    for m in VIDEO_MODELS:
        ax.scatter([], [], s=95, marker="s", color=MODEL_COLOR[m], label=MODEL_NAME[m])
    ax.set_xlim(-3, 108); ax.set_ylim(-8, 108)
    ax.annotate("always answers “Novice”\n(accuracy = 50%, no skill)", xy=(99.5, 1),
                xytext=(30, 15), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(52, 92, "better than chance", fontsize=9, color=INK2, style="italic")
    ax.text(10, 32, "worse than chance", fontsize=9, color=INK2, style="italic")
    ax.text(4, 46, "chance line\n(accuracy = 50%)", fontsize=8, color=CHANCE, rotation=-38)
    style(ax, ylabel="Recall on Expert clips (%)", xlabel="Recall on Novice clips (%)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    ax.set_title("Binary task: which runs actually tell the two classes apart?", pad=10)
    fig.text(0.5, -0.02,
             "The test set is class-balanced, so accuracy is the midpoint of the two recalls and every point on the\n"
             "dashed line scores 50%. Only Gemini leaves the bottom-right corner; Qwen3-VL sits in it, "
             "answering “Novice” almost always.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "vfig02_binary_discrimination")


# ------------------------------------------------------------------ vfig 3 --
def vfig3(df):
    """Qwen3-VL samples frames adaptively — does the resulting count matter?"""
    q = df[(df.model == "qwen3vl") & df.nframes.notna() & (df.nframes > 0)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    ax = axes[0]
    bins = np.arange(0, 280, 10)
    ax.hist(q[q.activity == "climbing"].nframes.values, bins=bins, color=AQUA,
            alpha=0.8, label="Climbing")
    ax.hist(q[q.activity == "dance"].nframes.values, bins=bins, histtype="step",
            color=AQUA, linewidth=1.8, linestyle="--", label="Dance")
    ax.set_xlim(0, 280)
    style(ax, ylabel="Clip-views", xlabel="Frames the model actually sampled")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Adaptive frame count varies 4–263", fontsize=10.5)

    ax = axes[1]
    for a, ls, mk in (("climbing", "-", "o"), ("dance", "--", "^")):
        g = q[(q.activity == a) & q.valid].copy()
        g["bin"] = pd.qcut(g.nframes, 4, duplicates="drop")
        agg = g.groupby("bin", observed=True).agg(acc=("correct", "mean"), n=("correct", "size"))
        xs = [iv.mid for iv in agg.index]
        ax.plot(xs, 100 * agg.acc.values, color=AQUA, marker=mk, markersize=6,
                linewidth=2.4, linestyle=ls, label=a.capitalize())
    ax.axhline(25, color=CHANCE, linestyle="--", linewidth=1.2)
    ax.text(0.99, 26.5, "chance (4-way)", fontsize=8, color=CHANCE, ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_ylim(0, 60)
    style(ax, ylabel="Accuracy (%)", xlabel="Frames sampled (quartile midpoint)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("More frames does not mean more accuracy", fontsize=10.5)

    ax = axes[2]
    for a, ls, mk in (("climbing", "-", "o"), ("dance", "--", "^")):
        g = q[q.activity == a].copy()
        g["bin"] = pd.qcut(g.nframes, 4, duplicates="drop")
        agg = g.groupby("bin", observed=True).apply(lambda x: 100 * (~x.valid).mean(), include_groups=False)
        xs = [iv.mid for iv in agg.index]
        ax.plot(xs, agg.values, color=AQUA, marker=mk, markersize=6, linewidth=2.4,
                linestyle=ls, label=a.capitalize())
    ax.set_ylim(-0.4, 9)
    style(ax, ylabel="Unusable responses (%)", xlabel="Frames sampled (quartile midpoint)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Long clips truncate before a verdict", fontsize=10.5)
    fig.suptitle("Qwen3-VL's adaptive sampling: more frames buys nothing and costs reliability",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    save(fig, "vfig03_adaptive_frame_count")


# ------------------------------------------------------------------ vfig 4 --
def vfig4(df, frame_df):
    """Where the predictions pile up — and how the pile moved between generations."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    rows = [("Ground truth", {c: 25.0 for c in CLASSES})]
    d4 = df[df.prompt.isin(FOURWAY) & df.valid & df.model.isin(VIDEO_MODELS)]
    for m in VIDEO_MODELS:
        for a in ("climbing", "dance"):
            s = d4[(d4.model == m) & (d4.activity == a)].pred.value_counts(normalize=True) * 100
            rows.append((f"{MODEL_NAME[m]} — {a}", {c: s.get(c, 0.0) for c in CLASSES}))
    if frame_df is not None:
        f4 = frame_df[frame_df.prompt.isin(FOURWAY) & frame_df.valid]
        for m in ("qwen", "videollava"):
            s = f4[f4.model == m].pred.value_counts(normalize=True) * 100
            rows.append((f"{MODEL_NAME[m]} (frame-sampled)", {c: s.get(c, 0.0) for c in CLASSES}))
    ypos = np.arange(len(rows))[::-1]
    for y, (label, dist) in zip(ypos, rows):
        left = 0
        for c in CLASSES:
            v = dist[c]
            ax.barh(y, v, left=left, height=0.62, color=CLASS_COLOR[c], edgecolor="white", linewidth=1.2)
            if v > 8:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color=INK if c in CLASSES[:2] else "white")
            left += v
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 100)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlabel("Share of predictions (%)")
    ax.set_title("The collapse target moved from “Novice” to “Intermediate Expert”", fontsize=10.5)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c in CLASSES], CLASSES,
              frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    ax = axes[1]
    b = df[(df.prompt == "binary") & df.valid & df.model.isin(VIDEO_MODELS)]
    groups = [(m, a) for m in VIDEO_MODELS for a in ("climbing", "dance")]
    x = np.arange(len(groups))
    for off, view, alpha in ((-0.18, "exo", 1.0), (0.18, "ego", 0.45)):
        vals = [100 * (b[(b.model == m) & (b.activity == a) & (b.view == view)].pred == "Novice").mean()
                for m, a in groups]
        bars = ax.bar(x + off, vals, width=0.34, color=[MODEL_COLOR[m] for m, _ in groups],
                      alpha=alpha, label=view + "centric")
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(50, color=CHANCE, linestyle="--", linewidth=1.1)
    ax.text(-0.45, 52, "true share (50%)", fontsize=8, color=CHANCE, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_SHORT[m]}\n{a}" for m, a in groups], fontsize=8.5)
    ax.set_ylim(0, 112)
    style(ax, ylabel="“Novice” answers (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Binary prompt: Gemini splits, Qwen3-VL does not", fontsize=10.5)
    fig.suptitle("Label collapse persists in video-native models — it just changed target", y=1.03, fontsize=12)
    fig.tight_layout()
    save(fig, "vfig04_label_collapse")


# ------------------------------------------------------------------ vfig 5 --
def vfig5(df):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6))
    d4 = df[df.prompt.isin(FOURWAY) & df.valid]
    for i, activity in enumerate(["climbing", "dance"]):
        for j, model in enumerate(VIDEO_MODELS):
            ax = axes[i][j]
            sub = d4[(d4.activity == activity) & (d4.model == model)]
            mat = np.zeros((4, 4))
            for gi, g in enumerate(CLASSES):
                gsub = sub[sub["gt"] == g]
                for pi, p in enumerate(CLASSES):
                    mat[gi, pi] = 100 * (gsub["pred"] == p).mean() if len(gsub) else np.nan
            ax.imshow(mat, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                "v", ["#ffffff", MODEL_COLOR[model]]), vmin=0, vmax=100)
            for gi in range(4):
                for pi in range(4):
                    v = mat[gi, pi]
                    ax.text(pi, gi, f"{v:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if v > 55 else INK)
            ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[c] for c in CLASSES], fontsize=8.5)
            ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[c] for c in CLASSES], fontsize=8.5)
            if i == 1:
                ax.set_xlabel("Predicted")
            if j == 0:
                ax.set_ylabel("Ground truth")
            ax.set_title(f"{activity.capitalize()} — {MODEL_NAME[model]}", fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle("Confusion matrices (row-normalised %, four-way prompts pooled)", y=0.98, fontsize=12)
    fig.text(0.5, 0.005, "Rows remain near-identical: the predicted label barely responds to the true skill level.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "vfig05_confusion_matrices")


# ------------------------------------------------------------------ vfig 6 --
def vfig6(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    ax = axes[0]
    piv = runs[(runs.n_valid >= MIN_VALID) & runs.model.isin(VIDEO_MODELS)].pivot_table(
        index=["activity", "model", "prompt"], columns="view", values="acc").dropna().reset_index()
    for m in VIDEO_MODELS:
        for a, mk in (("climbing", "o"), ("dance", "^")):
            g = piv[(piv.model == m) & (piv.activity == a)]
            ax.scatter(g.exo, g.ego, s=48, marker=mk, color=MODEL_COLOR[m], alpha=0.85,
                       edgecolors="white", linewidths=0.7, label=f"{MODEL_NAME[m]}, {a}")
    ax.plot([0, 85], [0, 85], color=GRAY, linewidth=1, zorder=1)
    ax.set_xlim(0, 85); ax.set_ylim(0, 85)
    style(ax, ylabel="Egocentric accuracy (%)", xlabel="Exocentric accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Accuracy by viewpoint", fontsize=10.5)

    ax = axes[1]
    sub = runs[(runs.n_valid >= MIN_VALID) & runs.model.isin(VIDEO_MODELS)]
    groups = [(m, a) for m in VIDEO_MODELS for a in ("climbing", "dance")]
    x = np.arange(len(groups))
    for off, view, alpha in ((-0.18, "exo", 1.0), (0.18, "ego", 0.45)):
        vals = [sub[(sub.model == m) & (sub.activity == a) & (sub.view == view)].bal.mean()
                for m, a in groups]
        bars = ax.bar(x + off, vals, width=0.34, color=[MODEL_COLOR[m] for m, _ in groups],
                      alpha=alpha, label=view + "centric")
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_SHORT[m]}\n{a}" for m, a in groups], fontsize=8.5)
    ax.set_ylim(0, 60)
    style(ax, ylabel="Mean balanced accuracy (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Balanced accuracy by viewpoint", fontsize=10.5)
    fig.suptitle("Egocentric video helps Gemini on climbing and hurts it on dance", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "vfig06_viewpoint_asymmetry")


# ------------------------------------------------------------------ vfig 7 --
def vfig7(df, frame_df):
    """Cross-generation: frame-sampled models vs video-native models."""
    if frame_df is None:
        print("skipping vfig07 (frame-sampling master_rows.csv not found)")
        return
    order = ["videollava", "qwen", "qwen3vl", "gemini"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for i, prompt_set in enumerate([["binary"], FOURWAY]):
        ax = axes[i]
        chance = 50 if prompt_set == ["binary"] else 25
        x = np.arange(2)
        w = 0.19
        for k, m in enumerate(order):
            src = frame_df if m in ("qwen", "videollava") else df
            vals, labels = [], []
            for a in ("climbing", "dance"):
                g = src[(src.model == m) & (src.activity == a) & src.prompt.isin(prompt_set) & src.valid]
                vals.append(balanced_acc(g, prompt_set[0]) if len(g) else np.nan)
            off = (k - 1.5) * w
            bars = ax.bar(x + off, vals, width=w * 0.92, color=MODEL_COLOR[m],
                          label=MODEL_NAME[m])
            for bb, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.0f}", ha="center",
                            fontsize=7.5, color=INK2)
        ax.axhline(chance, color=CHANCE, linestyle="--", linewidth=1.2)
        ax.text(0.5, chance + 1.5, f"chance = {chance}%", fontsize=8, color=CHANCE, ha="center")
        ax.set_xticks(x)
        ax.set_xticklabels(["climbing", "dance"], fontsize=9.5)
        ax.set_ylim(0, 85)
        ax.set_title("2-way task" if prompt_set == ["binary"] else "4-way tasks (pooled)", fontsize=11)
        style(ax, ylabel="Balanced accuracy (%)" if i == 0 else None)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    fig.suptitle("Four models, two generations: only Gemini on climbing escapes chance", y=1.02, fontsize=12)
    fig.text(0.5, -0.04, "Balanced accuracy (mean per-class recall) so that a model answering one label for every "
             "clip scores at chance.\nFrame-sampled models (Video-LLaVA, Qwen2.5-VL) pooled over frame counts and "
             "entire/trimmed; video models pooled over viewpoint.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "vfig07_generation_comparison")


# ------------------------------------------------------------------ vfig 8 --
def vfig8(df, runs):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    groups = [(m, a) for m in VIDEO_MODELS for a in ("climbing", "dance")]
    x = np.arange(len(groups))
    hatches = ["", "///", "...", "xxx"]
    for off, p, hatch in zip((-0.27, -0.09, 0.09, 0.27), PROMPTS, hatches):
        vals = []
        for m, a in groups:
            g = runs[(runs.model == m) & (runs.activity == a) & (runs.prompt == p)]
            vals.append(100 * (g.n - g.n_valid).sum() / g.n.sum() if len(g) else np.nan)
        bars = ax.bar(x + off, vals, width=0.17, color=[MODEL_COLOR[m] for m, _ in groups],
                      alpha=0.85, hatch=hatch, edgecolor="white", linewidth=0.8,
                      label=PROMPT_LABEL[p])
        for bb, v in zip(bars, vals):
            if not np.isnan(v) and v > 0.2:
                ax.text(bb.get_x() + bb.get_width() / 2, v + 0.4, f"{v:.0f}", ha="center",
                        fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_SHORT[m]}\n{a}" for m, a in groups], fontsize=8.5)
    ax.set_ylim(0, 23)
    style(ax, ylabel="Unusable responses (%)")
    leg = ax.legend(frameon=False, fontsize=8.5, ncol=2, title="prompt", title_fontsize=8.5,
                    loc="upper left")
    leg._legend_box.align = "left"
    ax.set_title("Unusable responses by prompt", fontsize=10.5)

    ax = axes[1]
    cov = [("Gemini\nall prompts", 100.0), ("Qwen3-VL\nadaptive frames", 100.0),
           ("Qwen3-VL\nnative video", 6.0)]
    bars = ax.bar(range(3), [c[1] for c in cov], width=0.55,
                  color=[VIOLET, AQUA, AQUA], alpha=1.0)
    bars[2].set_alpha(0.4)
    for b, (_, v) in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}%", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[0] for c in cov], fontsize=8.5)
    ax.set_ylim(0, 158)
    style(ax, ylabel="Clips with a recorded response (%)")
    ax.annotate("run never completed:\n3 of 4 prompt files empty,\nbinary covers 12 of 200 clip-views",
                xy=(1.96, 14), xytext=(-0.42, 122), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9,
                                connectionstyle="arc3,rad=-0.15"))
    ax.set_title("Experiment coverage", fontsize=10.5)
    fig.suptitle("Output validity and coverage", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "vfig08_output_validity_coverage")


# ------------------------------------------------------------------ vfig 9 --
def vfig9(df):
    """Answer richness and stereotypy of the long-form prompts."""
    d = df[df.prompt.isin(["structured", "reasoning"]) & df.model.isin(VIDEO_MODELS)].copy()
    d = d[d.raw.astype(str).str.len() > 10]
    d["len"] = d.raw.astype(str).str.len()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    ax = axes[0]
    cats, data, colors = [], [], []
    for m in VIDEO_MODELS:
        for p in ("structured", "reasoning"):
            g = d[(d.model == m) & (d.prompt == p)]
            if len(g):
                cats.append(f"{MODEL_SHORT[m]}\n{PROMPT_LABEL[p]}")
                data.append(g["len"].values)
                colors.append(MODEL_COLOR[m])
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_edgecolor(c); patch.set_alpha(0.85)
    for part in ("whiskers", "caps"):
        for art in bp[part]:
            art.set_color(INK2)
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels(cats, fontsize=8.5)
    style(ax, ylabel="Answer length (characters)")
    ax.set_title("Both models write long, detailed answers", fontsize=10.5)

    ax = axes[1]
    r = df[(df.prompt == "reasoning") & df.model.isin(VIDEO_MODELS)].copy()
    r = r[r.raw.astype(str).str.len() > 10]
    r["opening"] = r.raw.astype(str).str.split("\n").str[0].str.strip().str.slice(0, 90)
    ylab, vals, cols = [], [], []
    for m in VIDEO_MODELS:
        for a in ("climbing", "dance"):
            for v in ("exo", "ego"):
                g = r[(r.model == m) & (r.activity == a) & (r.view == v)]
                if len(g) < 30:
                    continue
                ylab.append(f"{MODEL_SHORT[m]} — {a}, {v}")
                vals.append(100 * g.opening.value_counts(normalize=True).iloc[0])
                cols.append(MODEL_COLOR[m])
    y = np.arange(len(vals))[::-1]
    for yi, v, c in zip(y, vals, cols):
        ax.barh(yi, v, color=c, height=0.62, alpha=0.85)
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(ylab, fontsize=8.5)
    ax.set_xlim(0, 105)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.7); ax.set_axisbelow(True)
    ax.set_xlabel("Responses sharing the most frequent opening line (%)")
    ax.set_title("…but every answer opens differently", fontsize=10.5)
    fig.suptitle("Answer style: verbose and non-repetitive, unlike the frame-sampled models", y=1.02, fontsize=12)
    fig.text(0.5, -0.04, "Compare the frame-sampled set, where Qwen2.5-VL opened 99–100% of egocentric "
             "reasoning answers with an identical sentence.", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "vfig09_answer_style")


# ----------------------------------------------------------------- vfig 10 --
def vfig10(df):
    s = df[df.stored_pred.notna() & (df.stored_pred != "Unknown")].copy()
    s["stored_ok"] = s.stored_pred == s["gt"]
    rows = []
    for key, g in s.groupby(["activity", "model", "prompt", "view"], dropna=False):
        rows.append(dict(zip(["activity", "model", "prompt", "view"], key),
                         stored=100 * g.stored_ok.mean(), mine=100 * g.correct.mean(), n=len(g)))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for m in VIDEO_MODELS:
        g = t[t.model == m]
        ax.scatter(g.stored, g.mine, s=42, color=MODEL_COLOR[m], alpha=0.85,
                   edgecolors="white", linewidths=0.7, label=MODEL_NAME[m])
    lim = float(max(t.stored.max(), t.mine.max())) + 6
    ax.plot([0, lim], [0, lim], color=GRAY, linewidth=1, zorder=1)
    t["delta"] = (t.mine - t.stored).abs()
    for k, (_, r) in enumerate(t.nlargest(2, "delta").iterrows()):
        ax.annotate(f"{r.activity} {PROMPT_LABEL[r.prompt]} {r.view} ({r.stored:.0f}→{r.mine:.0f})",
                    xy=(r.stored, r.mine), xytext=(r.stored - 4, r.mine + 9 + 7 * k), fontsize=7.5,
                    color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    style(ax, ylabel="Re-parsed accuracy (%)", xlabel="Originally recorded accuracy (%)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Effect of re-parsing model answers", pad=10)
    fig.tight_layout()
    save(fig, "vfig10_rescoring_impact")


def main():
    df = load()
    runs = runs_table(df)
    runs.round(1).to_csv(os.path.join(OUT, "master_summary.csv"), index=False)
    frame_df = None
    if FRAME_DIR:
        frame_df = pd.read_csv(os.path.join(FRAME_DIR, "master_rows.csv"))
        frame_df["valid"] = frame_df["pred"].notna() & (frame_df["pred"] != "Unparseable")
        frame_df["correct"] = frame_df["valid"] & (frame_df["pred"] == frame_df["gt"])
    vfig1(runs)
    vfig1b(runs)
    vfig2(df)
    vfig3(df)
    vfig4(df, frame_df)
    vfig5(df)
    vfig6(runs)
    vfig7(df, frame_df)
    vfig8(df, runs)
    vfig9(df)
    vfig10(df)


if __name__ == "__main__":
    main()
