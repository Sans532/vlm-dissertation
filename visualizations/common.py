"""Shared CSV-loading helpers for the dissertation chart scripts.

Handles the inconsistent column naming across models/activities:
- GT column: gt_binary | gt_fourclass | ground_truth
- Prediction columns: {exo,ego}_predicted | {exo,ego}_answer | predicted | answer
- Correctness columns: {exo,ego}_correct | correct
"""

import glob
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAVY = "#1E2761"
TEAL = "#028090"
AMBER = "#EDA100"
GREY = "#9AA0A6"


def gt_col(df):
    for c in ["gt_binary", "gt_fourclass", "ground_truth"]:
        if c in df.columns:
            return c
    return None


def views_present(df):
    """Return list of (view, pred_col, correct_col) tuples actually present in df."""
    out = []
    if any(c.startswith("exo_") for c in df.columns):
        for view, preds, corr in [
            ("exo", ["exo_predicted", "exo_answer"], "exo_correct"),
            ("ego", ["ego_predicted", "ego_answer"], "ego_correct"),
        ]:
            pred = next((c for c in preds if c in df.columns), None)
            if pred and corr in df.columns:
                out.append((view, pred, corr))
    else:
        pred = next((c for c in ["predicted", "answer"] if c in df.columns), None)
        if pred and "correct" in df.columns:
            out.append(("single", pred, "correct"))
    return out


def accuracy(df, correct_col):
    s = pd.to_numeric(df[correct_col], errors="coerce").dropna()
    return (s.mean() * 100, len(s)) if len(s) else (None, 0)


def pred_distribution(df, pred_col, normalize=True):
    return df[pred_col].dropna().astype(str).value_counts(normalize=normalize) * (100 if normalize else 1)


def load(relpath):
    path = os.path.join(BASE, relpath)
    return pd.read_csv(path)


def find(activity_dir, pattern):
    return sorted(glob.glob(os.path.join(BASE, activity_dir, pattern), recursive=True))


def style_ax(ax, ylabel="Accuracy (%)"):
    ax.set_ylabel(ylabel, fontsize=11, fontfamily="sans-serif")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
