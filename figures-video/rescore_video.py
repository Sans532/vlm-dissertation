"""Re-score the video-native / adaptive-sampling model results (Gemini, Qwen3-VL).

Uses the same answer parser as figures/rescore.py so the two figure sets are
directly comparable, and writes:
  - figures-video/master_rows.csv    : one row per (file, clip, view)
  - figures-video/master_summary.csv : accuracy per condition
  - figures-video/parse_disagreements.txt
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

# The frame-sampling figure set holds the shared answer parser; find it wherever
# that folder currently lives so both figure sets score answers identically.
FRAME_DIR = next((os.path.join(BASE, d) for d in ("figures-frame_count", "figures")
                  if os.path.exists(os.path.join(BASE, d, "rescore.py"))), None)
if FRAME_DIR is None:
    raise SystemExit("cannot find rescore.py (expected in figures-frame_count/ or figures/)")
sys.path.insert(0, FRAME_DIR)

from rescore import parse, to_binary_gt  # noqa: E402  (shared answer parser)

OUT = HERE

# ---------------------------------------------------------------- manifest --
# Gemini: one file per view, columns answer/predicted/correct.
# Qwen3-VL: paired exo/ego in one file, plus per-clip adaptive frame count.
M = []


def add(path, activity, model, sampling, prompt, schema):
    M.append(dict(path=path, activity=activity, model=model, sampling=sampling,
                  prompt=prompt, schema=schema))


for activity, tag in (("climbing", "climbing"), ("dance", "dance")):
    for prompt in ("binary", "fourclass", "structured", "reasoning"):
        add(f"diss_{'climb' if activity == 'climbing' else 'dance'}/results/gemini/"
            f"gemini_{tag}_entire_{prompt}.csv", activity, "gemini", "native video",
            prompt, ("single", "exo"))
        add(f"diss_{'climb' if activity == 'climbing' else 'dance'}/results/gemini/"
            f"gemini_{tag}_entire_{prompt}_ego.csv", activity, "gemini", "native video",
            prompt, ("single", "ego"))
    for prompt in ("binary", "fourclass", "structured", "reasoning"):
        add(f"diss_{'climb' if activity == 'climbing' else 'dance'}/results/qwen3vl/"
            f"qwen3vl_{tag}_frames_{prompt}.csv", activity, "qwen3vl", "adaptive frames",
            prompt, "paired")

# The Qwen3-VL "native video" run never completed: three of its four files are
# empty and the binary file holds 12 of 50 clips. Loaded so coverage can be
# reported (fig07), but excluded from accuracy figures by the MIN_VALID gate.
add("diss_climb/results/qwen3vl/qwen3vl_climbing_native_binary.csv", "climbing",
    "qwen3vl-native", "native video (partial)", "binary", "paired")
for prompt in ("fourclass", "structured", "reasoning"):
    add(f"diss_climb/results/qwen3vl/qwen3vl_climbing_native_{prompt}.csv", "climbing",
        "qwen3vl-native", "native video (partial)", prompt, "paired")


def load_all():
    rows = []
    for spec in M:
        path = os.path.join(BASE, spec["path"])
        if not os.path.exists(path):
            print("MISSING:", spec["path"])
            continue
        df = pd.read_csv(path)
        if df.empty:
            print("EMPTY (run never produced rows):", spec["path"])
            continue
        cols = df.columns
        schema = spec["schema"]
        if isinstance(schema, tuple):
            pairs = [(schema[1], "answer", "predicted", None)]
        else:
            acol_e = "exo_full_answer" if "exo_full_answer" in cols else "exo_answer"
            acol_g = "ego_full_answer" if "ego_full_answer" in cols else "ego_answer"
            pairs = [("exo", acol_e, "exo_predicted", "exo_nframes"),
                     ("ego", acol_g, "ego_predicted", "ego_nframes")]

        # Gemini retried a few clips after an API error; keep the last attempt.
        if "clip_id" in cols and df.clip_id.duplicated().any():
            before = len(df)
            df = df.drop_duplicates(subset="clip_id", keep="last")
            print(f"deduped {before - len(df)} retry row(s) in {spec['path']}")

        for _, r in df.iterrows():
            gt = str(r["ground_truth"]).strip()
            for view, acol, pcol, ncol in pairs:
                if acol not in cols:
                    continue
                raw = r[acol] if isinstance(r[acol], str) else ""
                pred = parse(raw, spec["prompt"])
                gt_eff = to_binary_gt(gt) if spec["prompt"] == "binary" else gt
                nf = r[ncol] if ncol and ncol in cols and pd.notna(r[ncol]) else None
                rows.append(dict(
                    activity=spec["activity"], model=spec["model"],
                    sampling=spec["sampling"], prompt=spec["prompt"], view=view,
                    file=spec["path"], clip_id=r.get("clip_id"), take=r.get("take_folder"),
                    nframes=(int(nf) if nf is not None else None),
                    gt=gt_eff, raw=raw, pred=pred,
                    stored_pred=(str(r[pcol]).strip() if pcol in cols and pd.notna(r[pcol]) else None),
                ))
    return pd.DataFrame(rows)


def main():
    df = load_all()
    df["valid"] = df["pred"].notna() & (df["pred"] != "Unparseable")
    df["correct"] = df["valid"] & (df["pred"] == df["gt"])
    df.to_csv(os.path.join(OUT, "master_rows.csv"), index=False)

    g = df.groupby(["activity", "model", "sampling", "prompt", "view"], dropna=False)
    s = g.agg(n=("gt", "size"), n_valid=("valid", "sum"), n_correct=("correct", "sum")).reset_index()
    s["acc"] = (100 * s.n_correct / s.n_valid.where(s.n_valid > 0)).round(1)
    s["invalid_rate"] = (100 * (s.n - s.n_valid) / s.n).round(1)
    s.to_csv(os.path.join(OUT, "master_summary.csv"), index=False)
    print(s.to_string(index=False))

    d = df[df.stored_pred.notna() & df.valid & (df.stored_pred != "Unknown")]
    dis = d[d.pred != d.stored_pred]
    print(f"\nDisagreements with stored predictions: {len(dis)}/{len(d)}")
    with open(os.path.join(OUT, "parse_disagreements.txt"), "w") as f:
        for _, r in dis.iterrows():
            f.write(f"[{r['file']} | {r['view']}] stored={r['stored_pred']!r} "
                    f"mine={r['pred']!r} gt={r['gt']!r}\n  RAW: {r['raw'][:700]}\n\n")


if __name__ == "__main__":
    main()
