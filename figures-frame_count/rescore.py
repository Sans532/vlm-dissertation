"""Re-score all VLM proficiency-estimation result CSVs from raw model answers.

Loads every canonical results file for qwen / videollava x climbing / dance,
re-parses the predicted class from the full answer text (the stored
`*_predicted` columns are unreliable), and emits:
  - figures/master_rows.csv   : one row per (file, clip, view) with re-parsed prediction
  - figures/master_summary.csv: accuracy per experimental condition
"""

import os
import re
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))

FOUR = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]

# ---------------------------------------------------------------- manifest --
# (path, activity, model, condition, frames, prompt, views_spec)
# views_spec: "paired_full" (exo/ego_full_answer), "paired_short" (exo/ego_answer),
#             "binary_n8_qwen" (binary_exo_answer), ("single", view)
M = []


def add(path, activity, model, condition, frames, prompt, schema):
    M.append(dict(path=path, activity=activity, model=model, condition=condition,
                  frames=frames, prompt=prompt, schema=schema))


# The 32-frame condition (diss_climb/results/{qwen,videollava}/binary_n32.csv) is
# deliberately excluded: the study reports the 8- vs 16-frame comparison only.

# --- climbing / qwen ---
add("diss_climb/results/qwen/qwen_climbing_entire_n8_binary.csv", "climbing", "qwen", "entire", 8, "binary", "binary_n8_qwen")
add("diss_climb/results/qwen/qwen_climbing_entire_n16_binary.csv", "climbing", "qwen", "entire", 16, "binary", "paired_short")
add("diss_climb/results/qwen/qwen_climbing_entire_n64_binary.csv", "climbing", "qwen", "entire", 64, "binary", "paired_short")
for n in (8, 16):
    add(f"diss_climb/results/qwen/qwen_climbing_entire_n{n}_fourclass.csv", "climbing", "qwen", "entire", n, "fourclass", "paired_short")
    add(f"diss_climb/results/qwen/qwen_climbing_entire_n{n}_structured.csv", "climbing", "qwen", "entire", n, "structured", "paired_full")
    add(f"diss_climb/results/qwen/qwen_climbing_entire_n{n}_reasoning.csv", "climbing", "qwen", "entire", n, "reasoning", "paired_full")
add("diss_climb/results/qwen/qwen_climbing_entire_n64_fourclass.csv", "climbing", "qwen", "entire", 64, "fourclass", "paired_full")
add("diss_climb/results/qwen/qwen_climbing_entire_n64_structured.csv", "climbing", "qwen", "entire", 64, "structured", "paired_full")
add("diss_climb/results/qwen/qwen_climbing_entire_n64_reasoning.csv", "climbing", "qwen", "entire", 64, "reasoning", "paired_full")
for n in (8, 16):
    add(f"diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_exo_n{n}_binary.csv", "climbing", "qwen", "trimmed", n, "binary", ("single", "exo"))
    add(f"diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_ego_n{n}_binary.csv", "climbing", "qwen", "trimmed", n, "binary", ("single", "ego"))
    add(f"diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n{n}_structured.csv", "climbing", "qwen", "trimmed", n, "structured", "paired_full")
    add(f"diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n{n}_reasoning.csv", "climbing", "qwen", "trimmed", n, "reasoning", "paired_full")
add("diss_climb/results/qwen/trimmed/fourclass_trimmed.csv", "climbing", "qwen", "trimmed", None, "fourclass", "paired_short")

# --- climbing / videollava ---
add("diss_climb/results/videollava/vl_climbing_entire_n8_binary.csv", "climbing", "videollava", "entire", 8, "binary", "paired_short")
add("diss_climb/results/videollava/vl_climbing_entire_n16_binary.csv", "climbing", "videollava", "entire", 16, "binary", "paired_short")
add("diss_climb/results/videollava/vl_climbing_entire_n8_fourclass.csv", "climbing", "videollava", "entire", 8, "fourclass", "paired_short")
add("diss_climb/results/videollava/vl_climbing_entire_n8_structured.csv", "climbing", "videollava", "entire", 8, "structured", "paired_full")
add("diss_climb/results/videollava/vl_climbing_entire_n8_reasoning_fixed.csv", "climbing", "videollava", "entire", 8, "reasoning", "paired_full")
add("diss_climb/results/videollava/structured_n16_vl.csv", "climbing", "videollava", "entire", 16, "structured", "paired_full")
for n in (8, 16):
    add(f"diss_climb/results/videollava/trimmed/vl_climbing_trimmed_exo_n{n}_binary.csv", "climbing", "videollava", "trimmed", n, "binary", ("single", "exo"))
    add(f"diss_climb/results/videollava/trimmed/vl_climbing_trimmed_ego_n{n}_binary.csv", "climbing", "videollava", "trimmed", n, "binary", ("single", "ego"))
add("diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_fourclass.csv", "climbing", "videollava", "trimmed", 8, "fourclass", "paired_short")
add("diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_structured.csv", "climbing", "videollava", "trimmed", 8, "structured", "paired_full")
add("diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_reasoning.csv", "climbing", "videollava", "trimmed", 8, "reasoning", "paired_full")
add("diss_climb/results/videollava/trimmed/fourclass_n16.csv", "climbing", "videollava", "trimmed", 16, "fourclass", "paired_short")
add("diss_climb/results/videollava/trimmed/reasoning_n16.csv", "climbing", "videollava", "trimmed", 16, "reasoning", "paired_full")

# --- dance / qwen ---
for cond, sub in (("entire", ""), ("trimmed", "trimmed/")):
    for n in (8, 16):
        add(f"diss_dance/results/qwen/{sub}qwen_dance_{cond}_n{n}_binary.csv", "dance", "qwen", cond, n, "binary", "paired_short")
        add(f"diss_dance/results/qwen/{sub}qwen_dance_{cond}_n{n}_fourclass.csv", "dance", "qwen", cond, n, "fourclass", "paired_short")
        add(f"diss_dance/results/qwen/{sub}qwen_dance_{cond}_n{n}_structured.csv", "dance", "qwen", cond, n, "structured", "paired_full")
        add(f"diss_dance/results/qwen/{sub}qwen_dance_{cond}_n{n}_reasoning.csv", "dance", "qwen", cond, n, "reasoning", "paired_full")
add("diss_dance/results/qwen/qwen_dance_entire_n64_binary.csv", "dance", "qwen", "entire", 64, "binary", "paired_short")
add("diss_dance/results/qwen/qwen_dance_entire_n64_fourclass.csv", "dance", "qwen", "entire", 64, "fourclass", "paired_full")
add("diss_dance/results/qwen/qwen_dance_entire_n64_structured.csv", "dance", "qwen", "entire", 64, "structured", "paired_full")
add("diss_dance/results/qwen/qwen_dance_entire_n64_reasoning.csv", "dance", "qwen", "entire", 64, "reasoning", "paired_full")

# --- dance / videollava ---
for cond, sub in (("entire", ""), ("trimmed", "trimmed/")):
    add(f"diss_dance/results/videollava/{sub}vl_dance_{cond}_n8_binary.csv", "dance", "videollava", cond, 8, "binary", "paired_short")
    add(f"diss_dance/results/videollava/{sub}vl_dance_{cond}_n16_binary.csv", "dance", "videollava", cond, 16, "binary", "paired_short")
    add(f"diss_dance/results/videollava/{sub}vl_dance_{cond}_n8_fourclass.csv", "dance", "videollava", cond, 8, "fourclass", "paired_short")
    add(f"diss_dance/results/videollava/{sub}vl_dance_{cond}_n8_structured.csv", "dance", "videollava", cond, 8, "structured", "paired_full")
    add(f"diss_dance/results/videollava/{sub}vl_dance_{cond}_n8_reasoning.csv", "dance", "videollava", cond, 8, "reasoning", "paired_full")

# ----------------------------------------------------------------- parsing --

NEG = re.compile(
    r"(?:\bnot\b|\bisn'?t\b|\bno longer\b|\brather than\b|\bbeyond\b|\binstead of\b"
    r"|\bmore than\b|\bbecom(?:e|ing)\b|\bway to becoming\b|\bprogress(?:ed|ing)? (?:from|to|toward)\b"
    r"|\bdevelop(?:ing)? into\b|\bpotential to (?:be|become)\b|\bcloser to\b|\bapproaching\b"
    r"|\btransition(?:ing)? (?:to|toward)\b|\bfrom\b|\bpast\b|\baspiring\b|\bon (?:his|her|their) way to\b)"
    r"\s+(?:(?:a|an|the|merely|just|complete|total|absolute|mere|typical|being)\s+){0,3}$")

PHRASES = [
    ("late expert", "Late Expert"),
    ("advanced expert", "Late Expert"),
    ("intermediate expert", "Intermediate Expert"),
    ("early expert", "Early Expert"),
    ("novice", "Novice"),
    ("beginner", "Novice"),
    ("intermediate", "Intermediate Expert"),
    ("expert", "Expert"),  # bare 'expert' — only meaningful for binary
]

# cues that mark a sentence as carrying the model's verdict
VERDICT_CUE = re.compile(
    r"appears? to be|seems? to be|is likely|would (?:classify|rate|place|categor)"
    r"|classif(?:y|ied)|described as|skill level|overall|in (?:summary|conclusion)"
    r"|therefore|i would say|suggest(?:s|ing)? (?:that )?(?:the|a|an|this)?", re.I)


def find_mentions(text):
    """Return list of (pos, label) mentions, longest-phrase-first, no overlaps, negations dropped."""
    low = text.lower()
    taken = [False] * len(low)
    mentions = []
    for phrase, label in PHRASES:
        for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            ctx = low[max(0, m.start() - 30):m.start()]
            if NEG.search(ctx):
                continue
            mentions.append((m.start(), label))
    mentions.sort()
    return mentions


def parse_binary(text):
    if not isinstance(text, str) or not text.strip() or text.strip().upper() == "ERROR":
        return None
    mentions = [(p, ("Expert" if l != "Novice" else "Novice"))
                for p, l in find_mentions(text)]
    labels = {l for _, l in mentions}
    if len(labels) == 1:
        return labels.pop()
    if not mentions:
        return "Unparseable"
    return mentions[-1][1]  # conflicting mentions: final verdict wins


def parse_fourclass(text, structured=False):
    if not isinstance(text, str) or not text.strip() or text.strip().upper() == "ERROR":
        return None
    t = text
    if structured:
        # take the text after the LAST "Skill Level:" marker if present
        parts = re.split(r"skill\s*level\s*:", t, flags=re.I)
        if len(parts) > 1:
            tail = parts[-1].strip()
            m = find_mentions(tail)
            if m:
                lab = m[0][1]
                return lab if lab != "Expert" else "Unparseable"
    mentions = [(p, l) for p, l in find_mentions(t) if l != "Expert"]
    if not mentions:
        return "Unparseable"
    labels = {l for _, l in mentions}
    if len(labels) == 1:
        return mentions[0][1]
    # conflict: prefer mentions in a sentence carrying a verdict cue
    sent_bounds = [m.start() for m in re.finditer(r"[.!?\n]", t)]

    def sentence_of(pos):
        lo = max([b for b in sent_bounds if b < pos], default=-1) + 1
        hi = min([b for b in sent_bounds if b >= pos], default=len(t))
        return t[lo:hi]

    cued = [(p, l) for p, l in mentions if VERDICT_CUE.search(sentence_of(p))]
    if cued:
        if len({l for _, l in cued}) == 1:
            return cued[0][1]
        return cued[-1][1]  # multiple verdict sentences: last one wins
    # no verdict cue anywhere: majority vote, tie broken by first mention
    from collections import Counter
    counts = Counter(l for _, l in mentions)
    top = counts.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return mentions[0][1]


def parse(text, prompt):
    if prompt == "binary":
        return parse_binary(text)
    return parse_fourclass(text, structured=(prompt == "structured"))


# ----------------------------------------------------------------- loading --

def gt_of(row, cols):
    for c in ("gt_binary", "ground_truth"):
        if c in cols:
            return str(row[c]).strip()
    raise KeyError("no ground truth col")


def to_binary_gt(gt):
    return "Novice" if gt == "Novice" else "Expert"


def load_all():
    rows = []
    for spec in M:
        path = os.path.join(BASE, spec["path"])
        if not os.path.exists(path):
            print("MISSING:", spec["path"]); continue
        df = pd.read_csv(path)
        if df.empty:
            print("EMPTY:", spec["path"]); continue
        cols = df.columns
        schema = spec["schema"]
        if isinstance(schema, tuple):  # single-view file
            view = schema[1]
            pairs = [(view, "answer", "predicted")]
        elif schema == "binary_n8_qwen":
            pairs = [("exo", "binary_exo_answer", None), ("ego", "binary_ego_answer", None)]
        elif schema == "paired_short":
            pairs = [("exo", "exo_answer", "exo_predicted" if "exo_predicted" in cols else None),
                     ("ego", "ego_answer", "ego_predicted" if "ego_predicted" in cols else None)]
        else:  # paired_full
            pairs = [("exo", "exo_full_answer", "exo_predicted"),
                     ("ego", "ego_full_answer", "ego_predicted")]
        for _, r in df.iterrows():
            gt = gt_of(r, cols)
            for view, acol, pcol in pairs:
                if acol not in cols:
                    continue
                raw = r[acol]
                pred = parse(raw if isinstance(raw, str) else "", spec["prompt"])
                gt_eff = to_binary_gt(gt) if spec["prompt"] == "binary" else gt
                rows.append(dict(
                    activity=spec["activity"], model=spec["model"],
                    condition=spec["condition"], frames=spec["frames"],
                    prompt=spec["prompt"], view=view, file=spec["path"],
                    clip_id=r.get("clip_id"), take=r.get("take_folder"),
                    gt=gt_eff, raw=raw if isinstance(raw, str) else "",
                    pred=pred,
                    stored_pred=(str(r[pcol]).strip() if pcol and pd.notna(r[pcol]) else None),
                ))
    return pd.DataFrame(rows)


def main():
    df = load_all()
    df["valid"] = df["pred"].notna() & (df["pred"] != "Unparseable")
    df["correct"] = df["valid"] & (df["pred"] == df["gt"])
    df.to_csv(os.path.join(OUT, "master_rows.csv"), index=False)

    g = df.groupby(["activity", "model", "condition", "frames", "prompt", "view"], dropna=False)
    summ = g.agg(n=("gt", "size"), n_valid=("valid", "sum"), n_correct=("correct", "sum")).reset_index()
    summ["acc_valid"] = (100 * summ.n_correct / summ.n_valid.where(summ.n_valid > 0)).round(1)
    summ["acc_all"] = (100 * summ.n_correct / summ.n).round(1)
    summ["error_rate"] = (100 * (summ.n - summ.n_valid) / summ.n).round(1)
    summ.to_csv(os.path.join(OUT, "master_summary.csv"), index=False)
    print(summ.to_string(index=False))

    # parser vs stored disagreement report
    d = df[df.stored_pred.notna() & df.valid & (df.stored_pred != "Unknown")]
    dis = d[d.pred != d.stored_pred]
    print(f"\nDisagreements with stored predictions: {len(dis)}/{len(d)}")
    with open(os.path.join(OUT, "parse_disagreements.txt"), "w") as f:
        for _, r in dis.iterrows():
            f.write(f"[{r['file']} | {r['view']}] stored={r['stored_pred']!r} mine={r['pred']!r} gt={r['gt']!r}\n  RAW: {r['raw'][:600]}\n\n")


if __name__ == "__main__":
    main()
