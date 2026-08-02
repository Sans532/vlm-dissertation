"""
Re-extract skill-level labels for dissertation_v2/results/videollava/reasoning_n8_test03.csv.

Same bug class as fix_reasoning_extraction.py, plus a worse variant: the
original extractor's "conclusion" regex matched the first full label string
after "overall/therefore/in conclusion" regardless of negation, so sentences
like "Overall, the person is not yet at a late expert level, but they are not
a novice either." got read as a positive "Late Expert" verdict, when the
model's actual stated conclusion (earlier in the same answer) was
"appears to be at an intermediate skill level."

New priority order:
  1. An explicit final verdict sentence ("overall/in conclusion/in this
     case ... skill level is/appears to be X", or "... person appears to be
     an X"), which structurally can't collide with the "not yet at an X
     level" negation phrasing (different word order).
  2. The model's opening self-assessment ("appears to be at a(n) X
     (expert )?skill level").
  3. Hedge detection -> Unknown, when the answer explicitly says it cannot
     determine the level (avoids picking an arbitrary mentioned label out of
     a list of possibilities).
  4. Negation-aware substring fallback (same as the climbing-entire fix).
"""
import csv, re
from collections import Counter

SRC = "/Users/sans/Documents/Msc courses/vlm-dissertation/dissertation_v2/results/videollava/reasoning_n8_test03.csv"
DST = "/Users/sans/Documents/Msc courses/vlm-dissertation/dissertation_v2/results/videollava/reasoning_n8_test03_fixed.csv"
LOG = "/Users/sans/Documents/Msc courses/vlm-dissertation/dissertation_v2/logs/videollava/reasoning_n8_test03_fixed.log"
CHANGES_CSV = "/Users/sans/Documents/Msc courses/vlm-dissertation/dissertation_v2/logs/videollava/reasoning_n8_test03_fixed_changes.csv"

LABEL = r"(late\s+expert|intermediate\s+expert|early\s+expert|novice|late|intermediate|early)"

CONCLUSION_RE = re.compile(
    rf"(?:overall|therefore|in conclusion|in this case)[, ]*(?:the\s+)?(?:person'?s?\s+)?"
    rf"(?:skill level\s+)?(?:is|appears to be|seems to be)\s+(?:an?\s+)?(?:the\s+)?{LABEL}",
    re.I,
)
SKILL_LEVEL_CONCLUSION_RE = re.compile(
    rf"skill level\s+(?:is|appears to be|seems to be|can be described as)\s+(?:an?\s+)?{LABEL}",
    re.I,
)
OPENING_RE = re.compile(
    rf"appears to be at (?:an?\s+)?{LABEL}(?:\s+skill)?\s+(?:skill\s+)?level",
    re.I,
)
NEG_WINDOW_RE = re.compile(
    r"not\b(?:\s+\w+){0,4}?\s+(novice|late\s+expert|intermediate\s+expert|early\s+expert|expert)",
    re.I,
)
FULL_LABEL_RE = re.compile(r"(late\s+expert|intermediate\s+expert|early\s+expert|novice)", re.I)
HEDGE_RE = re.compile(
    r"anywhere from|could range from|it is possible that they are|"
    r"difficult to (?:accurately )?determine|not possible to (?:accurately )?determine|"
    r"cannot (?:accurately )?determine|is not possible to determine",
    re.I,
)
DISJUNCTION_RE = re.compile(
    rf"{LABEL}\s+or\s+(?:an?\s+|the\s+)?{LABEL}",
    re.I,
)


def norm(tok):
    t = tok.lower()
    if "late" in t:
        return "Late Expert"
    if "intermediate" in t:
        return "Intermediate Expert"
    if "early" in t:
        return "Early Expert"
    if "novice" in t:
        return "Novice"
    return None


def extract_label(answer):
    if answer == "ERROR":
        return "Unknown"

    m = SKILL_LEVEL_CONCLUSION_RE.search(answer)
    if m:
        return norm(m.group(1))

    m = CONCLUSION_RE.search(answer)
    if m:
        return norm(m.group(1))

    m = OPENING_RE.search(answer)
    if m:
        return norm(m.group(1))

    if HEDGE_RE.search(answer):
        return "Unknown"

    if DISJUNCTION_RE.search(answer):
        return "Unknown"

    cleaned = NEG_WINDOW_RE.sub(" ", answer)
    matches = FULL_LABEL_RE.findall(cleaned)
    distinct = {norm(x) for x in matches}
    if len(distinct) >= 3:
        return "Unknown"
    if matches:
        return norm(matches[0])
    return "Unknown"


rows = list(csv.DictReader(open(SRC)))
fieldnames = ["clip_id", "take_folder", "ground_truth",
              "exo_full_answer", "exo_predicted", "exo_correct",
              "ego_full_answer", "ego_predicted", "ego_correct"]

stats = {"exo": [0, 0], "ego": [0, 0]}
preds = {"exo": Counter(), "ego": Counter()}
changed = {"exo": [], "ego": []}

with open(DST, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        gt = r["ground_truth"]
        out = {"clip_id": r["clip_id"], "take_folder": r["take_folder"], "ground_truth": gt}
        for view in ["exo", "ego"]:
            ans = r[f"{view}_full_answer"]
            old_pred = r[f"{view}_predicted"]
            pred = extract_label(ans)
            ok = pred.lower() == gt.lower()
            out[f"{view}_full_answer"] = ans
            out[f"{view}_predicted"] = pred
            out[f"{view}_correct"] = ok
            stats[view][1] += 1
            if ok:
                stats[view][0] += 1
            preds[view][pred] += 1
            if pred != old_pred:
                changed[view].append((r["take_folder"], old_pred, pred, gt))
        writer.writerow(out)

lines = []
lines.append(f"Wrote {DST}\n")
lines.append("=" * 60)
lines.append("RESULTS -- VideoLLaVA reasoning cam03 test (fixed extraction)")
lines.append("=" * 60)
for view, (c, t) in stats.items():
    lines.append(f"  {view}: {c}/{t} = {c/t:.1%}")
lines.append("\nRandom chance: 25%\n")
for view in ["exo", "ego"]:
    lines.append(f"{view.capitalize()} predictions: {dict(preds[view].most_common())}")
lines.append("=" * 60)
lines.append(f"\nRelabeled rows: exo={len(changed['exo'])}, ego={len(changed['ego'])}")
for view in ["exo", "ego"]:
    lines.append(f"\n--- {view} changes (take_folder: old -> new [gt]) ---")
    for tf, old, new, gt in changed[view]:
        lines.append(f"  {tf}: {old} -> {new} [{gt}]")

report = "\n".join(lines)
print(report)
with open(LOG, "w") as f:
    f.write(report + "\n")

with open(CHANGES_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["view", "take_folder", "old_predicted", "new_predicted", "ground_truth"])
    for view in ["exo", "ego"]:
        for tf, old, new, gt in changed[view]:
            writer.writerow([view, tf, old, new, gt])
print(f"\nWrote {CHANGES_CSV}")
