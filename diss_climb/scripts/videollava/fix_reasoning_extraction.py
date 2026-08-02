"""
Re-extract skill-level labels from the already-generated Video-LLaVA reasoning
answers (dissertation_v2/results/videollava/vl_climbing_entire_n8_reasoning.csv),
without re-running the model.

Bug in the original extract_label() (scripts/videollava/reasoning_n8.py): it only
matched the four full label strings ("Late Expert", "Intermediate Expert",
"Early Expert", "Novice") as raw substrings. Video-LLaVA very often states its
verdict as "at an intermediate skill level" (no "expert" suffix), so those
answers fell through to "Unknown". The substring check was also order-based
rather than meaning-based, so negated mentions like "not a complete novice"
were matched as a positive "Novice" verdict.

This script re-derives predicted/correct columns with a smarter extractor and
writes a new CSV alongside the original, plus prints updated accuracy.
"""
import csv, re
from collections import Counter

SRC = "/Users/sans/Documents/Msc courses/vlm-dissertation/dissertation_v2/results/videollava/vl_climbing_entire_n8_reasoning.csv"
DST = "/Users/sans/Documents/Msc courses/vlm-dissertation/dissertation_v2/results/videollava/vl_climbing_entire_n8_reasoning_fixed.csv"

FULL_LABEL_RE = re.compile(r"(late\s+expert|intermediate\s+expert|early\s+expert|novice)", re.I)
CONCLUSION_RE = re.compile(r"(?:overall|therefore|in conclusion)[^.]*?\b(late\s+expert|intermediate\s+expert|early\s+expert|novice)\b", re.I)
SKILL_LEVEL_RE = re.compile(r"skill level[, ]*(?:is|appears to be|seems to be)\s+(?:an?\s+)?(late\s+expert|intermediate\s+expert|early\s+expert|novice|late|intermediate|early)\b", re.I)
NEG_WINDOW_RE = re.compile(r"not\b(?:\s+\w+){0,4}?\s+(novice|late\s+expert|intermediate\s+expert|early\s+expert|expert)", re.I)
SKILL_BARE_RE = re.compile(r"\b(late|intermediate|early)\s+skill\b", re.I)
HEDGE_RE = re.compile(r"anywhere from|could be anywhere|it is possible that they are", re.I)


def norm(tok):
    t = tok.lower()
    if "late" in t: return "Late Expert"
    if "intermediate" in t: return "Intermediate Expert"
    if "early" in t: return "Early Expert"
    if "novice" in t: return "Novice"
    return None


def extract_label(answer):
    if answer == "ERROR":
        return "Unknown"
    m = SKILL_LEVEL_RE.search(answer)
    if m:
        return norm(m.group(1))
    m = CONCLUSION_RE.search(answer)
    if m:
        return norm(m.group(1))
    if HEDGE_RE.search(answer):
        return "Unknown"
    cleaned = NEG_WINDOW_RE.sub(" ", answer)
    matches = FULL_LABEL_RE.findall(cleaned)
    distinct = {norm(x) for x in matches}
    if len(distinct) >= 3:
        return "Unknown"
    if matches:
        return norm(matches[0])
    m = SKILL_BARE_RE.search(cleaned)
    if m:
        return norm(m.group(1))
    return "Unknown"


rows = list(csv.DictReader(open(SRC)))
fieldnames = ["clip_id", "take_folder", "ground_truth",
              "exo_full_answer", "exo_predicted", "exo_correct",
              "ego_full_answer", "ego_predicted", "ego_correct"]

stats = {"exo": [0, 0], "ego": [0, 0]}
preds = {"exo": Counter(), "ego": Counter()}

with open(DST, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        gt = r["ground_truth"]
        out = {"clip_id": r["clip_id"], "take_folder": r["take_folder"], "ground_truth": gt}
        for view in ["exo", "ego"]:
            ans = r[f"{view}_full_answer"]
            pred = extract_label(ans)
            ok = pred.lower() == gt.lower()
            out[f"{view}_full_answer"] = ans
            out[f"{view}_predicted"] = pred
            out[f"{view}_correct"] = ok
            stats[view][1] += 1
            if ok:
                stats[view][0] += 1
            preds[view][pred] += 1
        writer.writerow(out)

LOG = "/Users/sans/Documents/Msc courses/vlm-dissertation/dissertation_v2/logs/videollava/reasoning_n8_fixed.log"

lines = []
lines.append(f"Wrote {DST}\n")
lines.append("=" * 60)
lines.append("RESULTS — VideoLLaVA reasoning prompt 8 frames (fixed extraction)")
lines.append("=" * 60)
for view, (c, t) in stats.items():
    lines.append(f"  {view}: {c}/{t} = {c/t:.1%}")
lines.append("\nRandom chance: 25%\n")
for view in ["exo", "ego"]:
    lines.append(f"{view.capitalize()} predictions: {dict(preds[view].most_common())}")
lines.append("=" * 60)

report = "\n".join(lines)
print(report)
with open(LOG, "w") as f:
    f.write(report + "\n")
