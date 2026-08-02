"""
Consolidates every binary / fourclass / structured / reasoning result CSV
across dissertation_v2/results/ and diss_dance/results/ into one summary,
using corrected label extraction wherever the original extractor was found
to be buggy during this project's audit.

Background: throughout the audit, every 'structured' (Observations: / Errors:
/ Skill Level: ...) and 'binary' (Novice/Expert only) file's extraction logic
was verified to be clean -- their scripts anchor on either the mandatory
"Skill Level:" line or a two-word vocabulary, so a fixed-priority substring
scan can't misfire the way it does on free text. Those files' stored
predicted/correct columns are trusted as-is here.

Every 'reasoning' (free-form "Explain your reasoning") and 'fourclass'
("Answer only one: ...") file, across Gemini / Qwen / VideoLLaVA, climbing /
dance, entire / trimmed, n8 / n16, was found to use one of two buggy
extractors:
  - a naive fixed-priority substring scan (checks "Late Expert" before
    "Intermediate Expert" before "Early Expert" before "Novice", anywhere in
    the text) -- systematically misreads negated comparisons like "not yet a
    Late Expert" as a positive Late Expert verdict, and arbitrarily resolves
    hedges like "novice or early expert" to whichever label is checked first.
  - a "conclusion sentence" regex with no negation guard, same failure mode.

This script re-extracts every reasoning/fourclass row from its raw stored
answer text using a single corrected extractor that:
  1. Honors an explicit leading standalone verdict ("Novice\n...").
  2. Treats an unresolved "X or Y" disjunction as Unknown rather than
     arbitrarily picking a side.
  3. Matches an explicit "skill level is/appears to be X" statement.
  4. Matches an explicit conclusion sentence ("Overall/In conclusion/In this
     case ... X"), only after stripping negated mentions so "not yet a Late
     Expert" can't be picked up.
  5. Matches an opening "appears to be at an X (skill) level" statement.
  6. Falls back to a hedge check ("difficult to determine" etc.) -> Unknown.
  7. Falls back to negation-aware first-mention substring matching.

Output: statistics/master_results_summary.csv
"""
import csv
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIRS = [REPO_ROOT / "dissertation_v2" / "results", REPO_ROOT / "diss_dance" / "results"]
OUT_CSV = REPO_ROOT / "statistics" / "master_results_summary.csv"

CANONICAL_4CLASS = {"novice", "early expert", "intermediate expert", "late expert"}
LABELS_ORDER = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]

# ---------------------------------------------------------------------------
# Corrected free-text extractor (used for reasoning / fourclass row re-extraction)
# ---------------------------------------------------------------------------
LEADING_LABEL_RE = re.compile(r"^\s*\**\s*(late\s+expert|intermediate\s+expert|early\s+expert|novice)\s*\**\s*(?:[\n:.]|$)", re.I)
FULL_LABEL_RE = re.compile(r"(late\s+expert|intermediate\s+expert|early\s+expert|novice)", re.I)
CONCLUSION_RE = re.compile(r"(?:overall|therefore|in conclusion|in this case)[^.]*?\b(late\s+expert|intermediate\s+expert|early\s+expert|novice)\b", re.I)
SKILL_LEVEL_RE = re.compile(r"skill level[, ]*(?:is|appears to be|seems to be|can be described as)\s+(?:an?\s+)?(late\s+expert|intermediate\s+expert|early\s+expert|novice|late|intermediate|early)\b", re.I)
NEG_WINDOW_RE = re.compile(r"not\b(?:\s+\w+){0,4}?\s+(novice|late\s+expert|intermediate\s+expert|early\s+expert|expert)", re.I)
SKILL_BARE_RE = re.compile(r"\b(late|intermediate|early)\s+skill\b", re.I)
HEDGE_RE = re.compile(r"anywhere from|could range from|it is possible that they are|difficult to (?:accurately )?determine|not possible to (?:accurately )?determine|cannot (?:accurately )?determine", re.I)
OPENING_RE = re.compile(r"appears to be at (?:an?\s+)?(late\s+expert|intermediate\s+expert|early\s+expert|novice|late|intermediate|early)(?:\s+skill)?\s+(?:skill\s+)?level", re.I)
DISJUNCTION_RE = re.compile(r"(novice|early expert|intermediate expert|late expert)\s+or\s+(?:an?\s+|the\s+)?(novice|early expert|intermediate expert|late expert)", re.I)
VERDICT_WINDOW = 400  # the model's real verdict is always near the start; later text often
                      # restates a label as a comparison/coaching aside, so hedge/disjunction/
                      # fallback checks are restricted to this window rather than the whole answer.


def norm(tok):
    t = tok.lower()
    if "late" in t: return "Late Expert"
    if "intermediate" in t: return "Intermediate Expert"
    if "early" in t: return "Early Expert"
    if "novice" in t: return "Novice"
    return None


def extract_label_corrected(answer):
    if not answer or answer == "ERROR":
        return "Unknown"
    m = LEADING_LABEL_RE.search(answer)
    if m:
        return norm(m.group(1))

    window = answer[:VERDICT_WINDOW]
    if DISJUNCTION_RE.search(window):
        return "Unknown"
    m = SKILL_LEVEL_RE.search(answer)
    if m:
        return norm(m.group(1))
    cleaned_for_conclusion = NEG_WINDOW_RE.sub(" ", answer)
    m = CONCLUSION_RE.search(cleaned_for_conclusion)
    if m:
        return norm(m.group(1))
    m = OPENING_RE.search(answer)
    if m:
        return norm(m.group(1))
    if HEDGE_RE.search(window):
        return "Unknown"

    cleaned_window = NEG_WINDOW_RE.sub(" ", window)
    matches_window = FULL_LABEL_RE.findall(cleaned_window)
    distinct_window = {norm(x) for x in matches_window}
    if len(distinct_window) >= 3:
        return "Unknown"
    if matches_window:
        return norm(matches_window[0])

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


# ---------------------------------------------------------------------------
# Filename / metadata parsing (same conventions as run_significance_tests.py)
# ---------------------------------------------------------------------------
def classify_filename(name):
    n = name.lower()
    if "binary" in n:
        return "binary"
    if "fourclass" in n:
        return "fourclass"
    if "structured" in n:
        return "structured"
    if "reasoning" in n:
        return "reasoning"
    return None


def parse_meta(path: Path):
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()

    activity = "climbing" if "dissertation_v2" in parts else ("dance" if "diss_dance" in parts else "unknown")

    if "gemini" in parts or "gemini" in name:
        model = "gemini"
    elif "qwen" in parts or "qwen" in name:
        model = "qwen"
    elif "videollava" in parts or re.search(r"\bvl_", name):
        model = "videollava"
    else:
        model = "unknown"

    trim = "trimmed" if ("trimmed" in parts or "trimmed" in name) else "entire"
    m = re.search(r"(?:^|_)n(\d+)(?:_|\.|$)", name)
    frames = m.group(1) if m else ""

    return model, activity, trim, frames


def resolve_single_view(filename):
    n = filename.lower()
    if "_ego_" in n or n.startswith("ego_") or "_ego." in n:
        return "ego"
    if "_exo_" in n or n.startswith("exo_") or "_exo." in n:
        return "exo"
    return "single"


def find_columns_ending(header, suffix):
    out = {}
    for col in header:
        cl = col.lower()
        if not cl.endswith(suffix):
            continue
        if "exo" in cl:
            out["exo"] = col
        elif "ego" in cl:
            out["ego"] = col
        else:
            out["single"] = col
    return out


def to_bool(val):
    if val is None:
        return None
    v = val.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return None


def find_ground_truth_column(header):
    for candidate in ("ground_truth", "gt_binary"):
        if candidate in header:
            return candidate
    return None


# ---------------------------------------------------------------------------
def process_binary(path, rows, header, warnings):
    """Binary files: trust stored correct column (verified clean)."""
    correct_cols = find_columns_ending(header, "correct")
    gt_col = find_ground_truth_column(header)
    if not correct_cols or gt_col is None:
        warnings.append(f"{path}: binary file missing correct/ground_truth column -- skipping")
        return []
    predicted_cols = find_columns_ending(header, "predicted")

    out = []
    for view, col in correct_cols.items():
        actual_view = view if view != "single" else resolve_single_view(path.name)
        bools = [to_bool(r.get(col)) for r in rows]
        bools = [b for b in bools if b is not None]
        n = len(bools)
        if n == 0:
            continue
        acc = sum(bools) / n

        # class breakdown by ground truth (Novice vs Expert-ish)
        gt_values = sorted({r.get(gt_col, "").strip() for r in rows if r.get(gt_col)})
        breakdown = {}
        for gt_val in gt_values:
            subset = [r for r in rows if r.get(gt_col) == gt_val]
            sub_bools = [to_bool(r.get(col)) for r in subset]
            sub_bools = [b for b in sub_bools if b is not None]
            if sub_bools:
                breakdown[gt_val] = f"{sum(sub_bools)}/{len(sub_bools)}={sum(sub_bools)/len(sub_bools):.1%}"

        pred_col = predicted_cols.get(view) or predicted_cols.get("single")
        dist = Counter(r.get(pred_col, "") for r in rows) if pred_col else Counter()

        out.append({
            "view": actual_view, "n": n, "accuracy": acc,
            "class_breakdown": "; ".join(f"{k}: {v}" for k, v in breakdown.items()),
            "distribution": "; ".join(f"{k}:{v}" for k, v in dist.most_common()) if dist else "",
            "extraction_status": "trusted-clean",
        })
    return out


def process_structured(path, rows, header, warnings):
    """Structured files: trust stored predicted/correct columns (verified clean)."""
    correct_cols = find_columns_ending(header, "correct")
    predicted_cols = find_columns_ending(header, "predicted")
    gt_col = find_ground_truth_column(header)
    if not correct_cols or not predicted_cols or gt_col is None:
        warnings.append(f"{path}: structured file missing predicted/correct/ground_truth -- skipping")
        return []

    gt_values = {r.get(gt_col, "").strip().lower() for r in rows if r.get(gt_col)}
    if gt_values - CANONICAL_4CLASS:
        warnings.append(f"{path}: ground truth {sorted(gt_values)} not 4-class taxonomy -- skipping")
        return []

    out = []
    for view, pred_col in predicted_cols.items():
        actual_view = view if view != "single" else resolve_single_view(path.name)
        corr_col = correct_cols.get(view) or correct_cols.get("single")
        n = len(rows)
        correct = sum(1 for r in rows if r.get(corr_col, "").strip().lower() == "true")
        breakdown = {}
        for label in LABELS_ORDER:
            subset = [r for r in rows if r.get(gt_col) == label]
            if subset:
                c = sum(1 for r in subset if r.get(corr_col, "").strip().lower() == "true")
                breakdown[label] = f"{c}/{len(subset)}={c/len(subset):.1%}"
        dist = Counter(r.get(pred_col, "") for r in rows)
        out.append({
            "view": actual_view, "n": n, "accuracy": correct / n if n else 0,
            "class_breakdown": "; ".join(f"{k}: {v}" for k, v in breakdown.items()),
            "distribution": "; ".join(f"{k}:{v}" for k, v in dist.most_common()),
            "extraction_status": "trusted-clean",
        })
    return out


GEMINI_LABEL_TOKEN_RE = re.compile(
    r"\b(late\s+expert|intermediate\s+expert|early\s+expert|late|intermediate|early|novice)\b",
    re.IGNORECASE,
)


def simple_position_extract(answer):
    """Exactly mirrors Gemini's own already-good extractor (reasoning.py): first label
    mentioned within the first 400 chars (including bare "late"/"intermediate"/"early"
    without "expert"), no negation/disjunction handling. Used only to detect whether a
    file's stored predictions already came from this clean extractor."""
    if not answer or answer == "ERROR":
        return "Unknown"
    m = GEMINI_LABEL_TOKEN_RE.search(answer[:VERDICT_WINDOW])
    if not m:
        m = GEMINI_LABEL_TOKEN_RE.search(answer)
    return norm(m.group(1)) if m else "Unknown"


def process_reasoning_or_fourclass(path, rows, header, warnings):
    """Reasoning / fourclass files: re-extract from raw answer text using the corrected
    extractor, UNLESS the stored predictions already match a clean, simple position-based
    re-extraction (i.e. the file was already generated by a good extractor -- in which case
    trust stored to avoid the broader corrected extractor over-correcting an already-fine file)."""
    answer_cols = find_columns_ending(header, "answer")
    predicted_cols = find_columns_ending(header, "predicted")
    gt_col = find_ground_truth_column(header)
    if not answer_cols or gt_col is None:
        warnings.append(f"{path}: missing answer/ground_truth column -- skipping")
        return []

    gt_values = {r.get(gt_col, "").strip().lower() for r in rows if r.get(gt_col)}
    if gt_values - CANONICAL_4CLASS:
        warnings.append(f"{path}: ground truth {sorted(gt_values)} not 4-class taxonomy -- skipping")
        return []

    out = []
    for view, ans_col in answer_cols.items():
        actual_view = view if view != "single" else resolve_single_view(path.name)
        n = len(rows)

        pred_col = predicted_cols.get(view) or predicted_cols.get("single")
        already_clean = False
        if pred_col:
            simple_preds = [simple_position_extract(r.get(ans_col, "")) for r in rows]
            stored_preds = [r.get(pred_col, "") for r in rows]
            already_clean = simple_preds == stored_preds

        if already_clean:
            recomputed = stored_preds
            status = "trusted-clean"
        else:
            recomputed = [extract_label_corrected(r.get(ans_col, "")) for r in rows]
            status = "recomputed"

        correct = sum(1 for r, p in zip(rows, recomputed) if p.lower() == r.get(gt_col, "").lower())
        breakdown = {}
        for label in LABELS_ORDER:
            idxs = [i for i, r in enumerate(rows) if r.get(gt_col) == label]
            if idxs:
                c = sum(1 for i in idxs if recomputed[i].lower() == label.lower())
                breakdown[label] = f"{c}/{len(idxs)}={c/len(idxs):.1%}"
        dist = Counter(recomputed)
        out.append({
            "view": actual_view, "n": n, "accuracy": correct / n if n else 0,
            "class_breakdown": "; ".join(f"{k}: {v}" for k, v in breakdown.items()),
            "distribution": "; ".join(f"{k}:{v}" for k, v in dist.most_common()),
            "extraction_status": status,
        })
    return out


def main():
    warnings = []
    rows_out = []

    csv_paths = []
    for d in RESULT_DIRS:
        csv_paths.extend(sorted(d.rglob("*.csv")))

    for path in csv_paths:
        # skip the scratch/derived files we generated during this audit itself
        if "_fixed" in path.name or "_patched" in path.name or path.name in ("structured_eval.csv",):
            continue

        prompt_type = classify_filename(path.name)
        if prompt_type is None:
            continue  # silently skip non-matching files (accuracy_summary.csv, 3class_structured.csv, etc.)

        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                rows = list(reader)
        except Exception as e:
            warnings.append(f"{path}: failed to read CSV ({e}) -- skipping")
            continue

        if not rows:
            warnings.append(f"{path}: empty file -- skipping")
            continue

        model, activity, trim, frames = parse_meta(path)

        if prompt_type == "binary":
            results = process_binary(path, rows, header, warnings)
        elif prompt_type == "structured":
            results = process_structured(path, rows, header, warnings)
        else:  # fourclass or reasoning
            results = process_reasoning_or_fourclass(path, rows, header, warnings)

        for res in results:
            rows_out.append({
                "model": model,
                "activity": activity,
                "trim_condition": trim,
                "frame_count": frames,
                "prompt_type": prompt_type,
                "view": res["view"],
                "file": str(path.relative_to(REPO_ROOT)),
                "n": res["n"],
                "accuracy": round(res["accuracy"], 4),
                "class_breakdown": res["class_breakdown"],
                "predicted_distribution": res["distribution"],
                "extraction_status": res["extraction_status"],
            })

    rows_out.sort(key=lambda r: (r["model"], r["activity"], r["prompt_type"], r["trim_condition"], r["frame_count"], r["view"]))

    fieldnames = ["model", "activity", "trim_condition", "frame_count", "prompt_type", "view",
                  "file", "n", "accuracy", "class_breakdown", "predicted_distribution", "extraction_status"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {OUT_CSV}")
    print(f"\nBy prompt_type: {dict(Counter(r['prompt_type'] for r in rows_out).most_common())}")
    print(f"By extraction_status: {dict(Counter(r['extraction_status'] for r in rows_out).most_common())}")

    if warnings:
        print(f"\n{len(warnings)} files skipped:")
        for w in warnings:
            print(f"  WARNING: {w}")


if __name__ == "__main__":
    main()
