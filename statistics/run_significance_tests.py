"""
Statistical significance testing for the dissertation VLM benchmark results.

Scans every result CSV under dissertation_v2/results/ and diss_dance/results/.

- Binary-prompt files (filename contains "binary"): one-sample binomial test
  of the observed accuracy against chance (p=0.5), via scipy.stats.binomtest,
  using the 'correct' column (or exo_correct/ego_correct, tested separately
  per view when both are present).

- Four-class files (filename contains "fourclass", "structured", or
  "reasoning"): chi-squared goodness-of-fit test of the observed distribution
  of predicted labels against a uniform expected distribution (25% each of
  Novice / Early Expert / Intermediate Expert / Late Expert), via
  scipy.stats.chisquare, using the 'predicted' column (or exo_predicted/
  ego_predicted, tested separately per view). Ground truth must itself be a
  subset of the four canonical labels -- files whose ground truth uses a
  different taxonomy (binary, 3-class, etc.) are skipped with a warning
  rather than forced into a 4-class test.

Files that don't match either naming convention, or whose columns/labels
don't match the expected structure, are skipped with a warning.

Output: statistics/statistical_significance_summary.csv
"""
import csv
import re
from pathlib import Path

from scipy.stats import binomtest, chisquare

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIRS = [REPO_ROOT / "dissertation_v2" / "results", REPO_ROOT / "diss_dance" / "results"]
OUT_CSV = REPO_ROOT / "statistics" / "statistical_significance_summary.csv"
ALPHA = 0.05

CANONICAL_4CLASS = {"novice", "early expert", "intermediate expert", "late expert"}
CANON_DISPLAY = {
    "novice": "Novice",
    "early expert": "Early Expert",
    "intermediate expert": "Intermediate Expert",
    "late expert": "Late Expert",
}
TRUE_STRINGS = {"true", "1", "yes"}
FALSE_STRINGS = {"false", "0", "no"}


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

    if "dissertation_v2" in parts:
        activity = "climbing"
    elif "diss_dance" in parts:
        activity = "dance"
    else:
        activity = "unknown"

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

    prompt_type = classify_filename(name)

    return model, activity, trim, frames, prompt_type


def to_bool(val):
    if val is None:
        return None
    v = val.strip().lower()
    if v in TRUE_STRINGS:
        return True
    if v in FALSE_STRINGS:
        return False
    return None


def find_correct_columns(header):
    """Return {view: column_name} for columns ending in 'correct'."""
    out = {}
    for col in header:
        cl = col.lower()
        if not cl.endswith("correct"):
            continue
        if "exo" in cl:
            out["exo"] = col
        elif "ego" in cl:
            out["ego"] = col
        else:
            out["single"] = col
    return out


def find_predicted_columns(header):
    out = {}
    for col in header:
        cl = col.lower()
        if not cl.endswith("predicted"):
            continue
        if "exo" in cl:
            out["exo"] = col
        elif "ego" in cl:
            out["ego"] = col
        else:
            out["single"] = col
    return out


def find_ground_truth_column(header):
    for candidate in ("ground_truth", "gt_binary", "ground_truth_original"):
        if candidate in header:
            return candidate
    return None


def resolve_single_view(filename):
    n = filename.lower()
    if "_ego_" in n or n.startswith("ego_") or "_ego." in n:
        return "ego"
    if "_exo_" in n or n.startswith("exo_") or "_exo." in n:
        return "exo"
    return "single"


def run_binary_test(path: Path, rows, header, warnings):
    correct_cols = find_correct_columns(header)
    if not correct_cols:
        warnings.append(f"{path}: no *_correct / correct column found -- skipping")
        return []

    results = []
    for view, col in correct_cols.items():
        actual_view = view if view != "single" else resolve_single_view(path.name)
        vals = [to_bool(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        n = len(vals)
        if n == 0:
            warnings.append(f"{path}: column '{col}' had no parseable True/False values -- skipping view")
            continue
        k = sum(vals)
        acc = k / n
        test = binomtest(k, n, p=0.5, alternative="two-sided")
        results.append({
            "path": path,
            "view": actual_view,
            "test_type": "binomial",
            "n": n,
            "observed_accuracy": acc,
            "test_statistic": test.statistic,
            "p_value": test.pvalue,
            "significant": test.pvalue < ALPHA,
        })
    return results


def run_fourclass_test(path: Path, rows, header, warnings):
    gt_col = find_ground_truth_column(header)
    if gt_col is None:
        warnings.append(f"{path}: no ground_truth column found -- skipping")
        return []

    gt_values = {r.get(gt_col, "").strip().lower() for r in rows if r.get(gt_col)}
    foreign = gt_values - CANONICAL_4CLASS
    if foreign:
        warnings.append(
            f"{path}: ground truth labels {sorted(gt_values)} don't match the 4-class "
            f"taxonomy (Novice/Early Expert/Intermediate Expert/Late Expert) -- skipping"
        )
        return []

    predicted_cols = find_predicted_columns(header)
    if not predicted_cols:
        warnings.append(f"{path}: no *_predicted / predicted column found -- skipping")
        return []

    correct_cols = find_correct_columns(header)

    results = []
    for view, pred_col in predicted_cols.items():
        actual_view = view if view != "single" else resolve_single_view(path.name)

        preds_norm = [r.get(pred_col, "").strip().lower() for r in rows]
        counts = {label: preds_norm.count(label) for label in CANONICAL_4CLASS}
        n = sum(counts.values())
        if n == 0:
            warnings.append(f"{path}: column '{pred_col}' had no values matching the 4-class taxonomy -- skipping view")
            continue

        f_obs = [counts[label] for label in sorted(CANONICAL_4CLASS)]
        f_exp = [n / 4] * 4
        stat, p = chisquare(f_obs=f_obs, f_exp=f_exp)

        acc = None
        corr_col = correct_cols.get(view) or correct_cols.get("single")
        if corr_col:
            bools = [to_bool(r.get(corr_col)) for r in rows]
            bools = [b for b in bools if b is not None]
            if bools:
                acc = sum(bools) / len(bools)

        results.append({
            "path": path,
            "view": actual_view,
            "test_type": "chi-squared",
            "n": n,
            "observed_accuracy": acc,
            "test_statistic": stat,
            "p_value": p,
            "significant": p < ALPHA,
        })
    return results


def main():
    warnings = []
    rows_out = []

    csv_paths = []
    for d in RESULT_DIRS:
        csv_paths.extend(sorted(d.rglob("*.csv")))

    for path in csv_paths:
        prompt_type = classify_filename(path.name)
        if prompt_type is None:
            warnings.append(f"{path}: filename doesn't match binary/fourclass/structured/reasoning naming -- skipping")
            continue

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

        model, activity, trim, frames, _ = parse_meta(path)

        if prompt_type == "binary":
            test_results = run_binary_test(path, rows, header, warnings)
        else:
            test_results = run_fourclass_test(path, rows, header, warnings)

        for res in test_results:
            rows_out.append({
                "model": model,
                "activity": activity,
                "trim_condition": trim,
                "frame_count": frames,
                "prompt_type": prompt_type,
                "view": res["view"],
                "file": str(res["path"].relative_to(REPO_ROOT)),
                "test_type": res["test_type"],
                "n": res["n"],
                "observed_accuracy": round(res["observed_accuracy"], 4) if res["observed_accuracy"] is not None else "",
                "test_statistic": round(res["test_statistic"], 4),
                "p_value": res["p_value"],
                "significant_at_0.05": "yes" if res["significant"] else "no",
            })

    rows_out.sort(key=lambda r: (r["model"], r["activity"], r["prompt_type"], r["trim_condition"], r["frame_count"], r["view"]))

    OUT_CSV.parent.mkdir(exist_ok=True)
    fieldnames = ["model", "activity", "trim_condition", "frame_count", "prompt_type", "view",
                  "file", "test_type", "n", "observed_accuracy", "test_statistic", "p_value", "significant_at_0.05"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} test results to {OUT_CSV}")
    print(f"\n{'='*70}\nWARNINGS ({len(warnings)} files skipped)\n{'='*70}")
    for w in warnings:
        print(f"  WARNING: {w}")

    above = [r for r in rows_out if r["significant_at_0.05"] == "yes" and r["observed_accuracy"] != "" and r["observed_accuracy"] > (0.5 if r["prompt_type"] == "binary" else 0.25)]
    not_sig = [r for r in rows_out if r["significant_at_0.05"] == "no"]

    print(f"\n{'='*70}\n1. SIGNIFICANTLY ABOVE RANDOM CHANCE (real signal), n={len(above)}\n{'='*70}")
    for r in above:
        chance = "50%" if r["prompt_type"] == "binary" else "25%"
        print(f"  {r['model']:11s} {r['activity']:9s} {r['prompt_type']:11s} {r['trim_condition']:8s} "
              f"n{r['frame_count'] or '?':<3s} {r['view']:6s} acc={r['observed_accuracy']:.2%} "
              f"(chance={chance}) p={r['p_value']:.2e}  [{r['file']}]")

    print(f"\n{'='*70}\n2. NOT SIGNIFICANTLY DIFFERENT FROM CHANCE (random-equivalent), n={len(not_sig)}\n{'='*70}")
    for r in not_sig:
        chance = "50%" if r["prompt_type"] == "binary" else "25%"
        acc_str = f"{r['observed_accuracy']:.2%}" if r["observed_accuracy"] != "" else "n/a"
        print(f"  {r['model']:11s} {r['activity']:9s} {r['prompt_type']:11s} {r['trim_condition']:8s} "
              f"n{r['frame_count'] or '?':<3s} {r['view']:6s} acc={acc_str} "
              f"(chance={chance}) p={r['p_value']:.2e}  [{r['file']}]")


if __name__ == "__main__":
    main()
