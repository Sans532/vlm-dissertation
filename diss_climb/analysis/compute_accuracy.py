"""
Reads all result CSVs in results/qwen/ and results/videollava/
Prints a master summary table of accuracy for every experiment.
Run this AFTER all your evaluation jobs have finished.
"""
import csv, os, glob

USER = os.environ.get("USER")
RESULTS_DIR = f"/home/{USER}/dissertation/repo/dissertation_v2/results"

def analyze_csv(path):
    """Auto-detect correctness columns and compute accuracy for each."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None

    total = len(rows)
    cols = rows[0].keys()
    correct_cols = [c for c in cols if "correct" in c.lower()]

    result = {"file": os.path.basename(path), "total": total, "metrics": {}}
    for col in correct_cols:
        n_correct = sum(1 for r in rows if r.get(col, "").strip() == "True")
        n_valid = sum(1 for r in rows if r.get(col, "").strip() in ["True", "False"])
        if n_valid:
            result["metrics"][col] = (n_correct, n_valid, n_correct / n_valid)
    return result


def main():
    csv_files = []
    for model_dir in ["qwen", "videollava"]:
        path = os.path.join(RESULTS_DIR, model_dir, "*.csv")
        csv_files.extend(sorted(glob.glob(path)))

    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}/qwen/ or {RESULTS_DIR}/videollava/")
        return

    print("=" * 90)
    print(f"{'FILE':<45} {'METRIC':<20} {'CORRECT/TOTAL':<15} {'ACCURACY':<10}")
    print("=" * 90)

    summary_rows = []

    for path in csv_files:
        result = analyze_csv(path)
        if not result:
            continue
        for metric, (correct, total, acc) in result["metrics"].items():
            print(f"{result['file']:<45} {metric:<20} {correct}/{total:<13} {acc:.1%}")
            summary_rows.append({
                "file": result["file"],
                "metric": metric,
                "correct": correct,
                "total": total,
                "accuracy": acc
            })

    print("=" * 90)

    # Save summary as its own CSV for easy reference/plotting
    out_path = os.path.join(RESULTS_DIR, "accuracy_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "metric", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary saved to: {out_path}")


if __name__ == "__main__":
    main()
