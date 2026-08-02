"""
Side-by-side comparison of Qwen vs Video-LLaVA on matching experiments.
Reads accuracy_summary.csv produced by compute_accuracy.py
Run compute_accuracy.py FIRST.
"""
import csv, os
from collections import defaultdict

USER = os.environ.get("USER")
SUMMARY_PATH = f"/home/{USER}/dissertation/repo/dissertation_v2/results/accuracy_summary.csv"

def main():
    if not os.path.exists(SUMMARY_PATH):
        print("Run compute_accuracy.py first to generate accuracy_summary.csv")
        return

    rows = list(csv.DictReader(open(SUMMARY_PATH)))

    qwen_rows = [r for r in rows if "qwen" in r["file"].lower() or "_qwen" in r["file"].lower()]
    vl_rows = [r for r in rows if "vl" in r["file"].lower() or "videollava" in r["file"].lower()]

    print("=" * 70)
    print("QWEN RESULTS")
    print("=" * 70)
    for r in qwen_rows:
        print(f"{r['file']:<40} {r['metric']:<15} {float(r['accuracy']):.1%}")

    print("\n" + "=" * 70)
    print("VIDEO-LLAVA RESULTS")
    print("=" * 70)
    for r in vl_rows:
        print(f"{r['file']:<40} {r['metric']:<15} {float(r['accuracy']):.1%}")

    # Print quick averages
    if qwen_rows:
        avg_qwen = sum(float(r["accuracy"]) for r in qwen_rows) / len(qwen_rows)
        print(f"\nQwen average accuracy across all experiments: {avg_qwen:.1%}")
    if vl_rows:
        avg_vl = sum(float(r["accuracy"]) for r in vl_rows) / len(vl_rows)
        print(f"Video-LLaVA average accuracy across all experiments: {avg_vl:.1%}")


if __name__ == "__main__":
    main()
