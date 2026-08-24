"""
Gemini 3.1 Flash-Lite | Climbing | Text-only 4-class (Novice included)
Given ONLY expert commentary (no video), can Gemini read the full commentary
and correctly state the skill level: Novice / Early Expert / Intermediate Expert / Late Expert?

This directly tests whether skill level is recoverable from LANGUAGE ALONE,
mirroring commentary/qwen.py but with Gemini instead of Qwen2.5-VL-7B.
Compare against video-based 4-class results (diss_climb/scripts/gemini/fourclass.py):
  - If text succeeds where video fails -> failure is in visual grounding, not reasoning
  - If text also fails -> skill may not be reliably encoded per-clip in any modality
"""
import json, os, time, csv, sys
import google.genai as genai
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO           = os.path.dirname(SCRIPT_DIR)
BENCHMARK_PATH = os.path.join(REPO, "diss_climb", "benchmark", "benchmark_commentary.json")
RESULTS_PATH   = os.path.join(SCRIPT_DIR, "gemini_commentary_4class.csv")
LOG_PATH       = os.path.join(SCRIPT_DIR, "gemini_commentary_4class_log.txt")
MODEL          = "gemini-3.1-flash-lite"
DAILY_LIMIT    = 500
LABELS         = ["Late Expert", "Intermediate Expert", "Early Expert", "Novice"]

QUESTION = (
    "Read this entire commentary carefully. Based ONLY on what the commentary says, "
    "what is this climber's skill level? "
    "Answer only one: Novice / Early Expert / Intermediate Expert / Late Expert"
)

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


log_file = open(LOG_PATH, "a")
sys.stdout = Tee(sys.stdout, log_file)
print(f"\n{'#' * 60}\nRun started: {datetime.now().isoformat()}\n{'#' * 60}\n")

client = genai.Client()


class QuotaExceeded(Exception):
    pass


def ask_gemini_text(commentary, question, max_retries=2, max_rate_limit_waits=5):
    prompt = (
        "The following is expert coaching commentary about a bouldering climber's attempt:\n\n"
        + commentary.strip() + "\n\n" + question
    )
    rate_limit_waits = 0
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.models.generate_content(model=MODEL, contents=[prompt])
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                if rate_limit_waits < max_rate_limit_waits:
                    rate_limit_waits += 1
                    print(f"  Rate limit hit, waiting 65s ({rate_limit_waits}/{max_rate_limit_waits})...")
                    time.sleep(65)
                    continue
                raise QuotaExceeded(err_str)
            elif attempt < max_retries - 1:
                print("  Retrying after error: " + err_str[:100])
                time.sleep(10)
                attempt += 1
                continue
            else:
                raise


def extract_label(answer):
    a = answer.lower()
    if "late expert" in a: return "Late Expert"
    if "intermediate" in a: return "Intermediate Expert"
    if "early expert" in a: return "Early Expert"
    if "novice" in a: return "Novice"
    return "Unknown"


benchmark = json.load(open(BENCHMARK_PATH))
completed_ids = set()
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            if row["answer"] != "ERROR":
                completed_ids.add(row["clip_id"])
    print(f"Found {len(completed_ids)} already-completed clips. Resuming.\n")
else:
    with open(RESULTS_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["clip_id", "take_name", "ground_truth", "answer", "predicted", "correct"])

remaining = [item for item in benchmark if item["clip_id"] not in completed_ids]
print(f"Total: {len(benchmark)} | Done: {len(completed_ids)} | Remaining: {len(remaining)}")
print(f"Model: {MODEL}\nPrompt: {QUESTION}\n")

if not remaining:
    print("All clips already completed!")
    sys.exit(0)

todays_batch = remaining[:DAILY_LIMIT]
processed = 0

for i, item in enumerate(todays_batch):
    gt = item["ground_truth"]
    commentary = item["commentary"]
    take_name = item["take_name"]

    print(f"[{i+1}/{len(todays_batch)}] {take_name} (GT={gt})")
    try:
        ans = ask_gemini_text(commentary, QUESTION)
        pred = extract_label(ans)
        ok = pred.lower() == gt.lower()
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.writer(f).writerow([item["clip_id"], take_name, gt, ans, pred, ok])
        print(f"  {pred} {'OK' if ok else 'X'} | raw: {ans[:60]}")
        processed += 1
    except QuotaExceeded:
        print("\nDAILY QUOTA REACHED. Progress saved. Run again tomorrow.")
        sys.exit(0)
    except Exception as e:
        print(f"  ERROR: {e}")
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.writer(f).writerow([item["clip_id"], take_name, gt, "ERROR", "Unknown", False])
    time.sleep(2)

print(f"\nBatch complete. Processed {processed} clips.")
remaining_after = len(remaining) - processed
if remaining_after > 0:
    print(f"{remaining_after} clips remaining.")
else:
    rows = list(csv.DictReader(open(RESULTS_PATH)))
    correct = sum(1 for r in rows if r["correct"] == "True")
    preds = Counter(r["predicted"] for r in rows)
    print(f"\nRESULTS — Gemini text-only 4-class climbing (commentary)")
    print(f"Overall: {correct}/{len(rows)} = {correct/len(rows):.1%}")
    for label in LABELS:
        lr = [r for r in rows if r["ground_truth"] == label]
        if lr:
            lc = sum(1 for r in lr if r["correct"] == "True")
            print(f"{label:22s}: {lc}/{len(lr)} = {lc/len(lr):.1%}")
    print("Random chance: 25%")
    print(f"Answers: {dict(preds.most_common())}")
    print("\nCompare to video-based 4-class (diss_climb/scripts/gemini/fourclass.py):")
    print("If text-only score is meaningfully higher -> failure is in visual grounding")
    print("If text-only also collapses -> skill may not be cleanly per-clip recoverable")
