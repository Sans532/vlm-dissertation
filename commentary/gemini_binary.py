"""
Gemini 3.1 Flash-Lite | Climbing | Text-only binary (Novice vs Expert)
Given ONLY expert commentary (no video), can Gemini read the full commentary
and correctly state whether the climber is a Novice or an Expert?

This directly tests whether skill level is recoverable from LANGUAGE ALONE,
mirroring commentary/qwen_binary.py but with Gemini instead of Qwen2.5-VL-7B.
Compare against video-based binary results (diss_climb/scripts/gemini/binary.py):
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
RESULTS_PATH   = os.path.join(SCRIPT_DIR, "gemini_commentary_binary.csv")
LOG_PATH       = os.path.join(SCRIPT_DIR, "gemini_commentary_binary_log.txt")
MODEL          = "gemini-3.1-flash-lite"
DAILY_LIMIT    = 500

QUESTION = (
    "Read this entire commentary carefully. Based ONLY on what the commentary says, "
    "is this climber a Novice or an Expert? "
    "Answer only one: Novice or Expert"
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


def to_binary_gt(gt):
    return "Novice" if gt.lower() == "novice" else "Expert"


def check(answer, gt_binary):
    a = answer.lower()
    has_nov = "novice" in a
    has_exp = "expert" in a
    if has_nov and not has_exp:
        return gt_binary == "Novice"
    if has_exp and not has_nov:
        return gt_binary == "Expert"
    pos_n = a.find("novice") if has_nov else 10**9
    pos_e = a.find("expert") if has_exp else 10**9
    if pos_n == pos_e:
        return False
    return (gt_binary == "Novice") == (pos_n < pos_e)


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
        csv.writer(f).writerow(["clip_id", "take_name", "ground_truth", "ground_truth_binary",
                                 "answer", "predicted", "correct"])

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
    gt_binary = to_binary_gt(gt)
    commentary = item["commentary"]
    take_name = item["take_name"]

    print(f"[{i+1}/{len(todays_batch)}] {take_name} (GT={gt} -> {gt_binary})")
    try:
        ans = ask_gemini_text(commentary, QUESTION)
        ok = check(ans, gt_binary)
        a = ans.lower()
        if "novice" in a and "expert" not in a:
            pred = "Novice"
        elif "expert" in a:
            pred = "Expert"
        else:
            pred = ans.strip()
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.writer(f).writerow([item["clip_id"], take_name, gt, gt_binary, ans, pred, ok])
        print(f"  {pred} {'OK' if ok else 'X'} | raw: {ans[:60]}")
        processed += 1
    except QuotaExceeded:
        print("\nDAILY QUOTA REACHED. Progress saved. Run again tomorrow.")
        sys.exit(0)
    except Exception as e:
        print(f"  ERROR: {e}")
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.writer(f).writerow([item["clip_id"], take_name, gt, gt_binary, "ERROR", "Unknown", False])
    time.sleep(2)

print(f"\nBatch complete. Processed {processed} clips.")
remaining_after = len(remaining) - processed
if remaining_after > 0:
    print(f"{remaining_after} clips remaining.")
else:
    rows = list(csv.DictReader(open(RESULTS_PATH)))
    correct = sum(1 for r in rows if r["correct"] == "True")
    preds = Counter(r["predicted"] for r in rows)
    print(f"\nRESULTS — Gemini text-only binary climbing (commentary)")
    print(f"Overall: {correct}/{len(rows)} = {correct/len(rows):.1%}")
    print(f"Random chance: 50%")
    print(f"Answers: {dict(preds.most_common())}")
    print("\nCompare to video-based binary (diss_climb/scripts/gemini/binary.py):")
    print("If text-only score is meaningfully higher -> failure is in visual grounding")
    print("If text-only also collapses -> skill may not be cleanly per-clip recoverable")
