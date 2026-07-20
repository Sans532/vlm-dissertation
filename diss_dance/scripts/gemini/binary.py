"""
Gemini 3.1 Flash-Lite | Dance | Binary | Entire video
Videos stored OUTSIDE repo to avoid git bloat.
"""
import json, os, time, csv, sys
import google.genai as genai
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

USER           = os.environ.get("USER")
REPO           = "/home/" + USER + "/dissertation/repo/diss_dance"
DATA_DIR       = "/home/" + USER + "/data_dance/videos"
BENCHMARK_PATH = REPO + "/benchmark/benchmark_binary_dance.json"
RESULTS_PATH   = "/home/" + USER + "/results/gemini/gemini_dance_entire_binary.csv"
LOG_PATH       = "/home/" + USER + "/results/gemini/binary_dance_log.txt"
MODEL          = "gemini-3.1-flash-lite"
DAILY_LIMIT    = 500

QUESTION = "Is this person a Novice or an Expert at this activity? Answer only: Novice or Expert"

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


def ask_gemini(video_path, question, max_retries=2):
    for attempt in range(max_retries):
        try:
            video_file = client.files.upload(file=video_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(4)
                video_file = client.files.get(name=video_file.name)
            if video_file.state.name == "FAILED":
                raise Exception("Video processing failed")
            response = client.models.generate_content(model=MODEL, contents=[video_file, question])
            client.files.delete(name=video_file.name)
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                raise QuotaExceeded(err_str)
            elif attempt < max_retries - 1:
                print("  Retrying after error: " + err_str[:100])
                time.sleep(10)
                continue
            else:
                raise


class QuotaExceeded(Exception):
    pass


def check(answer, gt):
    a = answer.lower()
    has_nov = "novice" in a
    has_exp = "expert" in a
    if has_nov and not has_exp:
        return gt.lower() == "novice"
    if has_exp and not has_nov:
        return gt.lower() in ["expert", "late expert"]
    pos_n = a.find("novice") if has_nov else 10**9
    pos_e = a.find("expert") if has_exp else 10**9
    if pos_n == pos_e:
        return False
    return (gt.lower() == "novice") == (pos_n < pos_e)


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
        csv.writer(f).writerow(["clip_id", "take_folder", "ground_truth", "answer", "predicted", "correct"])

remaining = [item for item in benchmark if item["clip_id"] not in completed_ids]
print(f"Total: {len(benchmark)} | Done: {len(completed_ids)} | Remaining: {len(remaining)}")
print(f"Model: {MODEL}\nPrompt: {QUESTION}\n")

if not remaining:
    print("All clips already completed!")
    exit(0)

todays_batch = remaining[:DAILY_LIMIT]
processed = 0

for i, item in enumerate(todays_batch):
    gt = item["ground_truth"]
    take_folder = item["take_folder"]
    video_path = os.path.join(DATA_DIR, item["video_path_exo"])

    if not os.path.exists(video_path):
        print(f"[{i+1}/{len(todays_batch)}] {take_folder} — SKIP (not found)")
        continue

    print(f"[{i+1}/{len(todays_batch)}] {take_folder} (GT={gt})")
    try:
        ans = ask_gemini(video_path, QUESTION)
        ok = check(ans, gt)
        if "novice" in ans.lower() and "expert" not in ans.lower():
            pred = "Novice"
        elif "expert" in ans.lower():
            pred = "Expert"
        else:
            pred = ans.strip()
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.writer(f).writerow([item["clip_id"], take_folder, gt, ans, pred, ok])
        print(f"  {pred} {'OK' if ok else 'X'} | raw: {ans[:60]}")
        processed += 1
    except QuotaExceeded:
        print("\nDAILY QUOTA REACHED. Progress saved. Run again tomorrow.")
        exit(0)
    except Exception as e:
        print(f"  ERROR: {e}")
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.writer(f).writerow([item["clip_id"], take_folder, gt, "ERROR", "Unknown", False])
    time.sleep(4)

print(f"\nBatch complete. Processed {processed} clips.")
remaining_after = len(remaining) - processed
if remaining_after > 0:
    print(f"{remaining_after} clips remaining.")
else:
    rows = list(csv.DictReader(open(RESULTS_PATH)))
    correct = sum(1 for r in rows if r["correct"] == "True")
    preds = Counter(r["predicted"] for r in rows)
    print(f"\nRESULTS — Gemini binary dance")
    print(f"Overall: {correct}/{len(rows)} = {correct/len(rows):.1%}")
    print(f"Random: 50%")
    print(f"Answers: {dict(preds.most_common())}")
