"""
Text-only binary classification (Novice vs Expert).
Given ONLY expert commentary (no video), can Qwen read the full commentary
and correctly state whether the climber is a Novice or an Expert?

This directly tests whether skill level is recoverable from LANGUAGE ALONE.
Compare against video-based binary results (structured_n8/binary):
  - If text succeeds where video fails -> failure is in visual grounding, not reasoning
  - If text also fails -> skill may not be reliably encoded per-clip in any modality
"""
import json, os, csv, gc, warnings
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER       = os.environ.get("USER")
MODEL_PATH = "/home/" + USER + "/dissertation/models/qwen25vl-7b"
BENCHMARK  = "/home/" + USER + "/dissertation/repo/dissertation_v2/benchmark/benchmark_commentary.json"
RESULTS    = "/home/" + USER + "/dissertation/repo/commentary/commentary_binary.csv"

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

QUESTION = (
    "Read this entire commentary carefully. Based ONLY on what the commentary says, "
    "is this climber a Novice or an Expert? "
    "Answer only one: Novice or Expert"
)

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def ask_text_only(commentary, question):
    prompt = (
        "The following is expert coaching commentary about a bouldering climber's attempt:\n\n"
        + commentary.strip() + "\n\n" + question
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out
    torch.cuda.empty_cache(); gc.collect()
    return clean


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


benchmark = json.load(open(BENCHMARK))
print("Clips (all levels): " + str(len(benchmark)))
print("Prompt: " + QUESTION + "\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow(["clip_id", "take_name", "ground_truth", "ground_truth_binary",
                             "answer", "predicted", "correct"])

correct = 0
nov_correct = 0; nov_total = 0
exp_correct = 0; exp_total = 0
preds = Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    gt_binary = to_binary_gt(gt)
    commentary = item["commentary"]

    print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + item["take_name"] + " (GT=" + gt + " -> " + gt_binary + ")")

    try:
        ans = ask_text_only(commentary, QUESTION)
        ok = check(ans, gt_binary)
        a = ans.lower()
        if "novice" in a and "expert" not in a:
            pred = "Novice"
        elif "expert" in a:
            pred = "Expert"
        else:
            pred = "Unknown"
    except Exception as e:
        print("  ERROR: " + str(e))
        ans = "ERROR"; pred = "Unknown"; ok = False

    preds[pred] += 1
    correct += int(ok)
    if gt_binary == "Novice":
        nov_total += 1
        nov_correct += int(ok)
    else:
        exp_total += 1
        exp_correct += int(ok)

    print("  Predicted: " + pred + " " + ("OK" if ok else "X"))
    print("  Answer: " + ans[:80])
    print()

    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow([item["clip_id"], item["take_name"], gt, gt_binary, ans, pred, ok])

n = len(benchmark)
print("=" * 60)
print("RESULTS — Qwen text-only binary (Novice vs Expert)")
print("=" * 60)
print("Overall: " + str(correct) + "/" + str(n) + " = " + str(round(correct/n*100, 1)) + "%")
print("Random chance: 50%")
if nov_total:
    print("Novice:  " + str(nov_correct) + "/" + str(nov_total) + " = " + str(round(nov_correct/nov_total*100, 1)) + "%")
if exp_total:
    print("Expert:  " + str(exp_correct) + "/" + str(exp_total) + " = " + str(round(exp_correct/exp_total*100, 1)) + "%")
print("\nPrediction distribution: " + str(dict(preds.most_common())))
print("=" * 60)
print("\nCompare to video-based binary (structured/binary): typically ~50% (random)")
print("If text-only score is meaningfully higher -> failure is in visual grounding")
print("If text-only also collapses -> skill may not be cleanly per-clip recoverable")
