"""
Qwen2.5-VL-7B | Text-only | Expert commentary → skill classification
No video input. Tests whether language alone encodes skill level.
Directly probes the language-gap hypothesis.
"""
import json, os, csv, gc, warnings
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER       = os.environ.get("USER")
MODEL_PATH = "/home/" + USER + "/dissertation/models/qwen25vl-7b"
BENCHMARK  = "/home/" + USER + "/dissertation/repo/dissertation_v2/benchmark/benchmark_commentary.json"
RESULTS    = "/home/" + USER + "/dissertation/repo/dissertation_v2/results/qwen/commentary_text_only.csv"
LABELS     = ["Late Expert", "Intermediate Expert", "Early Expert", "Novice"]

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def ask_text_only(commentary, question):
    """No image — text only."""
    prompt = (
        "The following is expert coaching commentary about a bouldering climber:\n\n"
        + commentary.strip()
        + "\n\n"
        + question
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


def extract_label(answer):
    a = answer.lower()
    for label in LABELS:
        if label.lower() in a:
            return label
    return "Unknown"


QUESTION = "Based only on this commentary, what is the climber's skill level? Answer only: Novice / Early Expert / Intermediate Expert / Late Expert"

benchmark = json.load(open(BENCHMARK))
print("Clips: " + str(len(benchmark)))
print("Input: TEXT ONLY (no video)")
print("Prompt: " + QUESTION + "\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow(["clip_id","take_name","ground_truth","answer","predicted","correct"])

correct = 0
preds = Counter()
by_class = Counter()
correct_by_class = Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    commentary = item["commentary"]

    print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + item["take_name"] + " (GT=" + gt + ")")
    print("  Commentary: " + commentary[:100].replace("\n"," ") + "...")

    try:
        ans = ask_text_only(commentary, QUESTION)
        pred = extract_label(ans)
        ok = pred.lower() == gt.lower()
    except Exception as e:
        print("  ERROR: " + str(e))
        ans="ERROR"; pred="Unknown"; ok=False

    preds[pred] += 1
    by_class[gt] += 1
    if ok:
        correct += 1
        correct_by_class[gt] += 1

    print("  Predicted: " + pred + " " + ("OK" if ok else "X"))
    print("  Answer: " + ans[:80])
    print()

    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow([item["clip_id"], item["take_name"], gt, ans, pred, ok])

n = len(benchmark)
print("=" * 60)
print("RESULTS — Qwen text-only commentary classification")
print("=" * 60)
print("Overall: " + str(correct) + "/" + str(n) + " = " + str(round(correct/n*100,1)) + "%")
print("Random chance: 25%")
print("\nPer class:")
for level in LABELS:
    c = correct_by_class[level]
    t = by_class[level]
    print("  " + level + ": " + str(c) + "/" + str(t) + " = " + (str(round(c/t*100,1))+"%" if t else "N/A"))
print("\nPrediction distribution: " + str(dict(preds.most_common())))
print("=" * 60)
print("\nKey question: does text-only score > video-only (25%)?")
print("If yes -> language encodes skill; visual processing is the bottleneck")
print("If no  -> neither modality alone is sufficient")
