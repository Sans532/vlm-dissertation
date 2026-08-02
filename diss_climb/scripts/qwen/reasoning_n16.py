"""
Qwen2.5-VL-7B | reasoning prompt | 16 frames | exo + ego
Benchmark: benchmark_reasoning.json (25 per level, 100 clips)
"""
import json, os, csv, gc, warnings, re
import torch, cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER          = os.environ.get("USER")
MODEL_PATH    = f"/home/{USER}/dissertation/models/qwen25vl-7b"
DATA_DIR      = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK     = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_reasoning.json"
RESULTS       = f"/home/{USER}/dissertation/repo/dissertation_v2/results/qwen/reasoning_n16_qwen.csv"
NUM_FRAMES    = 16
LABELS        = ["Late Expert", "Intermediate Expert", "Early Expert", "Novice"]

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")

def get_frames(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: cap.release(); return []
    indices = [int(i * total / NUM_FRAMES) for i in range(NUM_FRAMES)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (420,360)), cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def ask(path, question):
    frames = get_frames(path)
    if not frames: raise Exception("No frames")
    content = [{"type":"image"} for _ in frames]
    content.append({"type":"text","text":f"These are {NUM_FRAMES} frames from a video of a person performing an activity. {question}"})
    messages = [{"role":"user","content":content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out, frames; torch.cuda.empty_cache(); gc.collect()
    return clean

def extract_label(answer):
    a = answer.lower()
    for label in LABELS:
        if label.lower() in a:
            return label
    return "Unknown"

benchmark = json.load(open(BENCHMARK))
print(f"Clips: {len(benchmark)} | frames: {NUM_FRAMES} | views: exo + ego\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow([
        "clip_id","take_folder","ground_truth",
        "exo_full_answer","exo_predicted","exo_correct",
        "ego_full_answer","ego_predicted","ego_correct"
    ])

stats = {"exo":[0,0], "ego":[0,0]}
exo_preds, ego_preds = Counter(), Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])
    question = item["question_reasoning"]
    row = [item["clip_id"], item["take_folder"], gt]
    print(f"[{i+1}/{len(benchmark)}] {item['take_folder']} (GT={gt})")

    for view, path in [("exo", exo_path), ("ego", ego_path)]:
        try:
            ans = ask(path, question)
            pred = extract_label(ans)
            ok = pred.lower() == gt.lower()
        except Exception as e:
            print(f"  {view} ERROR: {e}"); ans="ERROR"; pred="Unknown"; ok=False
        row.extend([ans, pred, ok])
        stats[view][1] += 1
        if ok: stats[view][0] += 1
        if view == "exo": exo_preds[pred] += 1
        else: ego_preds[pred] += 1
        print(f"  {view}: {pred} {'OK' if ok else 'X'} | {ans[:80].replace(chr(10),' ')}")

    with open(RESULTS, "a", newline="") as f: csv.writer(f).writerow(row)
    print()

print("="*60)
print("RESULTS — Qwen reasoning prompt 16 frames")
print("="*60)
for v, (c,t) in stats.items():
    print(f"  {v}: {c}/{t} = {c/t:.1%}" if t else "")
print(f"\nRandom chance: 25%")
print(f"\nExo predictions: {dict(exo_preds.most_common())}")
print(f"Ego predictions: {dict(ego_preds.most_common())}")
print("="*60)
