"""
Qwen2.5-VL-7B | binary | 8 frames | EXOCENTRIC ONLY
Benchmark: benchmark_binary.json (25 Novice + 25 Late Expert)
"""
import json, os, csv, gc, warnings
import torch, cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER          = os.environ.get("USER")
MODEL_PATH    = f"/home/{USER}/dissertation/models/qwen25vl-7b"
DATA_DIR      = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK     = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_binary.json"
RESULTS       = f"/home/{USER}/dissertation/repo/dissertation_v2/results/qwen/binary_n8_qwen_exo.csv"
NUM_FRAMES    = 8
VIEW          = "exo"

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")

QUESTION = "Is this person a Novice or an Expert at this activity? Answer only: Novice or Expert"

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

def ask(path):
    frames = get_frames(path)
    if not frames: raise Exception("No frames")
    content = [{"type":"image"} for _ in frames]
    content.append({"type":"text","text":f"These are {NUM_FRAMES} frames from a video of a person performing an activity. {QUESTION}"})
    messages = [{"role":"user","content":content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out, frames; torch.cuda.empty_cache(); gc.collect()
    return clean

def check(answer, gt):
    a = answer.lower()
    has_nov = "novice" in a
    has_exp = "expert" in a
    if has_nov and not has_exp: return gt.lower() == "novice"
    if has_exp and not has_nov: return gt.lower() in ["expert", "late expert"]
    pos_n = a.find("novice") if has_nov else 10**9
    pos_e = a.find("expert") if has_exp else 10**9
    if pos_n == pos_e: return False
    return (gt.lower() == "novice") == (pos_n < pos_e)

benchmark = json.load(open(BENCHMARK))
print(f"Clips: {len(benchmark)} | frames: {NUM_FRAMES} | view: {VIEW}\n")
print(f"Prompt: {QUESTION}\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow(["clip_id","take_folder","ground_truth","answer","predicted","correct"])

correct_total = 0
nov_correct = 0; nov_total = 0
exp_correct = 0; exp_total = 0
preds = Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    path = os.path.join(DATA_DIR, item["video_path_exo"])
    print(f"[{i+1}/{len(benchmark)}] {item['take_folder']} (GT={gt})")

    try:
        ans = ask(path)
        ok = check(ans, gt)
        if "novice" in ans.lower() and "expert" not in ans.lower(): pred = "Novice"
        elif "expert" in ans.lower(): pred = "Expert"
        else: pred = ans.strip()
    except Exception as e:
        print(f"  ERROR: {e}"); ans="ERROR"; ok=False; pred="Unknown"

    preds[pred] += 1
    correct_total += int(ok)
    if gt == "Novice": nov_total+=1; nov_correct+=int(ok)
    else: exp_total+=1; exp_correct+=int(ok)

    print(f"  {VIEW}: {pred} {'OK' if ok else 'X'}")
    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow([item["clip_id"], item["take_folder"], gt, ans, pred, ok])

print("\n" + "="*60)
print(f"RESULTS — Qwen binary {NUM_FRAMES}f {VIEW.upper()} only")
print("="*60)
n = len(benchmark)
print(f"Overall:  {correct_total}/{n} = {correct_total/n:.1%}")
print(f"Novice:   {nov_correct}/{nov_total} = {nov_correct/nov_total:.1%}" if nov_total else "")
print(f"Expert:   {exp_correct}/{exp_total} = {exp_correct/exp_total:.1%}" if exp_total else "")
print(f"Random:   50%")
print(f"Answers:  {dict(preds.most_common())}")
print("="*60)
