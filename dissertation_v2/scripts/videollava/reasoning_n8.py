"""
Video-LLaVA | reasoning prompt | 8 frames | exo + ego
Benchmark: benchmark_reasoning.json (25 per level, 100 clips)
"""
import json, os, csv, gc, warnings
import torch, numpy as np, av
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER          = os.environ.get("USER")
MODEL_PATH    = f"/home/{USER}/dissertation/models/videollava"
DATA_DIR      = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK     = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_reasoning.json"
RESULTS       = f"/home/{USER}/dissertation/repo/dissertation_v2/results/videollava/reasoning_n8_vl.csv"
NUM_FRAMES    = 8
LABELS        = ["Late Expert", "Intermediate Expert", "Early Expert", "Novice"]

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

print("Loading Video-LLaVA ...")
model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = VideoLlavaProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")

def read_video(path):
    try:
        container = av.open(path)
        total = container.streams.video[0].frames
        if total == 0:
            frames_list = list(container.decode(video=0))
            total = len(frames_list)
            container.close(); container = av.open(path)
        if total == 0: container.close(); return None
        indices = set(np.linspace(0, total-1, NUM_FRAMES, dtype=int))
        frames, idx = [], 0
        for frame in container.decode(video=0):
            if idx in indices: frames.append(frame.to_ndarray(format="rgb24"))
            idx += 1
            if len(frames) >= NUM_FRAMES: break
        container.close()
        while len(frames) < NUM_FRAMES: frames.append(frames[-1])
        return np.stack(frames)
    except Exception:
        return None

def ask(path, question):
    video = read_video(path)
    if video is None: raise Exception("No frames")
    prompt = f"USER: <video>\nThese are {NUM_FRAMES} frames from a video of a person performing an activity. {question} ASSISTANT:"
    inputs = processor(text=prompt, videos=video, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw else raw.strip()
    del inputs, out, video; torch.cuda.empty_cache(); gc.collect()
    return clean

def extract_label(answer):
    a = answer.lower()
    for label in LABELS:
        if label.lower() in a: return label
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
print("RESULTS — VideoLLaVA reasoning prompt 8 frames")
print("="*60)
for v, (c,t) in stats.items():
    print(f"  {v}: {c}/{t} = {c/t:.1%}" if t else "")
print(f"\nRandom chance: 25%")
print(f"\nExo predictions: {dict(exo_preds.most_common())}")
print(f"Ego predictions: {dict(ego_preds.most_common())}")
print("="*60)
