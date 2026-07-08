"""
Build benchmark_3class.json and run Qwen binary evaluation on it.
Labels: Novice / Intermediate / Expert (instead of 4 confusing levels)
Mapping: Novice→Novice, Intermediate Expert→Intermediate, Late Expert→Expert
10 clips per class = 30 total, trimmed to task window.
"""
import json, os, csv, gc, warnings, random
import torch, cv2
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from collections import Counter

warnings.filterwarnings("ignore")
random.seed(42)

USER       = os.environ.get("USER")
MODEL_PATH = "/home/" + USER + "/dissertation/models/qwen3vl-2b"
DATA_DIR   = "/home/" + USER + "/dissertation/data/egoexo"
TAKES_PATH = "/home/" + USER + "/dissertation/data/egoexo/takes.json"
BENCHMARK_OUT = "/home/" + USER + "/dissertation/repo/dissertation_v2/benchmark/benchmark_3class.json"
RESULTS    = "/home/" + USER + "/dissertation/repo/dissertation_v2/results/qwen/3class_structured.csv"
NUM_FRAMES = 8

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

# Label mapping: original → simplified
LABEL_MAP = {
    "Novice":               "Novice",
    "Intermediate Expert":  "Intermediate",
    "Late Expert":          "Expert",
}

QUESTION = (
    "Based only on the expert commentary above:\n"
    "Step 1: Summarise the key technique observations mentioned.\n"
    "Step 2: Identify any errors or strengths noted by the expert.\n"
    "Step 3: Classify the climber's skill level as exactly one of: "
    "Novice / Early Expert / Intermediate Expert / Late Expert.\n"
    "Format:\n"
    "Observations: ...\n"
    "Errors/Strengths: ...\n"
    "Skill Level: ..."
)

# ── Build benchmark ────────────────────────────────────────────
print("Building benchmark_3class.json ...")

takes_list = json.load(open(TAKES_PATH))
take_info = {}
for t in takes_list:
    take_info[t.get("take_name", "")] = t

prof_train = json.load(open("/home/" + USER + "/dissertation/data/egoexo/annotations/proficiency_demonstrator_train.json"))["annotations"]
prof_val   = json.load(open("/home/" + USER + "/dissertation/data/egoexo/annotations/proficiency_demonstrator_val.json"))["annotations"]
all_clips  = prof_train + prof_val

boulder = [a for a in all_clips if "boulder" in json.dumps(a).lower() or "climb" in a.get("scenario_name","").lower()]

groups = {}
for clip in boulder:
    level = clip["proficiency_score"]
    if level not in LABEL_MAP:
        continue   # skip Early Expert
    take_folder = clip["video_paths"]["ego"].split("/")[1]
    exo_path = os.path.join(DATA_DIR, clip["video_paths"].get("exo1", ""))
    ego_path = os.path.join(DATA_DIR, clip["video_paths"].get("ego", ""))
    if os.path.exists(exo_path) and os.path.exists(ego_path):
        groups.setdefault(level, []).append((clip, take_folder))

print("Available:")
for level, clips in groups.items():
    print("  " + level + " -> " + LABEL_MAP[level] + ": " + str(len(clips)))

benchmark = []
for orig_level, clips in groups.items():
    random.shuffle(clips)
    for clip, take_folder in clips[:10]:
        benchmark.append({
            "clip_id":        clip["take_uid"],
            "take_folder":    take_folder,
            "video_path_ego": clip["video_paths"]["ego"],
            "video_path_exo": clip["video_paths"].get("exo1", ""),
            "ground_truth_original": orig_level,
            "ground_truth": LABEL_MAP[orig_level],
        })

random.shuffle(benchmark)
with open(BENCHMARK_OUT, "w") as f:
    json.dump(benchmark, f, indent=2)

print("\nBenchmark saved: " + BENCHMARK_OUT)
print("Total: " + str(len(benchmark)) + " clips")
print(Counter(d["ground_truth"] for d in benchmark))

# ── Load model ─────────────────────────────────────────────────
print("\nLoading Qwen3-VL-2B ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def get_frames_trimmed(video_path, take_folder):
    info = take_info.get(take_folder, {})
    cap = cv2.VideoCapture(video_path)
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0 or fps_v == 0:
        cap.release(); return []
    start = max(0, int(info.get("task_start_sec", 0) * fps_v))
    end   = min(total, int(info.get("task_end_sec", total/fps_v) * fps_v))
    indices = [int(start + i * (end - start) / NUM_FRAMES) for i in range(NUM_FRAMES)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(f,(420,360)), cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def ask(video_path, take_folder, question):
    frames = get_frames_trimmed(video_path, take_folder)
    if not frames: raise Exception("No frames")
    content = [{"type":"image"} for _ in frames]
    content.append({"type":"text","text":"These are 8 frames from a video of a person bouldering. " + question})
    messages = [{"role":"user","content":content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out, frames; torch.cuda.empty_cache(); gc.collect()
    return clean


def extract_label(answer):
    import re
    a = answer.lower()
    match = re.search(r'skill level[:\s]+(.+?)(?:\n|$)', a)
    search_text = match.group(1).strip() if match else a
    if "intermediate" in search_text: return "Intermediate"
    if "expert" in search_text:       return "Expert"
    if "novice" in search_text:       return "Novice"
    # fallback to full answer
    if "intermediate" in a: return "Intermediate"
    if "expert" in a:       return "Expert"
    if "novice" in a:       return "Novice"
    return "Unknown"

def check(answer, gt):
    return extract_label(answer).lower() == gt.lower()


# ── Evaluate ───────────────────────────────────────────────────
print("Evaluating on benchmark_3class.json ...")
print("Prompt: " + QUESTION + "\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow(["clip_id","take_folder","ground_truth_original",
                             "ground_truth","exo_answer","exo_predicted","exo_correct",
                             "ego_answer","ego_predicted","ego_correct"])

stats = {"exo":[0,0], "ego":[0,0]}
exo_preds = Counter()
ego_preds = Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])
    row = [item["clip_id"], item["take_folder"], item["ground_truth_original"], gt]

    print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + item["take_folder"] +
          " (orig=" + item["ground_truth_original"] + " -> " + gt + ")")

    for view, path in [("exo", exo_path), ("ego", ego_path)]:
        try:
            ans = ask(path, item["take_folder"], QUESTION)
            pred = extract_label(ans)
            ok = pred.lower() == gt.lower()
        except Exception as e:
            print("  " + view + " ERROR: " + str(e))
            ans="ERROR"; pred="Unknown"; ok=False
        row.extend([ans, pred, ok])
        stats[view][1] += 1
        if ok: stats[view][0] += 1
        if view == "exo": exo_preds[pred] += 1
        else: ego_preds[pred] += 1
        print("  " + view + ": " + pred + " " + ("OK" if ok else "X"))

    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print()

n = len(benchmark)
print("=" * 60)
print("RESULTS — Qwen 3-class (Novice/Intermediate/Expert)")
print("=" * 60)
for v, (c,t) in stats.items():
    print("  " + v + ": " + str(c) + "/" + str(t) + " = " + str(round(c/t*100,1)) + "%" if t else "")
print("Random chance: 33.3%")
print("Exo predictions: " + str(dict(exo_preds.most_common())))
print("Ego predictions: " + str(dict(ego_preds.most_common())))
print("=" * 60)
