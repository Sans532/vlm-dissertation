"""
Video-LLaVA | Structured prompt | 16 frames | Exo + Ego
Investigates the 16-frame anomaly (76% exo / 32% ego on binary).
Untrimmed — same setup as original binary_n16 that produced the anomaly.
"""
import json, os, csv, gc, warnings, re
import torch, cv2, numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER       = os.environ.get("USER")
MODEL_PATH = "/home/" + USER + "/dissertation/models/videollava"
DATA_DIR   = "/home/" + USER + "/dissertation/data/egoexo"
BENCHMARK  = "/home/" + USER + "/dissertation/repo/dissertation_v2/benchmark/benchmark_binary.json"
RESULTS    = "/home/" + USER + "/dissertation/repo/dissertation_v2/results/videollava/bin_struct_n16.csv"
NUM_FRAMES = 16

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

QUESTION = (
    "Watch these frames carefully.\n"
    "Step 1: Describe the person's body position and technique in detail.\n"
    "Step 2: Identify any errors or imprecisions in their movement.\n"
    "Step 3: Classify skill level as exactly one of: Novice / Expert.\n"
    "Format your answer as:\n"
    "Observations: ...\n"
    "Errors: ...\n"
    "Skill Level: ..."
)

print("Loading Video-LLaVA ...")
model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = VideoLlavaProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def get_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames, idx = [], 0
    idx_set = set(indices)
    for frame in iter(lambda: cap.read(), (False, None)):
        ret, f = frame
        if not ret:
            break
        if idx in idx_set:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        idx += 1
        if len(frames) >= num_frames:
            break
    cap.release()
    if not frames:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames)


def ask(video_path, question):
    video = get_frames(video_path, NUM_FRAMES)
    if video is None:
        raise Exception("No frames")
    prompt = "USER: <video>\nThese are " + str(NUM_FRAMES) + " frames sampled from a video of a person performing an activity. " + question + " ASSISTANT:"
    inputs = processor(text=prompt, videos=video, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw else raw.strip()
    del inputs, out, video
    torch.cuda.empty_cache(); gc.collect()
    return clean


def extract_label(answer):
    a = answer.lower()
    match = re.search(r'skill level[:\s]+(.+?)(?:\n|$)', a)
    search_text = match.group(1).strip() if match else a
    if "expert" in search_text:
        return "Expert"
    if "novice" in search_text:
        return "Novice"
    if "expert" in a:
        return "Expert"
    if "novice" in a:
        return "Novice"
    return "Unknown"


benchmark = json.load(open(BENCHMARK))
print("Clips: " + str(len(benchmark)) + " | frames: " + str(NUM_FRAMES) + "\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow([
        "clip_id", "take_folder", "ground_truth",
        "exo_full_answer", "exo_predicted", "exo_correct",
        "ego_full_answer", "ego_predicted", "ego_correct"
    ])

stats = {"exo": [0, 0], "ego": [0, 0]}
exo_preds = Counter()
ego_preds = Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])
    row = [item["clip_id"], item["take_folder"], gt]

    print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + item["take_folder"] + " (GT=" + gt + ")")

    for view, path in [("exo", exo_path), ("ego", ego_path)]:
        try:
            ans = ask(path, QUESTION)
            pred = extract_label(ans)
            ok = pred.lower() == gt.lower()
        except Exception as e:
            print("  " + view + " ERROR: " + str(e))
            ans = "ERROR"; pred = "Unknown"; ok = False
        row.extend([ans, pred, ok])
        stats[view][1] += 1
        if ok: stats[view][0] += 1
        if view == "exo": exo_preds[pred] += 1
        else: ego_preds[pred] += 1
        print("  " + view + ": " + pred + " " + ("OK" if ok else "X") + " | " + ans[:100].replace("\n", " "))

    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print()

n = len(benchmark)
print("=" * 60)
print("RESULTS — VideoLLaVA structured 16f (investigating anomaly)")
print("=" * 60)
for v, (c, t) in stats.items():
    print("  " + v + ": " + str(c) + "/" + str(t) + " = " + str(round(c/t*100, 1)) + "%" if t else "")
print("\nRandom chance: 50%")
print("\nExo predictions: " + str(dict(exo_preds.most_common())))
print("Ego predictions: " + str(dict(ego_preds.most_common())))
print("\nCompare to binary_n16 (no reasoning): exo=76%, ego=32%")
print("=" * 60)
