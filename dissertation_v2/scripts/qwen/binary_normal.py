"""
Qwen3-VL binary skill assessment — greedy decoding, fixed prompt order.
Writes one CSV row per clip with exo/ego answers + correctness.
"""
import json, os, csv, gc, warnings
import torch
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH     = f"/home/{USER}/dissertation/models/qwen25vl-7b"
DATA_DIR       = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_binary.json"
RESULTS_PATH   = f"/home/{USER}/dissertation/repo/dissertation_v2/results/qwen/binary_normal.csv"

# --- experiment knobs ----------------------------------------------------
USE_SAMPLING = False           # greedy
SHUFFLE_OPTIONS = False        # fixed order: "Novice or Expert"
# -------------------------------------------------------------------------

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

print("Loading Qwen2.5-VL-7B-Instruct ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def get_video_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (420, 360))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def make_binary_prompt(_seed):
    # Fixed canonical order
    return "Is this person a novice or an expert at this activity? Answer only: Novice or Expert"


def ask_qwen(video_path, question):
    frames = get_video_frames(video_path, num_frames=8)
    if not frames:
        raise Exception("Could not extract frames")

    content = [{"type": "image"} for _ in frames]
    content.append({
        "type": "text",
        "text": f"These are 8 frames sampled from a video of a person performing an activity. {question}",
    })
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")

    if USE_SAMPLING:
        out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9)
    else:
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()

    del inputs, out, frames
    torch.cuda.empty_cache()
    gc.collect()
    return clean


def check_correct(answer, gt):
    a = answer.lower()
    has_nov = "novice" in a
    has_exp = "expert" in a
    if has_nov and not has_exp:
        return gt.lower() == "novice"
    if has_exp and not has_nov:
        return gt.lower() == "expert"
    # ambiguous: take first occurrence
    pos_nov = a.find("novice") if has_nov else 10**9
    pos_exp = a.find("expert") if has_exp else 10**9
    if pos_nov == pos_exp:
        return False
    return (gt.lower() == "novice") == (pos_nov < pos_exp)


with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

print(f"Clips: {len(benchmark)}   sampling={USE_SAMPLING}   shuffle={SHUFFLE_OPTIONS}")
print(f"Total inferences: {len(benchmark) * 2}\n")

with open(RESULTS_PATH, "w", newline="") as f:
    csv.writer(f).writerow([
        "clip_id", "take_folder", "gt_binary",
        "prompt", "exo_answer", "exo_correct", "ego_answer", "ego_correct",
    ])

stats = {"exo": [0, 0], "ego": [0, 0]}

for i, item in enumerate(benchmark):
    clip_seed = hash(item["clip_id"]) % 100000
    question = make_binary_prompt(clip_seed)
    gt = item["ground_truth"]
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])

    print(f"[{i+1}/{len(benchmark)}] {item['take_folder']} (GT={gt})")
    row = [item["clip_id"], item["take_folder"], gt, question]

    for view, path in [("exo", exo_path), ("ego", ego_path)]:
        try:
            ans = ask_qwen(path, question)
            ok = check_correct(ans, gt)
        except Exception as e:
            print(f"  {view} ERROR: {e}")
            ans, ok = "ERROR", False
        row.extend([ans, ok])
        stats[view][1] += 1
        stats[view][0] += int(ok)
        print(f"  {view}: {'OK' if ok else 'X '} {ans[:80]}")

    with open(RESULTS_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)

print("\n" + "=" * 60)
print("FINAL — qwen binary normal")
print("=" * 60)
for k, (c, t) in stats.items():
    if t:
        print(f"  {k}: {c}/{t} = {c/t:.1%}")
