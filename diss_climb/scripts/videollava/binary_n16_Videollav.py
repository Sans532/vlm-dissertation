"""
Video-LLaVA binary skill assessment — greedy decoding, fixed prompt order.
"""
import json, os, csv, gc, warnings
import torch
import numpy as np
import av
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH     = f"/home/{USER}/dissertation/models/videollava"
DATA_DIR       = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_binary.json"
RESULTS_PATH   = f"/home/{USER}/dissertation/repo/dissertation_v2/results/videollava/binary_normal.csv"

USE_SAMPLING = False
SHUFFLE_OPTIONS = False

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

print("Loading Video-LLaVA ...")
model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
)
processor = VideoLlavaProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def read_video_pyav(video_path, num_frames=16):
    container = av.open(video_path)
    total = container.streams.video[0].frames
    if total == 0:
        frames_list = list(container.decode(video=0))
        total = len(frames_list)
        container.close()
        container = av.open(video_path)
    if total == 0:
        container.close()
        return None
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames, idx = [], 0
    for frame in container.decode(video=0):
        if idx in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        idx += 1
        if len(frames) >= num_frames:
            break
    container.close()
    if not frames:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames)


def make_binary_prompt(_seed):
    return "Is this person a novice or an expert at this activity? Answer only: Novice or Expert"


def ask_videollava(video_path, question):
    video = read_video_pyav(video_path, num_frames=16)
    if video is None:
        raise Exception("Could not extract frames")
    prompt = (
        f"USER: <video>\nThese are 16 frames sampled from a video of a person performing an activity. "
        f"{question} ASSISTANT:"
    )
    inputs = processor(text=prompt, videos=video, return_tensors="pt").to("cuda")
    if USE_SAMPLING:
        out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9)
    else:
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw else raw.strip()
    del inputs, out, video
    torch.cuda.empty_cache()
    gc.collect()
    return clean


def check_correct(answer, gt):
    a = answer.lower()
    has_nov, has_exp = "novice" in a, "expert" in a
    if has_nov and not has_exp: return gt.lower() == "novice"
    if has_exp and not has_nov: return gt.lower() == "expert"
    pos_nov = a.find("novice") if has_nov else 10**9
    pos_exp = a.find("expert") if has_exp else 10**9
    if pos_nov == pos_exp: return False
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
            ans = ask_videollava(path, question)
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
print("FINAL — videollava binary greedy normal")
print("=" * 60)
for k, (c, t) in stats.items():
    if t:
        print(f"  {k}: {c}/{t} = {c/t:.1%}")
