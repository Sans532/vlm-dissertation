"""
Videollava | Binary | 8 frames | exo only | TRIMMED to task window
Uses task_start_sec / task_end_sec from takes.json so all frames
show active climbing only.
"""
import json, os, csv, gc, warnings
import torch
import cv2
from PIL import Image
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")

USER          = os.environ.get("USER")
MODEL_PATH    = "/home/" + USER + "/dissertation/models/videollava"
DATA_DIR      = "/home/" + USER + "/dissertation/data/egoexo"
TAKES_PATH    = "/home/" + USER + "/dissertation/data/egoexo/takes.json"
BENCHMARK     = "/home/" + USER + "/dissertation/repo/dissertation_v2/benchmark/benchmark_binary.json"
RESULTS       = "/home/" + USER + "/dissertation/repo/dissertation_v2/results/videollava/trimmed/binary_exo_n8.csv"
NUM_FRAMES    = 8

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

QUESTION = "Is this person a Novice or an Expert at this activity? Answer only: Novice or Expert"

# Load takes.json for timing
print("Loading takes metadata...")
takes_list = json.load(open(TAKES_PATH))
take_info = {}
for t in takes_list:
    take_info[t.get("take_name", "")] = t
print("Loaded " + str(len(take_info)) + " takes.\n")

print("Loading Videollava ...")
model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = VideoLlavaProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")

def get_frames_trimmed(video_path, take_folder):
    info = take_info.get(take_folder, {})
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0 or fps_video == 0:
        cap.release()
        return None

    start_sec = info.get("task_start_sec", 0)
    end_sec = info.get("task_end_sec", total / fps_video)
    start_frame = max(0, int(start_sec * fps_video))
    end_frame = min(total, int(end_sec * fps_video))

    indices = [int(start_frame + i * (end_frame - start_frame) / NUM_FRAMES) for i in range(NUM_FRAMES)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()

    if not frames:
        return None
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])
    return np.stack(frames)

def ask_videollava(video_path, take_folder, question):
    video = get_frames_trimmed(video_path, take_folder)
    if video is None:
        raise Exception("No frames")
    prompt = "USER: <video>\nThese are 8 frames sampled from a video of a person performing an activity. " + question + " ASSISTANT:"
    inputs = processor(text=prompt, videos=video, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw else raw.strip()
    del inputs, out, video
    torch.cuda.empty_cache()
    gc.collect()
    return clean

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


benchmark = json.load(open(BENCHMARK))
print("Clips: " + str(len(benchmark)))
print("Frames: " + str(NUM_FRAMES) + " (from task window only)")
print("Prompt: " + QUESTION + "\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow(["clip_id", "take_folder", "ground_truth", "task_start", "task_end", "answer", "predicted", "correct"])

correct_total = 0
nov_correct = 0
nov_total = 0
exp_correct = 0
exp_total = 0
preds = Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    take_folder = item["take_folder"]

    info = take_info.get(take_folder, {})
    start = round(info.get("task_start_sec", 0), 1)
    end = round(info.get("task_end_sec", 0), 1)

    print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + take_folder + " (GT=" + gt + ") task=" + str(start) + "-" + str(end) + "s")

    try:
        ans = ask_videollava(exo_path, take_folder, QUESTION)
        ok = check(ans, gt)
        if "novice" in ans.lower() and "expert" not in ans.lower():
            pred = "Novice"
        elif "expert" in ans.lower():
            pred = "Expert"
        else:
            pred = ans.strip()
    except Exception as e:
        print("  ERROR: " + str(e))
        ans = "ERROR"
        ok = False
        pred = "Unknown"

    preds[pred] += 1
    correct_total += int(ok)
    if gt == "Novice":
        nov_total += 1
        nov_correct += int(ok)
    else:
        exp_total += 1
        exp_correct += int(ok)

    print("  " + pred + " " + ("OK" if ok else "X"))

    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow([item["clip_id"], take_folder, gt, start, end, ans, pred, ok])

n = len(benchmark)
print("\n" + "=" * 60)
print("RESULTS — Videollava binary 8f TRIMMED (task window only)")
print("=" * 60)
print("Overall:  " + str(correct_total) + "/" + str(n) + " = " + str(round(correct_total/n*100, 1)) + "%")
print("Novice:   " + str(nov_correct) + "/" + str(nov_total) + " = " + str(round(nov_correct/nov_total*100, 1)) + "%" if nov_total else "")
print("Expert:   " + str(exp_correct) + "/" + str(exp_total) + " = " + str(round(exp_correct/exp_total*100, 1)) + "%" if exp_total else "")
print("Random:   50%")
print("Answers:  " + str(dict(preds.most_common())))
print("=" * 60)
