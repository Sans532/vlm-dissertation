"""
Dance | Video-LLaVA | Binary | 8 frames | Exo + Ego | TRIMMED
"""
import json, os, csv, gc, warnings
import torch, cv2, numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER       = os.environ.get("USER")
BASE_DIR   = "/home/" + USER + "/dissertation/repo/diss_dance"
MODEL_PATH = "/home/" + USER + "/dissertation/models/videollava"
DATA_DIR   = "/home/" + USER + "/dissertation/data/egoexo"
TAKES_PATH = "/home/" + USER + "/dissertation/data/egoexo/takes.json"
BENCHMARK  = BASE_DIR + "/benchmark/benchmark_binary_dance.json"
RESULTS    = BASE_DIR + "/results/videollava/trimmed/binary_n8.csv"
NUM_FRAMES = 8

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

QUESTION = "Is this person a Novice or an Expert at this activity? Answer only: Novice or Expert"

print("Loading takes metadata...")
takes_list = json.load(open(TAKES_PATH))
take_info = {}
for t in takes_list:
    take_info[t.get("take_name", "")] = t
print("Loaded " + str(len(take_info)) + " takes.\n")

print("Loading Video-LLaVA ...")
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
        ret, f = cap.read()
        if ret:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])
    return np.stack(frames)


def ask(video_path, take_folder, question):
    video = get_frames_trimmed(video_path, take_folder)
    if video is None:
        raise Exception("No frames")
    prompt = "USER: <video>\nThese are " + str(NUM_FRAMES) + " frames from a video of a person dancing. " + question + " ASSISTANT:"
    inputs = processor(text=prompt, videos=video, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw else raw.strip()
    del inputs, out, video
    torch.cuda.empty_cache(); gc.collect()
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
print("Clips: " + str(len(benchmark)) + " | frames: " + str(NUM_FRAMES) + " | TRIMMED | Dance\n")
print("Prompt: " + QUESTION + "\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow(["clip_id", "take_folder", "ground_truth",
                             "exo_answer", "exo_predicted", "exo_correct",
                             "ego_answer", "ego_predicted", "ego_correct"])

stats = {"exo": [0, 0], "ego": [0, 0]}
exo_preds = Counter()
ego_preds = Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])
    take_folder = item["take_folder"]
    row = [item["clip_id"], take_folder, gt]

    print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + take_folder + " (GT=" + gt + ")")

    for view, path in [("exo", exo_path), ("ego", ego_path)]:
        try:
            ans = ask(path, take_folder, QUESTION)
            ok = check(ans, gt)
            if "novice" in ans.lower() and "expert" not in ans.lower():
                pred = "Novice"
            elif "expert" in ans.lower():
                pred = "Expert"
            else:
                pred = ans.strip()
        except Exception as e:
            print("  " + view + " ERROR: " + str(e))
            ans = "ERROR"; ok = False; pred = "Unknown"
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
print("RESULTS — Dance | VideoLLaVA binary 8f TRIMMED")
print("=" * 60)
for v, (c, t) in stats.items():
    print("  " + v + ": " + str(c) + "/" + str(t) + " = " + str(round(c/t*100, 1)) + "%" if t else "")
print("Random chance: 50%")
print("Exo predictions: " + str(dict(exo_preds.most_common())))
print("Ego predictions: " + str(dict(ego_preds.most_common())))
print("=" * 60)
