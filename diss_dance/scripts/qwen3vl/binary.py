"""
Qwen3-VL-8B | Dance | Binary | OPENCV FRAMES (true ~1fps, matches Gemini fps=1) | Exo + Ego
Bypasses the native video pipeline (fps=24 fallback bug -> OOM).
"""
import json, os, csv, gc, warnings
import torch, cv2
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER       = os.environ.get("USER")
MODEL_PATH = "/home/" + USER + "/dissertation/models/qwen3vl-8b"
DATA_DIR   = "/home/" + USER + "/data_dance/videos"
BENCHMARK  = "/home/" + USER + "/dissertation/repo/diss_dance/benchmark/benchmark_binary_dance.json"
RESULTS    = "/home/" + USER + "/dissertation/repo/diss_dance/results/qwen3vl/qwen3vl_dance_frames_binary.csv"

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

QUESTION = "Is this person a Novice or an Expert at this activity? Answer only: Novice or Expert"

MAX_SIDE = 448  # resize longest side of each frame (keeps memory bounded without changing frame count)

print("Loading Qwen3-VL-8B ...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def extract_frames(video_path):
    """Sample frames at true ~1fps (one frame per second of video), matching Gemini's fps=1 setting."""
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps_video if fps_video > 0 else 0
    n = max(1, round(duration_sec))

    frames = []
    if total_frames <= 0:
        cap.release()
        return frames, 0
    indices = [int(i * (total_frames - 1) / max(1, n - 1)) for i in range(n)] if n > 1 else [total_frames // 2]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames, len(frames)


def ask_frames(video_path, question, max_new_tokens=50):
    frames, n = extract_frames(video_path)
    if n == 0:
        raise Exception("No frames extracted")
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out, frames
    torch.cuda.empty_cache(); gc.collect()
    return clean, n


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
print("Clips: " + str(len(benchmark)) + " | OPENCV FRAMES true ~1fps (matches Gemini fps=1) | Dance\n")
print("Prompt: " + QUESTION + "\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow(["clip_id", "take_folder", "ground_truth",
                             "exo_nframes", "exo_answer", "exo_predicted", "exo_correct",
                             "ego_nframes", "ego_answer", "ego_predicted", "ego_correct"])

stats = {"exo": [0, 0], "ego": [0, 0]}
exo_preds = Counter(); ego_preds = Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])
    row = [item["clip_id"], item["take_folder"], gt]

    print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + item["take_folder"] + " (GT=" + gt + ")")

    for view, path in [("exo", exo_path), ("ego", ego_path)]:
        try:
            if not os.path.exists(path):
                raise Exception("Video not found")
            ans, n = ask_frames(path, QUESTION)
            ok = check(ans, gt)
            if "novice" in ans.lower() and "expert" not in ans.lower(): pred = "Novice"
            elif "expert" in ans.lower(): pred = "Expert"
            else: pred = ans.strip()
        except Exception as e:
            print("  " + view + " ERROR: " + str(e))
            ans = "ERROR"; ok = False; pred = "Unknown"; n = 0
        row.extend([n, ans, pred, ok])
        stats[view][1] += 1
        if ok: stats[view][0] += 1
        if view == "exo": exo_preds[pred] += 1
        else: ego_preds[pred] += 1
        print("  " + view + " (n=" + str(n) + "): " + pred + " " + ("OK" if ok else "X"))

    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print()

n_total = len(benchmark)
print("=" * 60)
print("RESULTS — Qwen3-VL frames binary (Dance)")
print("=" * 60)
for v, (c, t) in stats.items():
    print("  " + v + ": " + str(c) + "/" + str(t) + " = " + str(round(c/t*100, 1)) + "%" if t else "")
print("Random chance: 50%")
print("Exo predictions: " + str(dict(exo_preds.most_common())))
print("Ego predictions: " + str(dict(ego_preds.most_common())))
print("=" * 60)
