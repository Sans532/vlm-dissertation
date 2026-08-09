"""
Qwen2.5-VL-7B | Climbing | Reasoning | 8 frames | best_exo (per-clip optimal camera)
Video-based (not text-only). Uses the dataset's best_exo field instead of the
fixed cam01 used elsewhere, to check whether camera choice affects the
reasoning-prompt collapse pattern. Companion to qwen_binary_bestexo.py.
"""
import json, os, csv, gc, warnings
import torch, cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER       = os.environ.get("USER")
MODEL_PATH = "/home/" + USER + "/dissertation/models/qwen25vl-7b"
DATA_DIR   = "/home/" + USER + "/dissertation/data/egoexo"
TAKES_PATH = "/home/" + USER + "/dissertation/data/egoexo/takes.json"
BENCHMARK  = "/home/" + USER + "/dissertation/repo/diss_climb/benchmark/benchmark_reasoning.json"
RESULTS    = "/home/" + USER + "/dissertation/repo/diss_climb/results/qwen/qwen_climbing_bestexo_n8_reasoning.csv"
NUM_FRAMES = 8
LABELS     = ["Late Expert", "Intermediate Expert", "Early Expert", "Novice"]

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

QUESTION = (
    "You are an expert coach evaluating this person's technique.\n"
    "Focus on: body alignment, movement fluency, technical precision.\n"
    "What is their skill level? "
    "Novice / Early Expert / Intermediate Expert / Late Expert\n"
    "Explain your reasoning."
)

print("Loading takes metadata...")
takes_list = json.load(open(TAKES_PATH))
take_info = {t.get("take_name", ""): t for t in takes_list}
print("Loaded " + str(len(take_info)) + " takes.\n")

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def get_best_exo_path(item):
    info = take_info.get(item["take_folder"], {})
    best_exo_cam = info.get("best_exo", "")
    if not best_exo_cam:
        return item["video_path_exo"], "cam01(fallback-no-best_exo)"
    original_path = item["video_path_exo"]
    best_path = original_path.replace("cam01", best_exo_cam)
    return best_path, best_exo_cam


def get_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (420, 360)), cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def ask(video_path, question):
    frames = get_frames(video_path, NUM_FRAMES)
    if not frames:
        raise Exception("No frames")
    content = [{"type": "image"} for _ in frames]
    content.append({"type": "text", "text": "These are " + str(NUM_FRAMES) + " frames from a video of a person performing an activity. " + question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out, frames
    torch.cuda.empty_cache(); gc.collect()
    return clean


def extract_label(answer):
    """Earliest-mention-wins extraction, same fix used for the Gemini reasoning bug."""
    a = answer.lower()
    positions = {}
    for label in LABELS:
        pos = a.find(label.lower())
        if pos != -1:
            positions[label] = pos
    if positions:
        return min(positions, key=positions.get)
    return "Unknown"


benchmark = json.load(open(BENCHMARK))
print("Clips: " + str(len(benchmark)) + " | frames: " + str(NUM_FRAMES) + " | best_exo per clip\n")
print("Prompt: " + QUESTION + "\n")

with open(RESULTS, "w", newline="") as f:
    csv.writer(f).writerow(["clip_id", "take_folder", "ground_truth", "camera_used",
                             "full_answer", "predicted", "correct"])

correct = 0
preds = Counter()
by_class = Counter()
correct_by_class = Counter()
camera_usage = Counter()

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    take_folder = item["take_folder"]

    best_exo_path_rel, camera_used = get_best_exo_path(item)
    full_path = os.path.join(DATA_DIR, best_exo_path_rel)
    camera_usage[camera_used] += 1

    print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + take_folder + " (GT=" + gt + ", camera=" + camera_used + ")")

    try:
        if not os.path.exists(full_path):
            raise Exception("Video not found at " + full_path)
        ans = ask(full_path, QUESTION)
        pred = extract_label(ans)
        ok = pred.lower() == gt.lower()
    except Exception as e:
        print("  ERROR: " + str(e))
        ans = "ERROR"; pred = "Unknown"; ok = False

    preds[pred] += 1
    by_class[gt] += 1
    if ok:
        correct += 1
        correct_by_class[gt] += 1

    print("  " + pred + " " + ("OK" if ok else "X") + " | " + ans[:80].replace("\n", " "))

    with open(RESULTS, "a", newline="") as f:
        csv.writer(f).writerow([item["clip_id"], take_folder, gt, camera_used, ans, pred, ok])

n = len(benchmark)
print("\n" + "=" * 60)
print("RESULTS -- Qwen reasoning 8f, best_exo per clip")
print("=" * 60)
print("Overall: " + str(correct) + "/" + str(n) + " = " + str(round(correct/n*100, 1)) + "%")
print("Random chance: 25%")
print("\nPer class:")
for level in LABELS:
    c = correct_by_class[level]
    t = by_class[level]
    print("  " + level + ": " + str(c) + "/" + str(t) + " = " + (str(round(c/t*100,1))+"%" if t else "N/A"))
print("\nPrediction distribution: " + str(dict(preds.most_common())))
print("\nCamera usage: " + str(dict(camera_usage.most_common())))
print("=" * 60)
print("\nCompare against existing cam01-fixed reasoning result.")
