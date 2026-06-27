"""
Qwen2.5-VL structured skill assessment — 4-class, greedy decoding.
Forces model to describe technique before classifying.
Tests both ego and exo views.
"""
import json, os, csv, gc, warnings, re
import torch
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH     = f"/home/{USER}/dissertation/models/qwen25vl-7b"
DATA_DIR       = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_structured.json"
RESULTS_PATH   = f"/home/{USER}/dissertation/repo/dissertation_v2/results/qwen/structured_eval.csv"

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

print("Loading Qwen2.5-VL-7B ...")
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

    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()

    del inputs, out, frames
    torch.cuda.empty_cache()
    gc.collect()
    return clean


def extract_skill_level(answer):
    """Extract the skill level label from a structured response."""
    answer_lower = answer.lower()

    # Try to find "Skill Level: X" pattern first
    match = re.search(r'skill level:\s*(.*?)(?:\n|$)', answer_lower)
    if match:
        label_text = match.group(1).strip()
    else:
        label_text = answer_lower

    # Check for each label (longest first to avoid partial matches)
    labels = ["late expert", "intermediate expert", "early expert", "novice"]
    for label in labels:
        if label in label_text:
            return label.title()  # "Late Expert", "Novice", etc.

    # Fallback: check entire answer if not found in Skill Level line
    for label in labels:
        if label in answer_lower:
            return label.title()

    return "Unknown"


def check_correct(predicted, gt):
    return predicted.lower() == gt.lower()


# Load benchmark
with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

print(f"Clips: {len(benchmark)}")
print(f"Total inferences: {len(benchmark) * 2} (ego + exo)\n")

# CSV header
with open(RESULTS_PATH, "w", newline="") as f:
    csv.writer(f).writerow([
        "clip_id", "take_folder", "ground_truth",
        "exo_full_answer", "exo_predicted", "exo_correct",
        "ego_full_answer", "ego_predicted", "ego_correct",
    ])

# Track accuracy per class
from collections import Counter
exo_by_class = Counter()
ego_by_class = Counter()
exo_correct_by_class = Counter()
ego_correct_by_class = Counter()
total_exo_correct = 0
total_ego_correct = 0
total_tested = 0

for i, item in enumerate(benchmark):
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])
    gt = item["ground_truth"]
    question = item["question_structured"]

    print(f"[{i+1}/{len(benchmark)}] {item['take_folder']} (GT={gt})")

    row = [item["clip_id"], item["take_folder"], gt]

    for view, path in [("exo", exo_path), ("ego", ego_path)]:
        try:
            answer = ask_qwen(path, question)
            predicted = extract_skill_level(answer)
            correct = check_correct(predicted, gt)
        except Exception as e:
            print(f"  {view} ERROR: {e}")
            answer = "ERROR"
            predicted = "Unknown"
            correct = False

        row.extend([answer, predicted, correct])

        if view == "exo":
            exo_by_class[gt] += 1
            if correct:
                exo_correct_by_class[gt] += 1
                total_exo_correct += 1
        else:
            ego_by_class[gt] += 1
            if correct:
                ego_correct_by_class[gt] += 1
                total_ego_correct += 1

        ans_short = answer[:100].replace("\n", " ")
        print(f"  {view}: predicted={predicted} {'OK' if correct else 'X'}")
        print(f"       {ans_short}")

    total_tested += 1
    print()

    with open(RESULTS_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)

# Summary
print(f"\n{'='*60}")
print("FINAL RESULTS — Qwen structured 4-class")
print(f"{'='*60}")
print(f"\nOverall:")
print(f"  Exo: {total_exo_correct}/{total_tested} = {total_exo_correct/total_tested:.1%}")
print(f"  Ego: {total_ego_correct}/{total_tested} = {total_ego_correct/total_tested:.1%}")
print(f"  Random chance: 25%")

print(f"\nPer class (EXO):")
for level in ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]:
    n = exo_by_class[level]
    c = exo_correct_by_class[level]
    print(f"  {level:25} {c}/{n} = {c/n:.1%}" if n else f"  {level:25} N/A")

print(f"\nPer class (EGO):")
for level in ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]:
    n = ego_by_class[level]
    c = ego_correct_by_class[level]
    print(f"  {level:25} {c}/{n} = {c/n:.1%}" if n else f"  {level:25} N/A")

# Distribution of model predictions
print(f"\nModel prediction distribution (EXO):")
exo_predictions = Counter()
with open(RESULTS_PATH) as f:
    for row in csv.DictReader(f):
        exo_predictions[row["exo_predicted"]] += 1
for pred, count in exo_predictions.most_common():
    print(f"  {pred}: {count}")
print(f"{'='*60}")
