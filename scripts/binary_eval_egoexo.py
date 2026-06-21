import json, os, csv, gc, warnings
import torch
import cv2
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen3vl-2b"
DATA_DIR = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/benchmark/benchmark_500_available.json"
RESULTS_PATH = f"/home/{USER}/dissertation/results/eval_binary_egoexo_500.csv"

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.")


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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def ask_qwen(video_path, question):
    frames = get_video_frames(video_path, num_frames=8)
    if not frames:
        raise Exception("Could not extract frames")

    content = [{"type": "image"} for _ in frames]
    content.append({
        "type": "text",
        "text": f"These are 8 frames sampled from a video of a person doing rock climbing. {question}"
    })

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")

    out = model.generate(**inputs, max_new_tokens=200)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]

    if "assistant\n" in raw:
        clean = raw.split("assistant\n")[-1].strip()
    elif "assistant" in raw.lower():
        clean = raw.split("assistant")[-1].strip()
    else:
        clean = raw.strip()

    del inputs, out, frames
    torch.cuda.empty_cache()
    gc.collect()

    return clean


# Load benchmark
with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

print(f"Testing {len(benchmark)} clips (both ego and exo)...\n")

# Write CSV header
with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "clip_id", "take_folder", "activity",
        "ground_truth_binary", "ground_truth_4class",
        "answer_exo", "correct_exo",
        "answer_ego", "correct_ego"
    ])

# Track accuracy
exo_correct = 0
ego_correct = 0
total_tested = 0

for i, item in enumerate(benchmark):
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])

    answer_exo = "ERROR"
    answer_ego = "ERROR"
    correct_exo = False
    correct_ego = False

    # Exocentric
    try:
        answer_exo = ask_qwen(exo_path, item["question_binary"])
        correct_exo = item["ground_truth_binary"].lower() in answer_exo.lower()
    except Exception as e:
        print(f"[{i+1}] EXO ERROR: {e}")
        torch.cuda.empty_cache()
        gc.collect()

    # Egocentric
    try:
        answer_ego = ask_qwen(ego_path, item["question_binary"])
        correct_ego = item["ground_truth_binary"].lower() in answer_ego.lower()
    except Exception as e:
        print(f"[{i+1}] EGO ERROR: {e}")
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[{i+1}/{len(benchmark)}] {item['take_folder']}")
    print(f"  Ground truth: {item['ground_truth_binary']}")
    print(f"  EXO answer: {answer_exo} {'✓' if correct_exo else '✗'}")
    print(f"  EGO answer: {answer_ego} {'✓' if correct_ego else '✗'}")
    print()

    # Write to CSV
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            item["clip_id"], item["take_folder"], item["activity"],
            item["ground_truth_binary"], item["ground_truth_4class"],
            answer_exo, correct_exo,
            answer_ego, correct_ego
        ])

    if answer_exo != "ERROR" and answer_ego != "ERROR":
        total_tested += 1
        if correct_exo: exo_correct += 1
        if correct_ego: ego_correct += 1

# Summary
print(f"\n{'='*50}")
print(f"RESULTS SUMMARY — {total_tested} clips")
print(f"{'='*50}")
if total_tested:
    print(f"Exocentric accuracy: {exo_correct}/{total_tested} = {exo_correct/total_tested:.1%}")
    print(f"Egocentric accuracy: {ego_correct}/{total_tested} = {ego_correct/total_tested:.1%}")
print(f"{'='*50}")
