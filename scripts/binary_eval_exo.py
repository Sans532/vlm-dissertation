import json, os, csv, gc, warnings
import torch
import cv2
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen3vl-2b"
DATA_DIR = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/benchmark/benchmark_available.json"
RESULTS_PATH = f"/home/{USER}/dissertation/results/eval_binary.csv"

# Load model
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

    # Build message content - one {"type": "image"} per frame
    content = []
    for _ in frames:
        content.append({"type": "image"})
    content.append({
        "type": "text",
        "text": f"These are 8 frames sampled from a video of a person doing rock climbing. {question}"
    })

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=frames,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    out = model.generate(**inputs, max_new_tokens=200)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]

    # Extract answer after assistant marker
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

print(f"Testing {len(benchmark)} clips...\n")

# Write CSV header
with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["clip_id", "take_folder", "activity", "ground_truth_binary", "answer_binary", "correct"])

# Track accuracy
total_correct = 0
total_tested = 0
beginner_correct = 0
beginner_total = 0
expert_correct = 0
expert_total = 0

for i, item in enumerate(benchmark):
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])

    try:
        answer = ask_qwen(exo_path, item["question_binary"])
        correct = item["ground_truth_binary"].lower() in answer.lower()

        print(f"[{i+1}/{len(benchmark)}] {item['take_folder']}")
        print(f"  Ground truth: {item['ground_truth_binary']}")
        print(f"  Model answer: {answer}")
        print(f"  Correct: {'✓' if correct else '✗'}")
        print()

        # Write to CSV immediately
        with open(RESULTS_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                item["clip_id"], item["take_folder"], item["activity"],
                item["ground_truth_binary"], answer, correct
            ])

        total_tested += 1
        if correct:
            total_correct += 1
        if item["ground_truth_binary"] == "Beginner":
            beginner_total += 1
            if correct:
                beginner_correct += 1
        else:
            expert_total += 1
            if correct:
                expert_correct += 1

    except Exception as e:
        print(f"[{i+1}] ERROR: {e}\n")
        with open(RESULTS_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                item["clip_id"], item["take_folder"], item["activity"],
                item["ground_truth_binary"], "ERROR", False
            ])
        torch.cuda.empty_cache()
        gc.collect()

# Print summary
print(f"\n{'='*50}")
print(f"RESULTS SUMMARY")
print(f"{'='*50}")
print(f"Total clips tested: {total_tested}")
if total_tested:
    print(f"Overall accuracy:   {total_correct}/{total_tested} = {total_correct/total_tested:.1%}")
if beginner_total:
    print(f"Beginner clips:     {beginner_correct}/{beginner_total} = {beginner_correct/beginner_total:.1%}")
if expert_total:
    print(f"Expert clips:       {expert_correct}/{expert_total} = {expert_correct/expert_total:.1%}")
print(f"{'='*50}")
