import json, os, csv, gc, warnings
import torch
import cv2
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen3vl-2b"
DATA_DIR = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/benchmark/benchmark_balanced.json"
RESULTS_PATH = f"/home/{USER}/dissertation/results/eval_full_balanced.csv"

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
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
        "text": f"These are 8 frames sampled from a video of a person performing an activity. {question}"
    })

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")

    out = model.generate(**inputs, max_new_tokens=300)
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


def check_correct(answer, gt):
    return gt.lower() in answer.lower()


with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

print(f"Testing {len(benchmark)} clips with 3 prompts x 2 views = 6 evals per clip")
print(f"Total inferences: {len(benchmark) * 6}\n")

with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "clip_id", "take_folder",
        "gt_binary", "gt_4class",
        "binary_exo_answer", "binary_exo_correct",
        "binary_ego_answer", "binary_ego_correct",
        "fourclass_exo_answer", "fourclass_exo_correct",
        "fourclass_ego_answer", "fourclass_ego_correct",
        "structured_exo_answer", "structured_exo_correct",
        "structured_ego_answer", "structured_ego_correct",
    ])

stats = {
    "binary_exo": [0, 0], "binary_ego": [0, 0],
    "fourclass_exo": [0, 0], "fourclass_ego": [0, 0],
    "structured_exo": [0, 0], "structured_ego": [0, 0],
}

for i, item in enumerate(benchmark):
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])

    row = [item["clip_id"], item["take_folder"],
           item["ground_truth_binary"], item["ground_truth_4class"]]

    print(f"[{i+1}/{len(benchmark)}] {item['take_folder']} (GT: {item['ground_truth_binary']} / {item['ground_truth_4class']})")

    for prompt_key, view, video_path in [
        ("binary", "exo", exo_path),
        ("binary", "ego", ego_path),
        ("fourclass", "exo", exo_path),
        ("fourclass", "ego", ego_path),
        ("structured", "exo", exo_path),
        ("structured", "ego", ego_path),
    ]:
        if prompt_key == "binary":
            question = item["question_binary"]
            gt = item["ground_truth_binary"]
        elif prompt_key == "fourclass":
            question = item["question_baseline"]
            gt = item["ground_truth_4class"]
        else:
            question = item["question_structured"]
            gt = item["ground_truth_4class"]

        try:
            answer = ask_qwen(video_path, question)
            correct = check_correct(answer, gt)
        except Exception as e:
            print(f"  {prompt_key}_{view} ERROR: {e}")
            answer = "ERROR"
            correct = False

        row.extend([answer, correct])

        key = f"{prompt_key}_{view}"
        stats[key][1] += 1
        if correct:
            stats[key][0] += 1

        marker = "OK" if correct else "X"
        ans_short = answer[:80].replace("\n", " ")
        print(f"  {prompt_key:10} {view}: {marker} {ans_short}")

    print()

    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
for key, (correct, total) in stats.items():
    if total:
        print(f"{key:20} {correct}/{total} = {correct/total:.1%}")
print(f"{'='*60}")
