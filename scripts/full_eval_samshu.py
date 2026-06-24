import json, os, csv, gc, warnings, random
import torch
import cv2
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen3vl-2b"
DATA_DIR = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/benchmark/benchmark_balanced.json"
RESULTS_PATH = f"/home/{USER}/dissertation/results/eval_full_SAMSHU.csv"

# Seed for reproducible shuffle
random.seed(42)

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


# CHANGE 1: ask_qwen now accepts sampling toggle
def ask_qwen(video_path, question, use_sampling=False):
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

    # CHANGE 2: Switch between greedy and sampling based on use_sampling flag
    if use_sampling:
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    else:
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


# CHANGE 3: Functions to generate prompts with SHUFFLED label order per clip
def make_binary_prompt(seed):
    rng = random.Random(seed)
    options = ["Beginner", "Expert"]
    rng.shuffle(options)
    return f"Is this person a beginner or an expert at this activity? Answer only one: {options[0]} or {options[1]}"


def make_fourclass_prompt(seed):
    rng = random.Random(seed)
    options = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]
    rng.shuffle(options)
    return f"What is the skill level of the person in this video? Answer only one: {' / '.join(options)}"


def make_structured_prompt(seed):
    rng = random.Random(seed)
    options = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]
    rng.shuffle(options)
    return (
        "Watch this video carefully.\n"
        "Step 1: Describe the person's body position and technique in detail.\n"
        "Step 2: Identify any errors or imprecisions in their movement.\n"
        f"Step 3: Based only on what you observed, classify skill level: {' / '.join(options)}.\n"
        "Format your answer as:\n"
        "Observations: ...\n"
        "Errors: ...\n"
        "Skill Level: ..."
    )


with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

# CHANGE 4: Now testing 4 conditions per prompt type = 12 evals per clip
# (greedy + original order, greedy + shuffled, sampling + original, sampling + shuffled)
# To keep runtime manageable, we'll do: greedy+shuffled and sampling+shuffled
# That's 6 evals per clip (3 prompts x 2 conditions x 2 views = 12, but we only do 2 conditions)
# Simplified: 3 prompts x 2 decode strategies x 2 views = 12 evals

print(f"Testing {len(benchmark)} clips")
print("Conditions: 3 prompts x 2 decoding (greedy/sampling) x 2 views (ego/exo) = 12 evals/clip")
print(f"Total inferences: {len(benchmark) * 12}\n")

# CSV header with all conditions
with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["clip_id", "take_folder", "gt_binary", "gt_4class"]
    for prompt in ["binary", "fourclass", "structured"]:
        for decode in ["greedy", "sample"]:
            for view in ["exo", "ego"]:
                header.append(f"{prompt}_{decode}_{view}_answer")
                header.append(f"{prompt}_{decode}_{view}_correct")
    writer.writerow(header)

stats = {}
for prompt in ["binary", "fourclass", "structured"]:
    for decode in ["greedy", "sample"]:
        for view in ["exo", "ego"]:
            stats[f"{prompt}_{decode}_{view}"] = [0, 0]

for i, item in enumerate(benchmark):
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])

    row = [item["clip_id"], item["take_folder"],
           item["ground_truth_binary"], item["ground_truth_4class"]]

    print(f"[{i+1}/{len(benchmark)}] {item['take_folder']} (GT: {item['ground_truth_binary']} / {item['ground_truth_4class']})")

    # CHANGE 5: Use per-clip seed for shuffle so it's reproducible but varies per clip
    clip_seed = hash(item["clip_id"]) % 100000

    # Build shuffled prompts ONCE per clip (same for both views)
    prompts = {
        "binary": make_binary_prompt(clip_seed),
        "fourclass": make_fourclass_prompt(clip_seed),
        "structured": make_structured_prompt(clip_seed),
    }

    for prompt_key in ["binary", "fourclass", "structured"]:
        question = prompts[prompt_key]
        if prompt_key == "binary":
            gt = item["ground_truth_binary"]
        else:
            gt = item["ground_truth_4class"]

        for decode_mode in ["greedy", "sample"]:
            use_sampling = (decode_mode == "sample")

            for view, video_path in [("exo", exo_path), ("ego", ego_path)]:
                try:
                    answer = ask_qwen(video_path, question, use_sampling=use_sampling)
                    correct = check_correct(answer, gt)
                except Exception as e:
                    print(f"  {prompt_key}_{decode_mode}_{view} ERROR: {e}")
                    answer = "ERROR"
                    correct = False

                row.extend([answer, correct])

                key = f"{prompt_key}_{decode_mode}_{view}"
                stats[key][1] += 1
                if correct:
                    stats[key][0] += 1

                marker = "OK" if correct else "X"
                ans_short = answer[:60].replace("\n", " ")
                print(f"  {prompt_key:10} {decode_mode:7} {view}: {marker} {ans_short}")

    print()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
for key, (correct, total) in stats.items():
    if total:
        print(f"{key:30} {correct}/{total} = {correct/total:.1%}")
print(f"{'='*60}")
