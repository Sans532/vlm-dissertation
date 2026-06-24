import json, os, csv, gc, warnings, random
import torch
import numpy as np
import av
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/videollava"
DATA_DIR = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/benchmark/benchmark_balanced.json"
RESULTS_PATH = f"/home/{USER}/dissertation/results/videollava/eval_videollava_full.csv"

random.seed(42)

print("Loading Video-LLaVA model...")
model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
processor = VideoLlavaProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def read_video_pyav(video_path, num_frames=8):
    """Video-LLaVA expects numpy array of shape (num_frames, height, width, 3)"""
    container = av.open(video_path)
    total = container.streams.video[0].frames
    if total == 0:
        # Stream doesn't report total - count manually
        frames_list = []
        for frame in container.decode(video=0):
            frames_list.append(frame)
        total = len(frames_list)
        container.close()
        container = av.open(video_path)
    
    if total == 0:
        container.close()
        return None
    
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    frame_idx = 0
    for frame in container.decode(video=0):
        if frame_idx in indices:
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
        frame_idx += 1
        if len(frames) >= num_frames:
            break
    container.close()
    
    if len(frames) == 0:
        return None
    
    # Pad if we got fewer frames than expected
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return np.stack(frames)


def ask_videollava(video_path, question, use_sampling=False):
    video = read_video_pyav(video_path, num_frames=8)
    if video is None:
        raise Exception("Could not extract frames")

    # Video-LLaVA uses a specific prompt template with <video> token
    prompt_text = (
        f"USER: <video>\nThese are 8 frames sampled from a video of a person performing an activity. "
        f"{question} ASSISTANT:"
    )

    inputs = processor(text=prompt_text, videos=video, return_tensors="pt").to("cuda")

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

    # Extract answer after ASSISTANT marker
    if "ASSISTANT:" in raw:
        clean = raw.split("ASSISTANT:")[-1].strip()
    else:
        clean = raw.strip()

    del inputs, out, video
    torch.cuda.empty_cache()
    gc.collect()

    return clean


def check_correct(answer, gt):
    return gt.lower() in answer.lower()


# Shuffled prompts (same approach as Qwen ablation)
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
        "Look at these 8 frames carefully.\n"
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

print(f"Testing {len(benchmark)} clips on Video-LLaVA")
print("Conditions: 3 prompts x 2 views = 6 evals/clip (greedy only to save time)")
print(f"Total inferences: {len(benchmark) * 6}\n")

with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["clip_id", "take_folder", "gt_binary", "gt_4class"]
    for prompt in ["binary", "fourclass", "structured"]:
        for view in ["exo", "ego"]:
            header.append(f"{prompt}_{view}_answer")
            header.append(f"{prompt}_{view}_correct")
    writer.writerow(header)

stats = {}
for prompt in ["binary", "fourclass", "structured"]:
    for view in ["exo", "ego"]:
        stats[f"{prompt}_{view}"] = [0, 0]

for i, item in enumerate(benchmark):
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, item["video_path_ego"])

    row = [item["clip_id"], item["take_folder"],
           item["ground_truth_binary"], item["ground_truth_4class"]]

    print(f"[{i+1}/{len(benchmark)}] {item['take_folder']} (GT: {item['ground_truth_binary']} / {item['ground_truth_4class']})")

    clip_seed = hash(item["clip_id"]) % 100000

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

        for view, video_path in [("exo", exo_path), ("ego", ego_path)]:
            try:
                answer = ask_videollava(video_path, question, use_sampling=False)
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
            ans_short = answer[:60].replace("\n", " ")
            print(f"  {prompt_key:10} {view}: {marker} {ans_short}")

    print()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

print(f"\n{'='*60}")
print("FINAL RESULTS - Video-LLaVA")
print(f"{'='*60}")
for key, (correct, total) in stats.items():
    if total:
        print(f"{key:20} {correct}/{total} = {correct/total:.1%}")
print(f"{'='*60}")
