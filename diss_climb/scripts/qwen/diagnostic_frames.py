
"""
Diagnostic: picks 1 clip per skill level, asks 5 questions,
saves the 8 extracted frames to disk so you can manually verify
what the model actually sees.
"""
import json, os, gc, warnings
import torch
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

warnings.filterwarnings("ignore")

USER          = os.environ.get("USER")
MODEL_PATH    = f"/home/{USER}/dissertation/models/qwen25vl-7b"
DATA_DIR      = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK     = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_structured.json"
FRAMES_DIR    = f"/home/{USER}/dissertation/repo/dissertation_v2/results/diagnostic_frames"
NUM_FRAMES    = 8

os.makedirs(FRAMES_DIR, exist_ok=True)

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def extract_and_save_frames(video_path, clip_label, view):
    """Extract 8 frames, save each as JPEG, return PIL Image list."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps_video if fps_video > 0 else 0

    if total == 0:
        cap.release()
        return [], 0, 0

    indices = [int(i * total / NUM_FRAMES) for i in range(NUM_FRAMES)]
    frames_pil = []
    saved_paths = []

    # Clean label for folder name
    safe_label = clip_label.replace(" ", "_").replace("/", "_")
    frame_dir = os.path.join(FRAMES_DIR, f"{safe_label}_{view}")
    os.makedirs(frame_dir, exist_ok=True)

    for frame_num, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Save original-resolution frame for visual inspection
            timestamp = idx / fps_video if fps_video > 0 else 0
            save_path = os.path.join(frame_dir, f"frame_{frame_num+1:02d}_t{timestamp:.1f}s.jpg")
            cv2.imwrite(save_path, frame)
            saved_paths.append(save_path)

            # Resize for model input
            frame_resized = cv2.resize(frame, (420, 360))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(frame_rgb))

    cap.release()
    print(f"    Saved {len(saved_paths)} frames to: {frame_dir}/")
    return frames_pil, total, duration


def ask(frames, question):
    """Ask model a question given PIL frames."""
    if not frames:
        return "ERROR: no frames"
    content = [{"type": "image"} for _ in frames]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out
    torch.cuda.empty_cache(); gc.collect()
    return clean


QUESTIONS = [
    ("What is the person doing?",           "Activity recognition"),
    ("What color is the climbing wall or holds? Answer briefly.", "Color perception"),
    ("Is the person wearing a shirt? What color?", "Detail perception"),
    ("How many people are visible in the video? Answer with a number.", "Counting"),
    ("Is this person a Novice or an Expert at this activity? Answer only: Novice or Expert", "Skill assessment"),
]

# Load benchmark and pick 1 clip per skill level
all_clips = json.load(open(BENCHMARK))
levels = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]

selected = {}
for clip in all_clips:
    gt = clip["ground_truth"]
    if gt in levels and gt not in selected:
        exo_path = os.path.join(DATA_DIR, clip["video_path_exo"])
        if os.path.exists(exo_path):
            selected[gt] = clip
    if len(selected) == 4:
        break

print(f"Selected clips: {list(selected.keys())}\n")

results = {}

for level in levels:
    if level not in selected:
        print(f"No clip found for {level}\n")
        continue

    clip = selected[level]
    exo_path = os.path.join(DATA_DIR, clip["video_path_exo"])

    print(f"{'='*65}")
    print(f"LEVEL: {level}")
    print(f"Clip:  {clip['take_folder']}")
    print(f"Video: {exo_path}")
    print(f"{'='*65}")

    # Extract frames and save them
    frames, total_frames, duration = extract_and_save_frames(exo_path, level, "exo")

    if not frames:
        print("  ERROR: Could not extract frames\n")
        continue

    print(f"  Video: {total_frames} frames total, {duration:.1f}s duration")
    print(f"  Sampled: frames at positions {[int(i*total_frames/NUM_FRAMES) for i in range(NUM_FRAMES)]}\n")

    clip_results = {}

    for question, label in QUESTIONS:
        answer = ask(frames, question)
        clip_results[label] = answer
        print(f"  Q ({label}):")
        print(f"    {question}")
        print(f"    → {answer}")
        print()

    results[level] = {
        "take_folder": clip["take_folder"],
        "answers": clip_results
    }

# Final summary
print("\n" + "="*65)
print("SUMMARY — What the model says about skill assessment")
print("="*65)
for level in levels:
    if level in results:
        skill_ans = results[level]["answers"].get("Skill assessment", "N/A")
        correct = ("novice" in skill_ans.lower() and level == "Novice") or \
                  ("expert" in skill_ans.lower() and level != "Novice")
        marker = "CORRECT" if correct else "WRONG"
        print(f"  {level:25} → {skill_ans:10} [{marker}]")

print(f"\nFrames saved to: {FRAMES_DIR}/")
print("Each subfolder contains 8 JPEG frames the model actually processed.")
print("File names include the timestamp (e.g. frame_01_t0.5s.jpg)")
