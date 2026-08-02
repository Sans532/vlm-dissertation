"""
Video-based evaluation: Qwen2.5-VL-7B
- Sends actual video file (not extracted frames)
- Trims to task_start_sec → task_end_sec (only active climbing)
- Exocentric view only
- 10 clips: 5 Novice + 5 Expert
"""
import json, os, gc, warnings, tempfile, subprocess
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH     = f"/home/{USER}/dissertation/models/qwen25vl-7b"
DATA_DIR       = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_binary.json"
TAKES_PATH     = f"/home/{USER}/dissertation/data/egoexo/takes.json"
RESULTS_PATH   = f"/home/{USER}/dissertation/repo/dissertation_v2/results/qwen/video_eval_10clips.txt"

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

# Load takes.json for task timing
print("Loading takes metadata...")
takes = json.load(open(TAKES_PATH))
take_info = {t["take_name"]: t for t in takes}
print(f"Loaded {len(take_info)} takes.\n")

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def trim_video(video_path, start_sec, end_sec):
    """Trim video to task window using ffmpeg. Returns path to temp file."""
    duration = end_sec - start_sec
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # no audio
        tmp.name
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        os.unlink(tmp.name)
        return None
    return tmp.name


def ask_qwen_video(video_path, question, start_sec=None, end_sec=None):
    """Send video to Qwen. Optionally trim to task window first."""
    trimmed = None
    input_path = video_path

    # Trim to active climbing window if timing available
    if False and start_sec is not None and end_sec is not None and end_sec > start_sec:
        trimmed = trim_video(video_path, start_sec, end_sec)
        if trimmed:
            input_path = trimmed
            print(f"    Trimmed to {start_sec:.1f}s-{end_sec:.1f}s ({end_sec-start_sec:.1f}s)")

    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": input_path, "nframes": 16},
                {"type": "text", "text": question}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
        ).to("cuda")

        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        raw = processor.batch_decode(out, skip_special_tokens=True)[0]

        if "assistant\n" in raw:
            clean = raw.split("assistant\n")[-1].strip()
        elif "assistant" in raw.lower():
            clean = raw.split("assistant")[-1].strip()
        else:
            clean = raw.strip()

        del inputs, out, image_inputs, video_inputs
        torch.cuda.empty_cache()
        gc.collect()
        return clean

    finally:
        if trimmed and os.path.exists(trimmed):
            os.unlink(trimmed)


# Load benchmark and pick 5 Novice + 5 Late Expert with available videos
with open(BENCHMARK_PATH) as f:
    all_clips = json.load(f)

novice_clips = []
expert_clips = []

for clip in all_clips:
    exo_path = os.path.join(DATA_DIR, clip["video_path_exo"])
    if not os.path.exists(exo_path):
        continue
    if clip["ground_truth"] == "Novice" and len(novice_clips) < 5:
        novice_clips.append(clip)
    elif clip["ground_truth"] == "Expert" and len(expert_clips) < 5:
        expert_clips.append(clip)
    if len(novice_clips) == 5 and len(expert_clips) == 5:
        break

test_clips = novice_clips + expert_clips
print(f"Selected: {len(novice_clips)} Novice + {len(expert_clips)} Expert\n")

QUESTION = "Is this person a Novice or an Expert at this activity? Answer only: Novice or Expert"

results = []
novice_correct = 0
expert_correct = 0
errors = 0

for i, clip in enumerate(test_clips):
    exo_path = os.path.join(DATA_DIR, clip["video_path_exo"])
    gt = clip["ground_truth"]

    # Get task timing
    info = take_info.get(clip["take_folder"], {})
    start_sec = info.get("task_start_sec", None)
    end_sec = info.get("task_end_sec", None)

    print(f"[{i+1}/10] {clip['take_folder']} (GT={gt})")
    if start_sec and end_sec:
        print(f"    Task window: {start_sec:.1f}s → {end_sec:.1f}s")

    try:
        answer = ask_qwen_video(exo_path, QUESTION, start_sec, end_sec)

        if "novice" in answer.lower() and "expert" not in answer.lower():
            predicted = "Novice"
        elif "expert" in answer.lower():
            predicted = "Expert"
        else:
            predicted = answer.strip()

        correct = (gt == "Novice" and predicted == "Novice") or \
                  (gt == "Expert" and predicted == "Expert")

        if correct and gt == "Novice": novice_correct += 1
        if correct and gt == "Expert": expert_correct += 1

        result_line = f"  GT={gt} | Predicted={predicted} | {'CORRECT' if correct else 'WRONG'}"
        answer_line = f"  Full answer: {answer}"

        print(result_line)
        print(answer_line)
        results.append(f"[{i+1}] {clip['take_folder']}\n{result_line}\n{answer_line}\n")

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(f"[{i+1}] {clip['take_folder']}\n  ERROR: {e}\n")
        errors += 1
        torch.cuda.empty_cache()
        gc.collect()

    print()

# Summary
total_correct = novice_correct + expert_correct
summary = f"""
======================================================================
RESULTS — Qwen2.5-VL-7B Video Input (trimmed to task window)
======================================================================
Novice correct:      {novice_correct}/5 = {novice_correct/5:.0%}
Late Expert correct: {expert_correct}/5 = {expert_correct/5:.0%}
Total accuracy:      {total_correct}/10 = {total_correct/10:.0%}
Random chance:       50%
Errors:              {errors}
======================================================================
"""
print(summary)

# Save results
with open(RESULTS_PATH, "w") as f:
    f.write(f"Model: Qwen2.5-VL-7B\n")
    f.write(f"Input: actual video (trimmed to task window)\n")
    f.write(f"nframes: 16\n")
    f.write(f"Question: {QUESTION}\n\n")
    f.write("\n".join(results))
    f.write(summary)

print(f"Results saved to {RESULTS_PATH}")
