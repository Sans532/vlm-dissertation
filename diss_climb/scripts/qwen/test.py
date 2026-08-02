"""
Quick test: 10 clips, actual video input (not frame extraction), 
5 Novice + 5 Expert, using qwen_vl_utils for video processing.
"""
import json, os, gc, warnings
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen25vl-7b"
DATA_DIR = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_binary.json"

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def ask_qwen_video(video_path, question):
    """Send actual video file, not extracted frames."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "nframes": 16},
            {"type": "text", "text": question}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    ).to("cuda")

    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
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


# Load benchmark and pick 5 novice + 5 expert with available videos
with open(BENCHMARK_PATH) as f:
    all_clips = json.load(f)[:5]

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
print(f"Testing {len(test_clips)} clips: {len(novice_clips)} Novice + {len(expert_clips)} Expert")
print(f"Using ACTUAL VIDEO input (not frame extraction)")
print(f"nframes=16\n")

QUESTION = "Is this person a Novice or a Expert at this activity? Answer only: Novice or Expert"

print(f"Prompt: {QUESTION}\n")
print("=" * 60)

novice_correct = 0
expert_correct = 0

for i, clip in enumerate(test_clips):
    exo_path = os.path.join(DATA_DIR, clip["video_path_exo"])
    gt = clip["ground_truth"]

    try:
        answer = ask_qwen_video(exo_path, QUESTION)
        
        if "novice" in answer.lower() and "expert" not in answer.lower():
            predicted = "Novice"
        elif "expert" in answer.lower():
            predicted = "Expert"
        elif "expert" in answer.lower():
            predicted = "Expert"
        else:
            predicted = answer

        correct = (gt == "Novice" and predicted == "Novice") or \
                  (gt == "Expert" and predicted == "Expert")

        if correct and gt == "Novice":
            novice_correct += 1
        elif correct and gt == "Expert":
            expert_correct += 1

        print(f"[{i+1}/10] {clip['take_folder']}")
        print(f"  GT: {gt}")
        print(f"  Raw answer: {answer}")
        print(f"  Predicted: {predicted}")
        print(f"  {'CORRECT' if correct else 'WRONG'}")
        print()

    except Exception as e:
        print(f"[{i+1}/10] {clip['take_folder']}")
        print(f"  GT: {gt}")
        print(f"  ERROR: {e}")
        print()
        torch.cuda.empty_cache()
        gc.collect()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Novice correct:      {novice_correct}/5")
print(f"Expert correct: {expert_correct}/5")
print(f"Total:               {novice_correct + expert_correct}/10")
print("=" * 60)
