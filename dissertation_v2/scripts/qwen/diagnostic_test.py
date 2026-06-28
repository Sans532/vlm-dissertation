"""
DIAGNOSTIC: Proves the model actually sees the video frames.
Tests the same clip with control questions that have known visual answers.
If the model answers these correctly, the pipeline works and the skill-bias is real.
"""
import json, os, gc, warnings
import torch
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen25vl-7b"
DATA_DIR = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_binary.json"

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
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


def ask(video_path, question):
    frames = get_video_frames(video_path, num_frames=8)
    if not frames:
        return "NO_FRAMES"
    content = [{"type": "image"} for _ in frames]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out, frames
    torch.cuda.empty_cache(); gc.collect()
    return clean


# Pick 3 clips
with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

test_clips = []
for clip in benchmark:
    p = os.path.join(DATA_DIR, clip["video_path_exo"])
    if os.path.exists(p):
        test_clips.append(clip)
    if len(test_clips) == 3:
        break

# Control questions — these have OBJECTIVE answers we can verify
CONTROL_QUESTIONS = [
    "What is the person doing in this video? Answer in one sentence.",
    "What color is the climbing wall or holds? Answer briefly.",
    "Is the person wearing a shirt? What color?",
    "How many people are visible in the video?",
    "Is this person a novice or an expert at this activity? Answer only: Novice or Expert",
]

for clip in test_clips:
    path = os.path.join(DATA_DIR, clip["video_path_exo"])
    print("=" * 70)
    print(f"CLIP: {clip['take_folder']} (GT skill = {clip['ground_truth']})")
    print("=" * 70)
    for q in CONTROL_QUESTIONS:
        answer = ask(path, q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")
    print()
