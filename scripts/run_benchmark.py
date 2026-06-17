import json
import csv
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen2vl"
BENCHMARK_PATH = f"/home/{USER}/dissertation/benchmark/benchmark.json"
RESULTS_PATH = f"/home/{USER}/dissertation/results/qwen_results.csv"

# Load model
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.")

def ask_qwen(video_path, question):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path,
             "max_pixels": 360 * 420, "fps": 1.0},
            {"type": "text", "text": question}
        ]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs,
        videos=video_inputs, return_tensors="pt"
    ).to("cuda")
    out = model.generate(**inputs, max_new_tokens=200)
    return processor.batch_decode(out, skip_special_tokens=True)[0]

# Load benchmark
with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

print(f"Running benchmark on {len(benchmark)} clips...")

# Run evaluation
results = []
for i, item in enumerate(benchmark):
    try:
        answer_baseline = ask_qwen(item["video_path"], item["question_baseline"])
        answer_binary = ask_qwen(item["video_path"], item["question_binary"])
        answer_structured = ask_qwen(item["video_path"], item["question_structured"])

        results.append({
            "clip_id": item["clip_id"],
            "activity": item["activity"],
            "ground_truth_4class": item["ground_truth_4class"],
            "ground_truth_binary": item["ground_truth_binary"],
            "answer_baseline": answer_baseline,
            "answer_binary": answer_binary,
            "answer_structured": answer_structured
        })
        print(f"[{i+1}/{len(benchmark)}] Done: {item['clip_id']}")

    except Exception as e:
        print(f"[{i+1}/{len(benchmark)}] ERROR on {item['clip_id']}: {e}")

# Save results
with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {RESULTS_PATH}")
print(f"Total processed: {len(results)}/{len(benchmark)}")
