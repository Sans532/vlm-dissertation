import json, os, csv, gc, warnings
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

USER = os.environ.get("USER")
MODEL_PATH = f"/home/{USER}/dissertation/models/qwen25vl-3b"
DATA_DIR = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK_PATH = f"/home/{USER}/dissertation/benchmark/benchmark_available.json"
RESULTS_PATH = f"/home/{USER}/dissertation/results/first_eval.csv"

# Load model
print("Loading model...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.")


def ask_qwen(video_path, question):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": 0.5},
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

    out = model.generate(**inputs, max_new_tokens=300)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip()

    del inputs, out, image_inputs, video_inputs
    torch.cuda.empty_cache()
    gc.collect()

    return raw, clean


# Load benchmark
"""
with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)  # Change to json.load(f) for all 91 clips

print(f"Testing {len(benchmark)} clips...\n")
"""
import sys

batch_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
batch_size = 10

with open(BENCHMARK_PATH) as f:
    all_clips = json.load(f)

start = batch_num * batch_size
end = min(start + batch_size, len(all_clips))
benchmark = all_clips[start:end]

print(f"Batch {batch_num}: clips {start+1} to {end} of {len(all_clips)}")

if batch_num == 0:
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "take_folder", "activity", "ground_truth_binary", "ground_truth_4class", "answer_binary", "answer_structured", "correct_binary"])

# Run evaluation
results = []
for i, item in enumerate(benchmark):
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])

    # --- Binary question ---
    try:
        raw_binary, answer_binary = ask_qwen(exo_path, item["question_binary"])

        print(f"[{i+1}/{len(benchmark)}] {item['take_folder']}")
        print(f"  Ground truth: {item['ground_truth_binary']}")
        print(f"  Full response (binary): {raw_binary}")
        print(f"  Extracted answer: {answer_binary}")
        print(f"  Correct: {'✓' if item['ground_truth_binary'].lower() in answer_binary.lower() else '✗'}")
        print()

    except Exception as e:
        print(f"[{i+1}] ERROR on binary question: {e}\n")
        answer_binary = "ERROR"
        raw_binary = "ERROR"
        torch.cuda.empty_cache()
        gc.collect()

"""
    # --- Structured question ---
    try:
        raw_structured, answer_structured = ask_qwen(exo_path, item["question_structured"])

        print(f"  Full response (structured): {raw_structured}")
        print(f"  Extracted answer: {answer_structured}")
        print()

    except Exception as e:
        print(f"[{i+1}] ERROR on structured question: {e}\n")
        answer_structured = "ERROR"
        raw_structured = "ERROR"
        torch.cuda.empty_cache()
        gc.collect()
"""
    results.append({
        "clip_id": item["clip_id"],
        "take_folder": item["take_folder"],
        "activity": item["activity"],
        "ground_truth_binary": item["ground_truth_binary"],
        "ground_truth_4class": item["ground_truth_4class"],
        "answer_binary": answer_binary,
        "answer_structured": "SKIPPED",
        "correct_binary": item["ground_truth_binary"].lower() in answer_binary.lower() if answer_binary != "ERROR" else False,
    })

    torch.cuda.empty_cache()
    gc.collect()

# Save results
"""
if results:
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    valid = [r for r in results if r["answer_binary"] != "ERROR"]
    correct_binary = sum(1 for r in valid if r["correct_binary"])

    # Breakdown by class
    beginners = [r for r in valid if r["ground_truth_binary"] == "Beginner"]
    experts = [r for r in valid if r["ground_truth_binary"] == "Expert"]
    correct_beginners = sum(1 for r in beginners if r["correct_binary"])
    correct_experts = sum(1 for r in experts if r["correct_binary"])

    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total clips tested: {len(valid)}")
    print(f"Overall accuracy:   {correct_binary}/{len(valid)} = {correct_binary/len(valid):.1%}")
    print(f"")
    print(f"Beginner clips:     {correct_beginners}/{len(beginners)} = {correct_beginners/len(beginners):.1%}" if beginners else "No beginner clips")
    print(f"Expert clips:       {correct_experts}/{len(experts)} = {correct_experts/len(experts):.1%}" if experts else "No expert clips")
    print(f"{'='*50}")
    print(f"Results saved to {RESULTS_PATH}")
else:
    print("No results to save.")
"""

# Write CSV header first
with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["clip_id", "take_folder", "activity", "ground_truth_binary", "ground_truth_4class", "answer_binary", "answer_structured", "correct_binary"])

total_correct = 0
total_tested = 0
beginner_correct = 0
beginner_total = 0
expert_correct = 0
expert_total = 0

for i, item in enumerate(benchmark):
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])

    answer_binary = "ERROR"
    answer_structured = "ERROR"

    try:
        raw_binary, answer_binary = ask_qwen(exo_path, item["question_binary"])
        print(f"[{i+1}/{len(benchmark)}] {item['take_folder']}")
        print(f"  Ground truth: {item['ground_truth_binary']}")
        print(f"  Model answer: {answer_binary}")
        correct = item["ground_truth_binary"].lower() in answer_binary.lower()
        print(f"  Correct: {'✓' if correct else '✗'}")
        print()
    except Exception as e:
        print(f"[{i+1}] ERROR binary: {e}\n")
        correct = False
        torch.cuda.empty_cache()
        gc.collect()

    try:
        raw_structured, answer_structured = ask_qwen(exo_path, item["question_structured"])
        print(f"  Structured response: {answer_structured[:200]}")
        print()
    except Exception as e:
        print(f"[{i+1}] ERROR structured: {e}\n")
        torch.cuda.empty_cache()
        gc.collect()

    # Write to CSV immediately
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            item["clip_id"], item["take_folder"], item["activity"],
            item["ground_truth_binary"], item["ground_truth_4class"],
            answer_binary, answer_structured, correct
        ])

    # Track accuracy
    if answer_binary != "ERROR":
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

    torch.cuda.empty_cache()
    gc.collect()

# Print final summary
print(f"\n{'='*50}")
print(f"RESULTS SUMMARY")
print(f"{'='*50}")
print(f"Total clips tested: {total_tested}")
print(f"Overall accuracy:   {total_correct}/{total_tested} = {total_correct/total_tested:.1%}" if total_tested else "No clips tested")
print(f"Beginner clips:     {beginner_correct}/{beginner_total} = {beginner_correct/beginner_total:.1%}" if beginner_total else "No beginner clips")
print(f"Expert clips:       {expert_correct}/{expert_total} = {expert_correct/expert_total:.1%}" if expert_total else "No expert clips")
print(f"{'='*50}")
