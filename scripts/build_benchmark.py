import json
import os

# This script builds benchmark.json from EgoExo4D annotations
# Update ANNOTATIONS_PATH when dataset arrives

USER = os.environ.get("USER")
ANNOTATIONS_PATH = f"/home/{USER}/dissertation/data/proficiency_annotations.json"

SELECTED_ACTIVITIES = ["Bouldering"]  # start with one activity

# Templates from prompts file
from sys import path
path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'prompts'))
from prompt_templates import PROMPTS

def build_benchmark(annotations_path, output_path):
    with open(annotations_path) as f:
        annotations = json.load(f)

    benchmark = []

    for take in annotations:
        # Filter by activity
        if take.get("scenario") not in SELECTED_ACTIVITIES:
            continue

        # Filter for clear skill separation
        label = take.get("proficiency_label", "")
        if label not in ["Novice", "Late Expert"]:
            continue

        binary_gt = "Beginner" if label == "Novice" else "Expert"

        benchmark.append({
            "clip_id": take["take_uid"],
            "video_path": f"videos/{take['take_uid']}.mp4",
            "activity": take["scenario"],
            "ground_truth_4class": label,
            "ground_truth_binary": binary_gt,
            "question_baseline": PROMPTS["baseline"],
            "question_binary": PROMPTS["binary"],
            "question_structured": PROMPTS["structured"],
            "question_reasoning": PROMPTS["reasoning"]
        })

    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"Benchmark created: {len(benchmark)} clips")
    return benchmark

if __name__ == "__main__":
    output = f"/home/{USER}/dissertation/benchmark/benchmark.json"
    build_benchmark(ANNOTATIONS_PATH, output)
