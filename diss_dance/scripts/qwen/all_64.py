"""
Qwen2.5-VL-7B | Dance | 64 frames | All four prompts | Exo + Ego
Entire video (not trimmed). Extends the existing 8/16 frame-count series.
Resume-safe: reruns skip clips already completed in each results file.

Mirrors diss_climb/scripts/qwen/all_64.py — prompts, frame sampling and
generation settings are identical so the two activities stay comparable.
Note: the dance binary benchmark labels its expert class "Late Expert"
(climbing uses "Expert"); check_binary accepts both.
"""
import json, os, csv, gc, warnings, re, sys
import torch, cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import Counter

warnings.filterwarnings("ignore")

USER         = os.environ.get("USER")
BASE_DIR     = "/home/" + USER + "/dissertation/repo/diss_dance"
MODEL_PATH   = "/home/" + USER + "/dissertation/models/qwen25vl-7b"
DATA_DIR     = "/home/" + USER + "/dissertation/data/egoexo"
BINARY_BM    = BASE_DIR + "/benchmark/benchmark_binary_dance.json"
FOURCLASS_BM = BASE_DIR + "/benchmark/benchmark_100.json"
RESULTS_DIR  = BASE_DIR + "/results/qwen"
NUM_FRAMES   = 64
LABELS_4C    = ["Late Expert", "Intermediate Expert", "Early Expert", "Novice"]

os.makedirs(RESULTS_DIR, exist_ok=True)

PROMPTS = {
    "binary": {
        "text": "Is this person a Novice or an Expert at this activity? Answer only: Novice or Expert",
        "benchmark": BINARY_BM,
        "max_new_tokens": 50,
        "results_file": "qwen_dance_entire_n64_binary.csv",
    },
    "fourclass": {
        "text": "What is the skill level of the person in this video? Answer only one: Novice / Early Expert / Intermediate Expert / Late Expert",
        "benchmark": FOURCLASS_BM,
        "max_new_tokens": 50,
        "results_file": "qwen_dance_entire_n64_fourclass.csv",
    },
    "structured": {
        "text": (
            "Watch these frames carefully.\n"
            "Step 1: Describe the person's body position and technique in detail.\n"
            "Step 2: Identify any errors or imprecisions in their movement.\n"
            "Step 3: Classify skill level as exactly one of: "
            "Novice / Early Expert / Intermediate Expert / Late Expert.\n"
            "Format your answer as:\n"
            "Observations: ...\n"
            "Errors: ...\n"
            "Skill Level: ..."
        ),
        "benchmark": FOURCLASS_BM,
        "max_new_tokens": 300,
        "results_file": "qwen_dance_entire_n64_structured.csv",
    },
    "reasoning": {
        "text": (
            "You are an expert coach evaluating this person's technique.\n"
            "Focus on: body alignment, movement fluency, technical precision.\n"
            "What is their skill level? "
            "Novice / Early Expert / Intermediate Expert / Late Expert\n"
            "Explain your reasoning."
        ),
        "benchmark": FOURCLASS_BM,
        "max_new_tokens": 300,
        "results_file": "qwen_dance_entire_n64_reasoning.csv",
    },
}

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def get_frames(video_path, num_frames=64):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (420, 360)), cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def ask(video_path, question, max_new_tokens):
    frames = get_frames(video_path, NUM_FRAMES)
    if not frames:
        raise Exception("No frames")
    content = [{"type": "image"} for _ in frames]
    content.append({"type": "text", "text": "These are " + str(NUM_FRAMES) + " frames from a video of a person performing an activity. " + question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out, frames
    torch.cuda.empty_cache(); gc.collect()
    return clean


def check_binary(answer, gt):
    a = answer.lower()
    has_nov = "novice" in a
    has_exp = "expert" in a
    if has_nov and not has_exp:
        return gt.lower() == "novice"
    if has_exp and not has_nov:
        return gt.lower() in ["expert", "late expert"]
    pos_n = a.find("novice") if has_nov else 10**9
    pos_e = a.find("expert") if has_exp else 10**9
    if pos_n == pos_e:
        return False
    return (gt.lower() == "novice") == (pos_n < pos_e)


def extract_label_4c(answer, look_for_field=False):
    a = answer.lower()
    search_text = a
    if look_for_field:
        match = re.search(r"skill level:\s*(.+?)(?:\n|$)", a, re.IGNORECASE)
        if match:
            search_text = match.group(1).strip()
    for label in LABELS_4C:
        if label.lower() in search_text:
            return label
    for label in LABELS_4C:
        if label.lower() in a:
            return label
    return "Unknown"


def run_prompt(prompt_name, cfg):
    results_path = os.path.join(RESULTS_DIR, cfg["results_file"])
    benchmark = json.load(open(cfg["benchmark"]))

    completed_ids = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for row in csv.DictReader(f):
                completed_ids.add(row["clip_id"])
        print("[" + prompt_name + "] Found " + str(len(completed_ids)) + " already-completed clips. Resuming.")
    else:
        with open(results_path, "w", newline="") as f:
            if prompt_name == "binary":
                csv.writer(f).writerow(["clip_id", "take_folder", "ground_truth",
                                         "exo_answer", "exo_predicted", "exo_correct",
                                         "ego_answer", "ego_predicted", "ego_correct"])
            else:
                csv.writer(f).writerow(["clip_id", "take_folder", "ground_truth",
                                         "exo_full_answer", "exo_predicted", "exo_correct",
                                         "ego_full_answer", "ego_predicted", "ego_correct"])

    remaining = [item for item in benchmark if item["clip_id"] not in completed_ids]
    print("\n" + "=" * 60)
    print("[" + prompt_name + "] Total: " + str(len(benchmark)) + " | Done: " + str(len(completed_ids)) + " | Remaining: " + str(len(remaining)))
    print("=" * 60 + "\n")

    stats = {"exo": [0, 0], "ego": [0, 0]}
    exo_preds = Counter(); ego_preds = Counter()

    for i, item in enumerate(remaining):
        gt = item["ground_truth"]
        exo_path = os.path.join(DATA_DIR, item["video_path_exo"])
        ego_path = os.path.join(DATA_DIR, item["video_path_ego"])
        take_folder = item["take_folder"]
        row = [item["clip_id"], take_folder, gt]

        print("[" + prompt_name + " " + str(i+1) + "/" + str(len(remaining)) + "] " + take_folder + " (GT=" + gt + ")")

        for view, path in [("exo", exo_path), ("ego", ego_path)]:
            try:
                if not os.path.exists(path):
                    raise Exception("Video not found")
                ans = ask(path, cfg["text"], cfg["max_new_tokens"])
                if prompt_name == "binary":
                    ok = check_binary(ans, gt)
                    if "novice" in ans.lower() and "expert" not in ans.lower():
                        pred = "Novice"
                    elif "expert" in ans.lower():
                        pred = "Expert"
                    else:
                        pred = ans.strip()
                else:
                    pred = extract_label_4c(ans, look_for_field=(prompt_name == "structured"))
                    ok = pred.lower() == gt.lower()
            except Exception as e:
                print("  " + view + " ERROR: " + str(e))
                ans = "ERROR"; pred = "Unknown"; ok = False

            row.extend([ans, pred, ok])
            stats[view][1] += 1
            if ok: stats[view][0] += 1
            if view == "exo": exo_preds[pred] += 1
            else: ego_preds[pred] += 1
            print("  " + view + ": " + pred + " " + ("OK" if ok else "X"))

        with open(results_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    print("\n" + "=" * 60)
    print("[" + prompt_name + "] Batch complete. Processed " + str(len(remaining)) + " clips this run.")
    for v, (c, t) in stats.items():
        if t:
            print("  " + v + ": " + str(c) + "/" + str(t) + " = " + str(round(c/t*100, 1)) + "%")
    print("  Exo predictions: " + str(dict(exo_preds.most_common())))
    print("  Ego predictions: " + str(dict(ego_preds.most_common())))
    print("=" * 60)


if __name__ == "__main__":
    order = ["binary", "fourclass", "structured", "reasoning"]
    if len(sys.argv) > 1:
        order = [p for p in sys.argv[1:] if p in PROMPTS]
        if not order:
            print("Unknown prompt names given. Choose from: binary fourclass structured reasoning")
            sys.exit(1)

    for prompt_name in order:
        run_prompt(prompt_name, PROMPTS[prompt_name])

    print("\nAll requested prompts complete.")
