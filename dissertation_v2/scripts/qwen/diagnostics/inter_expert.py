"""
Diagnostic: Novice — 6 questions per frame at 8 frames
Tests whether Qwen2.5-VL-7B can spot and describe the person in each frame.
One answer per question per frame — not averaged across all frames.
"""
import os, gc, json, warnings
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

warnings.filterwarnings("ignore")

USER        = os.environ.get("USER")
MODEL_PATH  = f"/home/{USER}/dissertation/models/qwen25vl-7b"
FRAMES_DIR  = f"/home/{USER}/dissertation/repo/dissertation_v2/results/all_frames"
RESULTS_DIR = f"/home/{USER}/dissertation/repo/dissertation_v2/results/diagnostis_all_levels"
LEVEL_FOLDER = "Intermediate_Expert"
LEVEL_LABEL  = "Intermediate Expert"

os.makedirs(RESULTS_DIR, exist_ok=True)

QUESTIONS = [
    ("person_visible",  "Is there a person visible in this image? Answer only: yes or no"),
    ("person_location", "Where is the person in this image? Choose one: on the wall / on the ground / not visible"),
    ("person_action",   "What is the person doing in this image? Choose one: climbing the wall / standing and looking at the wall / walking / sitting / not visible"),
    ("body_position",   "Describe the person's body position in one sentence. Focus on hands, feet, and posture."),
    ("confidence",      "Does the person look confident and in control of their movement? Answer only: yes / no / cannot tell"),
    ("errors",          "Can you see any obvious errors in the person's technique or body position? Answer only: yes / no / cannot tell"),
]

print("Loading Qwen2.5-VL-7B ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")


def load_frames(folder_path):
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".jpg"))
    return [(fname, Image.open(os.path.join(folder_path, fname)).convert("RGB")) for fname in files]


def ask_model(single_frame_pil, question):
    """Ask one question about one frame."""
    content = [{"type": "image"}, {"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[single_frame_pil], return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = raw.split("assistant\n")[-1].strip() if "assistant\n" in raw else raw.strip()
    del inputs, out
    torch.cuda.empty_cache()
    gc.collect()
    return clean


all_results = {}
SEP = "=" * 65

for n_frames in [8]:
    folder_name = LEVEL_FOLDER + "_exo_n" + str(n_frames)
    folder_path = os.path.join(FRAMES_DIR, folder_name)

    if not os.path.exists(folder_path):
        print("SKIP: " + folder_path + " not found")
        continue

    frame_data  = load_frames(folder_path)
    frame_names = [name for name, _ in frame_data]

    print(SEP)
    print("LEVEL: " + LEVEL_LABEL + "  |  n_frames=" + str(n_frames))
    print("Total frames: " + str(len(frame_data)))
    print(SEP)

    result = {
        "level": LEVEL_LABEL,
        "n_frames": n_frames,
        "frame_files": frame_names,
        "frames": {}
    }

    for i, (fname, img) in enumerate(frame_data):
        print("\n  --- Frame " + str(i+1) + "/" + str(len(frame_data)) + ": " + fname + " ---")
        frame_result = {}

        for q_key, q_text in QUESTIONS:
            try:
                answer = ask_model(img, q_text)
            except Exception as e:
                answer = "ERROR: " + str(e)
                torch.cuda.empty_cache()
                gc.collect()

            frame_result[q_key] = {"question": q_text, "answer": answer}
            print("    [" + q_key + "]: " + answer)

        result["frames"][fname] = frame_result

    all_results["n" + str(n_frames)] = result
    print()

# Save JSON
json_path = os.path.join(RESULTS_DIR, LEVEL_FOLDER + "_diagnostic.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

# Save clean text report
txt_path = os.path.join(RESULTS_DIR, LEVEL_FOLDER + "_diagnostic.txt")
with open(txt_path, "w") as f:
    f.write("DIAGNOSTIC REPORT — " + LEVEL_LABEL + "\n")
    f.write(SEP + "\n\n")
    for n_key, res in all_results.items():
        f.write("--- " + str(res["n_frames"]) + " FRAMES ---\n\n")
        for fname, frame_res in res["frames"].items():
            f.write("  Frame: " + fname + "\n")
            for q_key, qa in frame_res.items():
                f.write("    [" + q_key + "]: " + qa["answer"] + "\n")
            f.write("\n")
        f.write("\n")

print("Saved: " + json_path)
print("Saved: " + txt_path)
print("Done.")
