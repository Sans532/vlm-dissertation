"""
Saves 8/16/32 frames from one representative Dance clip per skill level
(Novice, Early Expert, Intermediate Expert, Late Expert) for visual inspection.
No GPU needed - pure OpenCV frame extraction, run on head node.
Mirrors save_all_frames.py used for Rock Climbing.
"""
import json, os, random
import cv2
random.seed(7)

USER      = os.environ.get("USER")
DATA_DIR  = "/home/" + USER + "/data_dance/videos"
OUT_DIR   = "/home/" + USER + "/dissertation/repo/diss_dance/results/diagnostic_frames"
FRAME_COUNTS = [8, 16, 32]
LEVELS = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading annotations...")
train = json.load(open("/home/" + USER + "/dissertation/data/egoexo/annotations/proficiency_demonstrator_train.json"))["annotations"]
val = json.load(open("/home/" + USER + "/dissertation/data/egoexo/annotations/proficiency_demonstrator_val.json"))["annotations"]
all_clips = train + val

dance = [a for a in all_clips if a.get("scenario_name") == "Dance"]


def clip_is_valid(c):
    exo = c["video_paths"].get("exo1", "")
    ego = c["video_paths"].get("ego", "")
    if not exo or not ego:
        return False
    return os.path.exists(os.path.join(DATA_DIR, exo)) and os.path.exists(os.path.join(DATA_DIR, ego))


def extract_and_save(video_path, out_folder, num_frames):
    os.makedirs(out_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0 or fps == 0:
        cap.release()
        return []

    indices = [int(i * total / num_frames) for i in range(num_frames)]
    saved = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            timestamp = idx / fps
            fname = "frame_" + str(i + 1).zfill(2) + "_t" + str(round(timestamp, 1)) + "s.jpg"
            save_path = os.path.join(out_folder, fname)
            cv2.imwrite(save_path, frame)
            saved.append(fname)
    cap.release()
    return saved


print("Selecting one representative clip per level...\n")

for level in LEVELS:
    level_clips = [c for c in dance if c["proficiency_score"] == level]
    valid = [c for c in level_clips if clip_is_valid(c)]
    random.shuffle(valid)

    if not valid:
        print(level + ": NO VALID CLIPS FOUND -- skipping")
        continue

    clip = valid[0]
    take_folder = clip["video_paths"]["ego"].split("/")[1]
    exo_path = os.path.join(DATA_DIR, clip["video_paths"]["exo1"])

    print(level + " -> " + take_folder)

    level_folder_name = level.replace(" ", "_")

    for n in FRAME_COUNTS:
        out_folder = os.path.join(OUT_DIR, level_folder_name + "_exo_n" + str(n))
        saved = extract_and_save(exo_path, out_folder, n)
        print("  n=" + str(n) + ": saved " + str(len(saved)) + " frames to " + out_folder)

    print()

print("Done. Frames saved under: " + OUT_DIR)
print("\nFor each level you should have 3 folders (n=8, n=16, n=32) containing JPEGs.")
print("Use these for the qualitative frame-inspection figures (same as the climbing")
print("'one frame per skill level' and 'dead time' figures in the dissertation).")
