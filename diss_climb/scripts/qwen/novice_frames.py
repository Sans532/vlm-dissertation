"""
Saves frames for ONE Novice clip at n_frames=16 and n_frames=32.
No model inference — just frame extraction and saving.
Runs on CPU, no GPU needed.
"""
import json, os, cv2

USER      = os.environ.get("USER")
DATA_DIR  = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_binary.json"
OUT_DIR   = f"/home/{USER}/dissertation/repo/dissertation_v2/results/novice_frames"

os.makedirs(OUT_DIR, exist_ok=True)

# Pick first available Novice clip
with open(BENCHMARK) as f:
    clips = json.load(f)

novice_clip = None
for clip in clips:
    if clip["ground_truth"] == "Novice":
        exo_path = os.path.join(DATA_DIR, clip["video_path_exo"])
        if os.path.exists(exo_path):
            novice_clip = clip
            break

if not novice_clip:
    print("No Novice clip found with available video.")
    exit(1)

exo_path = os.path.join(DATA_DIR, novice_clip["video_path_exo"])
print(f"Clip: {novice_clip['take_folder']}")
print(f"Video: {exo_path}")

cap = cv2.VideoCapture(exo_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_v = cap.get(cv2.CAP_PROP_FPS)
duration = total / fps_v if fps_v > 0 else 0
print(f"Total frames: {total}, Duration: {duration:.1f}s, FPS: {fps_v:.1f}\n")
cap.release()

for n_frames in [16, 32]:
    save_dir = os.path.join(OUT_DIR, f"Novice_exo_n{n_frames}")
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(exo_path)
    indices = [int(i * total / n_frames) for i in range(n_frames)]

    saved = 0
    for frame_num, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            timestamp = idx / fps_v if fps_v > 0 else 0
            save_path = os.path.join(save_dir, f"frame_{frame_num+1:02d}_t{timestamp:.1f}s.jpg")
            cv2.imwrite(save_path, frame)
            saved += 1

    cap.release()
    print(f"n_frames={n_frames}: saved {saved} frames to {save_dir}/")
    print(f"  Sampled at positions: {indices[:8]}{'...' if n_frames > 8 else ''}")
    print()

print("Done. No GPU used.")
