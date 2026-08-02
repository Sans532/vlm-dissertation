"""
Saves frames for ONE clip per skill level at n_frames = 8, 16, 32.
No model inference — just frame extraction and saving.
Runs on CPU, no GPU needed. Run directly on head node.
"""
import json, os, cv2

USER      = os.environ.get("USER")
DATA_DIR  = f"/home/{USER}/dissertation/data/egoexo"
BENCHMARK = f"/home/{USER}/dissertation/repo/dissertation_v2/benchmark/benchmark_structured.json"
OUT_DIR   = f"/home/{USER}/dissertation/repo/dissertation_v2/results/all_frames"

os.makedirs(OUT_DIR, exist_ok=True)

LEVELS = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]

# Pick one available clip per level
with open(BENCHMARK) as f:
    clips = json.load(f)

selected = {}
for clip in clips:
    gt = clip["ground_truth"]
    if gt in LEVELS and gt not in selected:
        exo_path = os.path.join(DATA_DIR, clip["video_path_exo"])
        if os.path.exists(exo_path):
            selected[gt] = clip
    if len(selected) == 4:
        break

print(f"Found clips for: {list(selected.keys())}\n")

for level in LEVELS:
    if level not in selected:
        print(f"SKIP: no available clip for {level}\n")
        continue

    clip = selected[level]
    exo_path = os.path.join(DATA_DIR, clip["video_path_exo"])
    safe_label = level.replace(" ", "_")

    cap = cv2.VideoCapture(exo_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps_v if fps_v > 0 else 0
    cap.release()

    print(f"{'='*55}")
    print(f"Level:  {level}")
    print(f"Clip:   {clip['take_folder']}")
    print(f"Video:  {duration:.1f}s  |  {total} frames  |  {fps_v:.1f} fps")
    print(f"{'='*55}")

    for n_frames in [8, 16, 32]:
        save_dir = os.path.join(OUT_DIR, f"{safe_label}_exo_n{n_frames}")
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(exo_path)
        indices = [int(i * total / n_frames) for i in range(n_frames)]

        saved = 0
        for frame_num, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                timestamp = idx / fps_v if fps_v > 0 else 0
                save_path = os.path.join(
                    save_dir,
                    f"frame_{frame_num+1:02d}_t{timestamp:.1f}s.jpg"
                )
                cv2.imwrite(save_path, frame)
                saved += 1
        cap.release()

        print(f"  n={n_frames:2d}: saved {saved} frames → {save_dir}/")

    print()

print("Done. Output structure:")
print(f"  {OUT_DIR}/")
for level in LEVELS:
    safe = level.replace(" ", "_")
    for n in [8, 16, 32]:
        print(f"    {safe}_exo_n{n}/   ({n} JPEGs)")
