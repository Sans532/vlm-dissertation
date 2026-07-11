"""
Rebuild benchmark_basketball.json using only clips with videos that actually exist.
Old benchmark had 50 entries but 16 were missing videos.
This filters existing clips and tops up to 50 (25 Novice + 25 Late Expert) if possible.
"""
import json, os, random
random.seed(7)

USER     = os.environ.get("USER")
DATA_DIR = "/home/" + USER + "/dissertation/data/egoexo"
OLD_BENCHMARK = "/home/" + USER + "/dissertation/benchmark/benchmark_basketball.json"
NEW_BENCHMARK = "/home/" + USER + "/dissertation/benchmark/benchmark_basketball.json"

from collections import Counter

# Step 1: check which clips from the OLD benchmark still exist
old_data = json.load(open(OLD_BENCHMARK))
print("Old benchmark: " + str(len(old_data)) + " entries")

existing = []
missing = []
for d in old_data:
    exo_path = os.path.join(DATA_DIR, d["video_path_exo"])
    ego_path = os.path.join(DATA_DIR, d["video_path_ego"])
    if os.path.exists(exo_path) and os.path.exists(ego_path):
        existing.append(d)
    else:
        missing.append(d["take_folder"])

print("Existing (both views): " + str(len(existing)))
print("Missing: " + str(len(missing)))
for m in missing:
    print("  - " + m)

print("\nExisting breakdown:")
print(Counter(d["ground_truth"] for d in existing))

# Step 2: try to top up missing slots from the full annotation pool
existing_uids = set(d["clip_id"] for d in existing)

train = json.load(open("/home/" + USER + "/dissertation/data/egoexo/annotations/proficiency_demonstrator_train.json"))["annotations"]
val   = json.load(open("/home/" + USER + "/dissertation/data/egoexo/annotations/proficiency_demonstrator_val.json"))["annotations"]
all_clips = train + val

basketball = [a for a in all_clips if a.get("scenario_name") == "Basketball"]

needed = Counter(d["ground_truth"] for d in existing)
target_per_class = 25
to_fill = {
    "Novice": max(0, target_per_class - needed.get("Novice", 0)),
    "Late Expert": max(0, target_per_class - needed.get("Late Expert", 0)),
}
print("\nNeed to add: " + str(to_fill))

BINARY_PROMPT = "Is this person a Novice or an Expert at this activity? Answer only: Novice or Expert"

added = []
for level, count in to_fill.items():
    if count == 0:
        continue
    candidates = []
    for clip in basketball:
        if clip["proficiency_score"] == level and clip["take_uid"] not in existing_uids:
            take_folder = clip["video_paths"]["ego"].split("/")[1]
            exo_path = os.path.join(DATA_DIR, clip["video_paths"].get("exo1", ""))
            ego_path = os.path.join(DATA_DIR, clip["video_paths"].get("ego", ""))
            if os.path.exists(exo_path) and os.path.exists(ego_path):
                candidates.append(clip)

    random.shuffle(candidates)
    for c in candidates[:count]:
        take_folder = c["video_paths"]["ego"].split("/")[1]
        added.append({
            "clip_id": c["take_uid"],
            "take_folder": take_folder,
            "video_path_ego": c["video_paths"]["ego"],
            "video_path_exo": c["video_paths"].get("exo1", ""),
            "activity": "Basketball",
            "ground_truth": level,
            "question_binary": BINARY_PROMPT
        })
        existing_uids.add(c["take_uid"])
    print("  Added " + str(len(candidates[:count])) + "/" + str(count) + " for " + level)

# Step 3: combine and save
final = existing + added
random.shuffle(final)

print("\nFinal benchmark: " + str(len(final)) + " clips")
print(Counter(d["ground_truth"] for d in final))

with open(NEW_BENCHMARK, "w") as f:
    json.dump(final, f, indent=2)

print("\nSaved: " + NEW_BENCHMARK)
