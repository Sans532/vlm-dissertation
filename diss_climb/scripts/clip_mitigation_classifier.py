"""
Mitigation test: frozen CLIP visual features -> lightweight classifier
Tests whether skill-relevant signal exists in general-purpose visual features
even though VLMs cannot articulate it in language.

4-class, 100 clips, 25 per level. 8 frames per clip, mean-pooled CLIP embeddings.
5-fold stratified cross-validation (no separate held-out test set needed - CV
means every clip gets evaluated exactly once, using a model trained on the rest).

No video-language model involved - CLIP's image encoder only. No prompt used at all.
"""
import json, os, warnings
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from collections import Counter

warnings.filterwarnings("ignore")

USER       = os.environ.get("USER")
DATA_DIR   = "/home/" + USER + "/dissertation/data/egoexo"
BENCHMARK  = "/home/" + USER + "/dissertation/repo/diss_climb/benchmark/benchmark_100_gemini.json"
RESULTS    = "/home/" + USER + "/dissertation/repo/diss_climb/results/mitigation/clip_classifier_results.txt"
NUM_FRAMES = 8
LABELS     = ["Novice", "Early Expert", "Intermediate Expert", "Late Expert"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

print("Loading CLIP ViT-L/14 ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("CLIP loaded on: " + device + "\n")


def get_frames(video_path, num_frames=8):
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
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def get_clip_features(frames):
    if not frames:
        return None
    inputs = clip_processor(images=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        image_features = outputs.pooler_output  # <-- the fix
    pooled = image_features.mean(dim=0)
    return pooled.cpu().numpy()

benchmark = json.load(open(BENCHMARK))
print("Clips: " + str(len(benchmark)))
print("Extracting frames + CLIP features (exo view)...\n")

X = []
y = []
skipped = []

for i, item in enumerate(benchmark):
    gt = item["ground_truth"]
    exo_path = os.path.join(DATA_DIR, item["video_path_exo"])

    if not os.path.exists(exo_path):
        print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + item["take_folder"] + " -- SKIP (video not found)")
        skipped.append(item["take_folder"])
        continue

    frames = get_frames(exo_path, NUM_FRAMES)
    if not frames:
        print("[" + str(i+1) + "/" + str(len(benchmark)) + "] " + item["take_folder"] + " -- SKIP (no frames extracted)")
        skipped.append(item["take_folder"])
        continue

    features = get_clip_features(frames)
    X.append(features)
    y.append(LABEL_TO_IDX[gt])

    if (i + 1) % 10 == 0:
        print("[" + str(i+1) + "/" + str(len(benchmark)) + "] processed")

X = np.array(X)
y = np.array(y)

print("\nFeature matrix shape: " + str(X.shape))
print("Labels shape: " + str(y.shape))
print("Skipped: " + str(len(skipped)))
print("Label distribution: " + str(Counter([LABELS[i] for i in y])))

print("\nRunning 5-fold stratified cross-validation...")

clf = LogisticRegression(max_iter=2000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

dummy = DummyClassifier(strategy="most_frequent")
dummy_scores = cross_val_score(dummy, X, y, cv=cv, scoring="accuracy")

random_clf = DummyClassifier(strategy="stratified", random_state=42)
random_scores = cross_val_score(random_clf, X, y, cv=cv, scoring="accuracy")

n = len(y)
random_chance = 1.0 / len(LABELS)

print("\n" + "=" * 60)
print("RESULTS -- CLIP features + Logistic Regression, 4-class")
print("=" * 60)
print("n = " + str(n) + " clips, " + str(NUM_FRAMES) + " frames each, exo view")
print("")
print("Logistic Regression (CLIP features):")
print("  Per-fold accuracy: " + str(np.round(scores, 3).tolist()))
print("  Mean: " + str(round(scores.mean(), 4)) + " +/- " + str(round(scores.std(), 4)))
print("")
print("Most-frequent-class baseline:")
print("  Mean: " + str(round(dummy_scores.mean(), 4)))
print("")
print("Stratified-random baseline:")
print("  Mean: " + str(round(random_scores.mean(), 4)))
print("")
print("Theoretical random chance (uniform 4-class): " + str(round(random_chance, 4)))
print("=" * 60)

mean_acc = scores.mean()
std_err = scores.std() / np.sqrt(len(scores))
ci_low = mean_acc - 1.96 * std_err
ci_high = mean_acc + 1.96 * std_err
print("\nApprox. 95% CI on mean CV accuracy: [" + str(round(ci_low, 4)) + ", " + str(round(ci_high, 4)) + "]")
print("(Normal approximation over only 5 fold-scores -- treat as indicative, not precise.)")

with open(RESULTS, "w") as f:
    f.write("CLIP feature classifier -- mitigation test\n")
    f.write("=" * 60 + "\n")
    f.write("n = " + str(n) + " clips\n")
    f.write("Frames per clip: " + str(NUM_FRAMES) + "\n")
    f.write("View: exo\n")
    f.write("Skipped clips: " + str(skipped) + "\n\n")
    f.write("Per-fold accuracy: " + str(np.round(scores, 4).tolist()) + "\n")
    f.write("Mean CV accuracy: " + str(round(mean_acc, 4)) + "\n")
    f.write("Std across folds: " + str(round(scores.std(), 4)) + "\n")
    f.write("Approx 95% CI: [" + str(round(ci_low, 4)) + ", " + str(round(ci_high, 4)) + "]\n")
    f.write("Most-frequent-class baseline: " + str(round(dummy_scores.mean(), 4)) + "\n")
    f.write("Stratified-random baseline: " + str(round(random_scores.mean(), 4)) + "\n")
    f.write("Theoretical uniform random chance: " + str(round(random_chance, 4)) + "\n")
    f.write("\nLabel distribution: " + str(dict(Counter([LABELS[i] for i in y]))) + "\n")

print("\nSaved: " + RESULTS)
