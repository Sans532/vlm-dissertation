# Thesis figures — video-native and adaptive-sampling models

Companion to `figures-frame_count/`. That set covers the models fed a **fixed number of
sampled frames** (Video-LLaVA, Qwen2.5-VL at n = 8 and 16). This set covers the models that
consume the **video itself or choose their own frame budget**:

| Source folder | Model | How video reaches the model |
|---|---|---|
| `diss_climb/results/gemini`, `diss_dance/results/gemini` | Gemini | native video upload, whole clip |
| `diss_climb/results/qwen3vl`, `diss_dance/results/qwen3vl` | Qwen3-VL | adaptive frame sampling (4–263 frames, chosen per clip) |

Every answer is re-parsed from the raw response text with the **same parser** as the
frame-sampling set (`figures-frame_count/rescore.py` is imported directly), so numbers from
the two folders are directly comparable. 171 of 2,765 stored labels changed on re-parse.

## Reproducing

```bash
.venv-stats/bin/python figures-video/rescore_video.py       # -> master_rows.csv, master_summary.csv
.venv-stats/bin/python figures-video/make_figures_video.py  # -> vfig01 … vfig10 (PNG + PDF)
```

`make_figures_video.py` also reads `figures-frame_count/master_rows.csv` for the
cross-generation figure; if that folder is renamed, update `FRAME_DIR` at the top of both scripts.

> **On the two chance baselines.** The binary prompt has 2 labels (chance = 50%) and the other
> three prompts have 4 labels (chance = 25%), so their raw accuracies are never directly
> comparable. `vfig01` subtracts each task's own chance so everything shares one zero line
> (**headline figure**); `vfig01b` shows raw accuracy with the task types in separate panels.

## Figures and suggested captions

| File | Message | Suggested caption |
|---|---|---|
| `vfig01_accuracy_above_chance` | Only one condition clears chance | Accuracy of every run relative to its own chance baseline (50% binary, 25% four-class). Each dot is one run (viewpoint × prompt). Only Gemini's binary climbing runs sit clearly above zero; every four-class run is within a few points of chance. |
| `vfig01b_accuracy_by_task_type` | Same data, raw, split | Raw accuracy with the 2-way and 4-way tasks in separate panels, each against its own chance line, marker shape distinguishing the three four-class prompts. |
| `vfig02_binary_discrimination` | Signal vs one-label habit | Per-class recall on the binary task. Because the test set is class-balanced, accuracy is the midpoint of the two recalls and every point on the dashed line scores 50%. Gemini reaches (84, 68) on egocentric climbing — genuine discrimination — while Qwen3-VL sits in the bottom-right corner, answering "Novice" for ~96–100% of clips. |
| `vfig03_adaptive_frame_count` | More frames buys nothing | Qwen3-VL chooses its own frame budget (4–263 frames, median 26 for climbing and 64 for dance). Accuracy is flat across frame-count quartiles, while unusable responses rise with frame count (0.5%→6.6% on climbing) as long answers truncate before stating a verdict. |
| `vfig04_label_collapse` | Collapse target moved | Predicted-label distributions against the balanced ground truth, with the frame-sampled models included for reference. Label collapse survives the move to video-native models but changes target: the earlier generation defaulted to "Novice" (71% for Qwen2.5-VL), while Gemini assigns "Intermediate Expert" to 79% of climbing clips. Right: on the binary prompt Gemini splits its answers, Qwen3-VL does not. |
| `vfig05_confusion_matrices` | Predictions ignore ground truth | Row-normalised confusion matrices (four-way prompts pooled). Gemini labels 86–89% of Intermediate and Late Expert climbers "Intermediate Expert" but also 67% of true novices; rows stay near-identical, so the predicted label barely responds to true skill. |
| `vfig06_viewpoint_asymmetry` | Viewpoint matters for Gemini | Exocentric vs egocentric accuracy per run, and mean balanced accuracy by viewpoint. Egocentric video helps Gemini on climbing (60%→76% binary) and hurts it on dance (60%→52%); Qwen3-VL is insensitive to viewpoint. |
| `vfig07_generation_comparison` | The money figure | Balanced accuracy of all four models across both figure sets. On the 2-way climbing task Gemini reaches 68% while the other three sit at 50–53%; on the 4-way tasks all four models are within 4 points of the 25% baseline. Video-native input helps only the coarsest distinction. |
| `vfig08_output_validity_coverage` | Reliability and caveats | Left: unusable responses by prompt — the reasoning prompt costs Qwen3-VL 18% of climbing responses (truncation), while Gemini stays ≤1% everywhere. Right: experiment coverage; the Qwen3-VL native-video run never completed, leaving three of four prompt files empty and only 12 of 200 clip-views in the binary file. |
| `vfig09_answer_style` | Answers are genuinely per-clip | Answer length and opening-line diversity for the long-form prompts. Both models write 1,000–2,500 character analyses, and at most 6–13% (Gemini) share an opening line — a sharp contrast with the frame-sampled set, where Qwen2.5-VL opened 99–100% of egocentric reasoning answers with an identical sentence. The verbosity is real, but per `vfig05` it is not diagnostic. |
| `vfig10_rescoring_impact` | Methodology | Originally recorded accuracy vs accuracy after re-parsing the answer text. Shifts stay within 3 points here (largest: reasoning-prompt egocentric runs), much smaller than in the frame-sampled set. |

## Key numbers (re-scored)

- **Gemini binary climbing is the only genuine signal in either figure set**: 76% egocentric
  (recall 84% Novice / 68% Expert), 60% exocentric. Gemini binary dance ego, by contrast, is
  52% with Expert recall of just 4% — accuracy near chance produced by answering "Novice" 98% of the time.
- **Qwen3-VL never escapes the one-label habit on the binary task**: 92–100% "Novice", Expert
  recall 0–8%, accuracy pinned at 50–54%.
- **Four-way tasks: no model is meaningfully above 25%** — Gemini 29% (climbing) / 25% (dance),
  Qwen3-VL 29% / 27%. Per-class recall is extremely uneven (Gemini climbing: Intermediate 86%,
  Early 0%, Late 1%).
- **Collapse target shifted between generations**: frame-sampled models defaulted to "Novice"
  (Qwen2.5-VL 71%, Video-LLaVA 51%); video-native models default to "Intermediate Expert"
  (Gemini climbing 79%, dance 62%; Qwen3-VL ~47%).
- **Adaptive frame count is not a lever**: accuracy flat across quartiles from ~10 to ~165 frames;
  the only monotone effect is a rise in truncated, unusable answers.

## Caveats encoded in the pipeline

- The **Qwen3-VL native-video run never completed**: `qwen3vl_climbing_native_{fourclass,structured,reasoning}.csv`
  are empty and `..._native_binary.csv` holds 12 of 50 clips. It is loaded so coverage can be
  reported (`vfig08`) but excluded from accuracy figures by the `MIN_VALID = 20` gate.
  There is no native-video Qwen3-VL run for dance at all.
- `gemini_climbing_entire_binary.csv` contains one retried clip (a first attempt that errored);
  the loader keeps the last attempt per `clip_id`.
- Gemini stores one file per viewpoint (`*_ego.csv` for egocentric); Qwen3-VL stores both
  viewpoints in one file. Both are normalised to one row per clip-view.
- `gemini_dance_entire_reasoning.csv.bak` is a backup and is not loaded.
- Qwen3-VL's `exo_nframes`/`ego_nframes` = 0 marks the clips whose source video is missing on
  disk (the same 9 climbing takes that affect the frame-sampled set); these are counted as
  unusable, not as wrong answers.
