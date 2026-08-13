# Thesis figures — VLM proficiency estimation (climbing + dance)

All figures are generated from the raw result CSVs in `diss_climb/results/{qwen,videollava}`
and `diss_dance/results/{qwen,videollava}` after **re-parsing every model answer from the
raw text** (the `*_predicted` / `*_correct` columns stored in the CSVs are unreliable —
see `fig08`). Each figure exists as `.png` (300 dpi, for Word/preview) and `.pdf` (vector,
for LaTeX).

## Reproducing

```bash
.venv-stats/bin/python figures/rescore.py      # re-parses all answers -> master_rows.csv, master_summary.csv
.venv-stats/bin/python figures/make_figures.py # renders all figures
```

- `master_rows.csv` — one row per (file, clip, view) with the re-parsed prediction.
- `master_summary.csv` — accuracy per condition (activity × model × entire/trimmed × frames × prompt × view).
  `acc` = correct / valid answers; `invalid_rate` = ERROR / empty / unclassifiable share.
- `parse_disagreements.txt` — audit trail: every case where the re-parse differs from the stored prediction.

**Scope: 8 and 16 frames only.** The 32-frame binary runs
(`diss_climb/results/{qwen,videollava}/binary_n32.csv`) are excluded from the manifest in
`rescore.py` and therefore appear in no table or figure here.

## Figures and suggested captions

> **On the two chance baselines.** The binary prompt has 2 labels (chance = 50%) and the other
> three prompts have 4 labels (chance = 25%), so their raw accuracies are *never* directly
> comparable — 50% on the binary task is exactly as uninformative as 25% on the four-class task.
> Two figures handle this: `fig01` subtracts each task's own chance so everything shares one zero
> line (**use this as the headline**), and `fig01b` shows raw accuracies with the two task types in
> separate panels. Elsewhere (`fig03`, `fig08`) binary rows are labelled and both baselines drawn.

| File | Message | Suggested caption |
|---|---|---|
| `fig01_accuracy_above_chance` | Everything sits at chance, on one scale | Accuracy of every experimental run expressed **relative to its own chance baseline** (50% for the 2-way binary prompt, 25% for the 4-way prompts), so both task types share a single zero line. Each dot is one run (frame count × entire/trimmed × exo/ego). No condition departs meaningfully from chance. |
| `fig01b_accuracy_by_task_type` | Same data, raw accuracy, split | Raw accuracy with the 2-way and 4-way tasks in separate panels, each against its own chance line. Presented separately because the two task types have different baselines and cannot be compared directly. |
| `fig02_frame_count_effect` | More frames ≠ better; VL breaks | Effect of sampled frame count. Doubling frames from 8 to 16 leaves four-class accuracy unchanged for Qwen2.5-VL; Video-LLaVA's output degenerates instead (right: valid-output rate on climbing falls from 92% at 8 frames to 27% at 16). |
| `fig03_entire_vs_trimmed` | Trimming doesn't help | Accuracy on the entire video vs the clip trimmed to the annotated task segment, pooled over frame counts and viewpoints. Trimming produces no systematic improvement in any model/activity/prompt combination. |
| `fig04_viewpoint_asymmetry` | Views agree on accuracy, differ in bias | Camera viewpoint effects. Left: exocentric vs egocentric accuracy per run (diagonal = parity). Right: share of "Novice" predictions by viewpoint — egocentric input pushes Qwen2.5-VL's Novice-default from 75%→85% (climbing) and 36%→90% (dance). |
| `fig05_label_collapse` | The core failure mode | Predicted-label distributions vs the balanced ground truth. Four-class predictions collapse onto one or two labels (Qwen2.5-VL virtually never answers "Early" or "Late Expert"); under binary prompts Qwen2.5-VL answers "Novice" on 100% of clips, making its 50% accuracy pure base rate. |
| `fig06_confusion_matrices` | Predictions independent of GT | Row-normalised confusion matrices (four-way tasks pooled, valid answers only). Rows are nearly identical: the predicted distribution barely changes with the true skill level, i.e. the models do not extract proficiency-relevant signal. |
| `fig07_output_validity` | Data/robustness caveats | Share of unusable responses (runtime errors, empty output, no extractable class). Climbing carries a ~9% baseline error rate from 9 takes missing on disk; Video-LLaVA additionally fails catastrophically at 16 frames on climbing (73% unusable). |
| `fig08_rescoring_impact` | Methodology: why re-parse | Recorded accuracy (original keyword-matching harness) vs accuracy after re-parsing the full answer text with negation/aspiration handling. 175/7,812 stored labels changed; per-run accuracy shifts up to 8 points (largest: Video-LLaVA climbing reasoning runs). |
| `fig09_response_stereotypy` | Answers are boilerplate | Stereotypy of "reasoning" responses: share of responses in each condition that begin with the single most frequent opening line. Qwen2.5-VL gives a near-identical answer to 99–100% of egocentric clips, confirming predictions are driven by a prior, not the video. |

## Key numbers (re-scored)

- Binary: Qwen2.5-VL = "Novice" on **100%** of clips (both activities) → accuracy pinned at 50%.
  Video-LLaVA: 89.5% "Novice" on climbing, 100% on dance.
- Four-class family (fourclass/structured/reasoning), pooled valid answers:
  Qwen climbing 80% Novice / 18% Intermediate / ~0% Early+Late; dance 63/36.
  Video-LLaVA spreads slightly more but is equally uninformative (near-identical confusion rows).
- Best single runs are isolated artefacts, e.g. Video-LLaVA climbing entire n16 binary exo (79%)
  coexists with 33% on ego for the same clips.
- Qwen dance structured exo is the only family consistently above chance (32–41% vs 25%), driven by
  its Novice bias aligning with the "Skill Level: Novice" template rather than genuine discrimination.

## Caveats encoded in the pipeline

- Runs with fewer than 20 valid answers are excluded from accuracy plots (they still appear in
  `master_summary.csv` and fig07): VL climbing n16 structured/reasoning, VL trimmed n16 fourclass.
- `diss_climb/results/qwen/trimmed/fourclass_trimmed.csv` was written by both the n8 and n16 scripts
  to the same path; its frame count is ambiguous (`frames` = blank) and it is excluded from frame plots.
- `vl_climbing_entire_n8_reasoning_fixed.csv` and non-`_fixed`, and `..._reasoning_patched.csv`, contain
  the same raw answers (only stored labels differ); the raw text is what gets re-parsed here.
- Ad-hoc files (`3class_structured.csv`, `structured_eval.csv`, `binary_basketball.csv`,
  `bin_struct_n16.csv`, `test_exo3.csv`, `binary_n8_qwen_{ego,exo}.csv`, `reasoning_n8_test03*.csv`)
  are not part of the canonical grid and are excluded.
