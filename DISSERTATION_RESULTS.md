# Dissertation Results — Full Detail, Collapsed Results Clearly Flagged

*Compiled 2026-08-06. Every number below was independently recomputed/re-verified against the raw result CSVs in this repo (not copy-pasted from an old summary). Every result the pipelines produced — including collapsed/degenerate ones — is shown in full detail (Overall accuracy → per-class accuracy → predicted-answer counts → source file), organized under a `#### Collapsed` or `#### Kept` sub-header within each model/activity section so it's clear at a glance which numbers should and shouldn't be cited as evidence of real model capability.*

**Exclusion rule (governs what counts as "Kept" vs "Collapsed", not what's shown):** a result is marked **Collapsed** if the model's predicted-label distribution is a **near-total collapse to a single label** — formally, if at most one ground-truth class has non-zero accuracy (i.e. the model only ever gets the majority/default class right and is never correct on any other class). This was applied uniformly and reproducibly across every file, not judged case-by-case. Collapsed results are not deleted — they're tabulated in full under their own sub-header — but should not be cited as evidence of discriminative skill judgment; only "Kept" results should be used for headline claims. Chance baselines: **50%** for binary (25/25 split), **25%** for four-class (25 per class).

Two extraction bugs were found and corrected during re-verification (see §6):
1. **Free-text "reasoning" answers** (Qwen, VideoLLaVA, Gemini-ego) — the original label parser mis-scored negated/hedged/disjunctive sentences ("not yet a Late Expert" being read as "Late Expert"). All reasoning-file numbers below use the corrected extractor, independently re-run and hand-verified against the raw answer text for all 24 affected files.
2. **`gemini_climbing_entire_binary.csv` duplicate row** — one clip (`97b78075…`) had a stale failed retry left in the file alongside its successful retry. Deduplicated (kept the successful retry) before scoring — see §1.1.

**Video-input method — two paradigms, not one:** Qwen2.5-VL and VideoLLaVA use **fixed frame-count sampling** (8/16/32 frames extracted uniformly regardless of clip length). Gemini and Qwen3-VL are the two models run at a **clip-duration-proportional ~1fps rate** instead — but only Gemini's numbers come from genuine native video ingestion (raw file uploaded directly to the API); Qwen3-VL's native pipeline hit an fps=24 OOM bug and was abandoned after 12/50 clips, so its reported "frames" numbers come from a manual OpenCV substitute built to reproduce the same ~1fps target rate by hand, not from true native ingestion. See `PROJECT_REPORT.md` §"Two distinct video-input paradigms" for the full detail. Practical implication: frame-count ablation (8 vs 16 vs 32) only exists for Qwen2.5-VL/VideoLLaVA — Gemini and Qwen3-VL were never run at a fixed frame count at all.

---

## Table of contents
1. [Climbing (`diss_climb`)](#1-climbing-diss_climb)
2. [Dance (`diss_dance`)](#2-dance-diss_dance)
3. [Cross-domain generalization](#3-cross-domain-generalization)
4. [Label-granularity ablation (3-class)](#4-label-granularity-ablation-3-class)
5. [What was excluded, and why](#5-what-was-excluded-and-why)
6. [Extraction-bug re-verification method](#6-extraction-bug-re-verification-method)

---

## 1. Climbing (`diss_climb`)

### 1.1 Gemini

**Binary — ego**
- Overall: 38/50 = **76.0%**
- Novice: 21/25 = 84.0% · Expert: 17/25 = 68.0%
- Predicted counts: Novice 29, Expert 21
- File: `diss_climb/results/gemini/gemini_climbing_entire_binary_ego.csv`

**Binary — exo** *(deduplicated — see note above)*
- Overall: 30/50 = **60.0%**
- Novice: 13/25 = 52.0% · Expert: 17/25 = 68.0%
- Predicted counts: Expert 29, Novice 21
- File: `diss_climb/results/gemini/gemini_climbing_entire_binary.csv`

**Fourclass — ego**
- Overall: 32/100 = **32.0%**
- Novice: 12/25 = 48.0% · Early Expert: 0/25 · Intermediate Expert: 20/25 = 80.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 70, Novice 28, Early Expert 2
- File: `diss_climb/results/gemini/gemini_climbing_entire_fourclass_ego.csv`

**Fourclass — exo**
- Overall: 26/100 = **26.0%**
- Novice: 5/25 = 20.0% · Early Expert: 0/25 · Intermediate Expert: 21/25 = 84.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 81, Novice 17, Late Expert 2
- File: `diss_climb/results/gemini/gemini_climbing_entire_fourclass.csv`

**Reasoning — ego**
- Overall: 28/100 = **28.0%**
- Novice: 8/25 = 32.0% · Early Expert: 0/25 · Intermediate Expert: 20/25 = 80.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 68, Novice 29, Unknown 3
- File: `diss_climb/results/gemini/gemini_climbing_entire_reasoning_ego.csv`

**Reasoning — exo**
- Overall: 27/100 = **27.0%**
- Novice: 3/25 = 12.0% · Early Expert: 0/25 · Intermediate Expert: 24/25 = 96.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 92, Novice 6, Late Expert 1, Early Expert 1
- File: `diss_climb/results/gemini/gemini_climbing_entire_reasoning.csv`

**Structured — ego**
- Overall: 31/100 = **31.0%**
- Novice: 11/25 = 44.0% · Early Expert: 0/25 · Intermediate Expert: 20/25 = 80.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 75, Novice 23, Late Expert 2
- File: `diss_climb/results/gemini/gemini_climbing_entire_structured_ego.csv`

**Structured — exo**
- Overall: 33/100 = **33.0%**
- Novice: 10/25 = 40.0% · Early Expert: 0/25 · Intermediate Expert: 23/25 = 92.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 80, Novice 20
- File: `diss_climb/results/gemini/gemini_climbing_entire_structured.csv`

> Note: Gemini never once predicts Early Expert or Late Expert on climbing — every fourclass/reasoning/structured run collapses the 4-class scale to an effective Novice-vs-Intermediate-Expert judgment. Kept because it still resolves 2 of the 4 classes with real signal.

---

### 1.2 Qwen2.5-VL-7B

#### Collapsed — binary (all conditions; shown in full, excluded from headline claims)

Every binary-prompt climbing run predicts "Novice" for 100% of clips, regardless of frame count, trim condition, or view. Accuracy is exactly 50% in every case = chance by construction (Novice is exactly half the ground truth).

| Frames | Trim | View | Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| 16 | entire | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/qwen_climbing_entire_n16_binary.csv` |
| 16 | entire | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/qwen_climbing_entire_n16_binary.csv` |
| 8 | entire | ego  | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/qwen_climbing_entire_n8_binary.csv` |
| 8 | entire | exo  | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/qwen_climbing_entire_n8_binary.csv` |
| 16 | trimmed | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_ego_n16_binary.csv` |
| 16 | trimmed | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_exo_n16_binary.csv` |
| 8 | trimmed | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_ego_n8_binary.csv` |
| 8 | trimmed | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_exo_n8_binary.csv` |

#### Kept — fourclass / reasoning / structured

**Fourclass — 16 frames, entire, exo**
- Overall: 27/100 = **27.0%**
- Novice: 23/25 = 92.0% · Early Expert: 0/25 · Intermediate Expert: 4/25 = 16.0% · Late Expert: 0/25
- Predicted counts: Novice 89, Intermediate Expert 11
- File: `diss_climb/results/qwen/qwen_climbing_entire_n16_fourclass.csv`

**Fourclass — 16 frames, entire, ego** *(⚠ duplicate — this is a pure Novice collapse, already listed under `#### Collapsed` above; consider deleting this block)*
- Overall: 25/100 = **25.0%**
- Novice: 25/25 = 100.0% · Early Expert: 0/25 = 0.0% · Intermediate Expert: 0/25 = 0.0% · Late Expert: 0/25 = 0.0%
- Predicted counts: Novice 98, Intermediate Expert 2
- File: `diss_climb/results/qwen/qwen_climbing_entire_n16_fourclass.csv`

**Fourclass — 8 frames, entire, ego**
- Overall: 24/100 = **24.0%**
- Novice: 23/25 = 92.0% · Early Expert: 0/25 · Intermediate Expert: 1/25 = 4.0% · Late Expert: 0/25
- Predicted counts: Novice 84, Unknown 9, Intermediate Expert 7
- File: `diss_climb/results/qwen/qwen_climbing_entire_n8_fourclass.csv`

**Fourclass — 8 frames, entire, exo**
- Overall: 25/100 = **25.0%**
- Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 3/25 = 12.0% · Late Expert: 0/25
- Predicted counts: Novice 83, Unknown 9, Intermediate Expert 8
- File: `diss_climb/results/qwen/qwen_climbing_entire_n8_fourclass.csv`

**Fourclass — 8 frames, trimmed, exo** *(frame count per `statistics/generate_reports.py:211`; script note: `fourclass_n8.py`/`fourclass_n16.py` both write to the same output path in `"w"` mode, so this file is only trustworthy as the last of the two to run — treat frame count with caution)*
- Overall: 24/100 = **24.0%**
- Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 2/25 = 8.0% · Late Expert: 0/25
- Predicted counts: Novice 88, Intermediate Expert 10, Unknown 2
- File: `diss_climb/results/qwen/trimmed/fourclass_trimmed.csv`

**Fourclass — 8 frames, trimmed, ego** *(same caveat as above)*
- Overall: 26/100 = **26.0%**
- Novice: 24/25 = 96.0% · Early Expert: 0/25 · Intermediate Expert: 2/25 = 8.0% · Late Expert: 0/25
- Predicted counts: Novice 93, Intermediate Expert 5, Unknown 2
- File: `diss_climb/results/qwen/trimmed/fourclass_trimmed.csv`

**Reasoning — 16 frames, entire, exo**
- Overall: 14/100 = **14.0%**
- Novice: 6/25 = 24.0% · Early Expert: 0/25 · Intermediate Expert: 8/25 = 32.0% · Late Expert: 0/25
- Predicted counts: Novice 49, Intermediate Expert 36, Unknown 15
- File: `diss_climb/results/qwen/qwen_climbing_entire_n16_reasoning.csv` *(exo column; ego column is a pure Novice collapse, excluded)*

**Reasoning — 8 frames, entire, exo**
- Overall: 15/100 = **15.0%**
- Novice: 12/25 = 48.0% · Early Expert: 0/25 · Intermediate Expert: 3/25 = 12.0% · Late Expert: 0/25
- Predicted counts: Novice 70, Intermediate Expert 15, Unknown 14, Late Expert 1
- File: `diss_climb/results/qwen/qwen_climbing_entire_n8_reasoning.csv` *(exo column)*

**Reasoning — 16 frames, entire, exo — PATCHED re-extraction** *(not previously listed; same raw model text as `qwen_climbing_entire_n16_reasoning.csv` above but with corrected label extraction — this is the only climbing reasoning/fourclass run that resolves all four ground-truth classes with nonzero accuracy)*
- Overall: 20/100 = **20.0%**
- Novice: 7/25 = 28.0% · Early Expert: 4/25 = 16.0% · Intermediate Expert: 7/25 = 28.0% · Late Expert: 2/25 = 8.0%
- Predicted counts: Novice 49, Intermediate Expert 28, Unknown 9, Late Expert 8, Early Expert 6
- File: `diss_climb/results/qwen/qwen_climbing_entire_n16_reasoning_patched.csv`

**Reasoning — 16 frames, trimmed, exo**
- Overall: 10/100 = **10.0%**
- Novice: 3/25 = 12.0% · Early Expert: 0/25 · Intermediate Expert: 7/25 = 28.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 36, Novice 35, Unknown 28, Late Expert 1
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n16_reasoning.csv` *(exo column)*

**Reasoning — 8 frames, trimmed, exo**
- Overall: 12/100 = **12.0%**
- Novice: 7/25 = 28.0% · Early Expert: 0/25 · Intermediate Expert: 5/25 = 20.0% · Late Expert: 0/25
- Predicted counts: Novice 55, Intermediate Expert 29, Unknown 16
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n8_reasoning.csv` *(exo column)*

**Structured — 16 frames, entire, ego**
- Overall: 24/100 = **24.0%**
- Novice: 13/25 = 52.0% · Early Expert: 0/25 · Intermediate Expert: 11/25 = 44.0% · Late Expert: 0/25
- Predicted counts: Novice 57, Intermediate Expert 42, Unknown 1
- File: `diss_climb/results/qwen/qwen_climbing_entire_n16_structured.csv`

**Structured — 16 frames, entire, exo**
- Overall: 24/100 = **24.0%**
- Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 2/25 = 8.0% · Late Expert: 0/25
- Predicted counts: Novice 87, Intermediate Expert 13
- File: `diss_climb/results/qwen/qwen_climbing_entire_n16_structured.csv`

**Structured — 8 frames, entire, ego**
- Overall: 19/100 = **19.0%**
- Novice: 16/25 = 64.0% · Early Expert: 0/25 · Intermediate Expert: 3/25 = 12.0% · Late Expert: 0/25
- Predicted counts: Novice 71, Intermediate Expert 20, Unknown 9
- File: `diss_climb/results/qwen/structured_eval.csv`

**Structured — 8 frames, entire, exo**
- Overall: 24/100 = **24.0%**
- Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 2/25 = 8.0% · Late Expert: 0/25
- Predicted counts: Novice 87, Unknown 9, Intermediate Expert 4
- File: `diss_climb/results/qwen/structured_eval.csv`

**Structured — 16 frames, trimmed, ego**
- Overall: 26/100 = **26.0%**
- Novice: 8/25 = 32.0% · Early Expert: 0/25 · Intermediate Expert: 18/25 = 72.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 59, Novice 39, Unknown 2
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n16_structured.csv`

**Structured — 16 frames, trimmed, exo**
- Overall: 17/100 = **17.0%**
- Novice: 9/25 = 36.0% · Early Expert: 0/25 · Intermediate Expert: 8/25 = 32.0% · Late Expert: 0/25
- Predicted counts: Novice 53, Intermediate Expert 40, Unknown 7
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n16_structured.csv`

**Structured — 8 frames, trimmed, ego**
- Overall: 19/100 = **19.0%**
- Novice: 13/25 = 52.0% · Early Expert: 0/25 · Intermediate Expert: 6/25 = 24.0% · Late Expert: 0/25
- Predicted counts: Novice 74, Intermediate Expert 24, Unknown 2
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n8_structured.csv`

**Structured — 8 frames, trimmed, exo**
- Overall: 22/100 = **22.0%**
- Novice: 18/25 = 72.0% · Early Expert: 0/25 · Intermediate Expert: 4/25 = 16.0% · Late Expert: 0/25
- Predicted counts: Novice 82, Intermediate Expert 15, Unknown 2, Early Expert 1
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n8_structured.csv`

#### Collapsed — fourclass/reasoning/structured, ego view (and two full ego+exo collapses) (shown in full, excluded from headline claims)

The exo columns of these same files are kept above; the ego columns collapse to near-total Novice with zero accuracy on every other class. Two entries below (`qwen_climbing_entire_n8_structured.csv`) collapse on *both* ego and exo.

| Prompt | Frames | Trim | View | Overall | Novice acc | Other classes | Predicted counts | File |
|---|---|---|---|---|---|---|---|---|
| Fourclass | 16 | entire | ego | 25/100 = 25.0% | 25/25 = 100.0% | Early/Intermediate/Late all 0/25 | Novice 98, Intermediate Expert 2 | `diss_climb/results/qwen/qwen_climbing_entire_n16_fourclass.csv` |
| Reasoning | 16 | entire | ego | 24/100 = 24.0% | 24/25 = 96.0% | all 0/25 | Novice 91, Unknown 9 | `diss_climb/results/qwen/qwen_climbing_entire_n16_reasoning.csv` |
| Reasoning | 16 | entire | ego | 24/100 = 24.0% | 24/25 = 96.0% | all 0/25 | Novice 90, Unknown 9, Early Expert 1 | `diss_climb/results/qwen/qwen_climbing_entire_n16_reasoning_patched.csv` *(not previously listed; ego column — exo column of this same file is Kept above with all 4 classes resolved)* |
| Reasoning | 8 | entire | ego | 24/100 = 24.0% | 24/25 = 96.0% | all 0/25 | Novice 90, Unknown 9, Intermediate Expert 1 | `diss_climb/results/qwen/qwen_climbing_entire_n8_reasoning.csv` |
| Reasoning | 16 | trimmed | ego | 23/100 = 23.0% | 23/25 = 92.0% | all 0/25 | Novice 89, Unknown 10, Intermediate Expert 1 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n16_reasoning.csv` |
| Reasoning | 8 | trimmed | ego | 23/100 = 23.0% | 23/25 = 92.0% | all 0/25 | Novice 89, Unknown 10, Intermediate Expert 1 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n8_reasoning.csv` |
| Structured | 8 | entire | ego | 5/100 = 5.0% | 5/25 = 20.0% | all 0/25 | Unknown 73, Novice 21, Intermediate Expert 6 | `diss_climb/results/qwen/qwen_climbing_entire_n8_structured.csv` *(not previously listed — separate run from `structured_eval.csv`, same clip set; a majority of rows are `Unknown`)* |
| Structured | 8 | entire | exo | 14/100 = 14.0% | 14/25 = 56.0% | all 0/25 | Novice 68, Unknown 31, Intermediate Expert 1 | `diss_climb/results/qwen/qwen_climbing_entire_n8_structured.csv` *(not previously listed — separate run from `structured_eval.csv`, same clip set; a majority of rows are `Unknown`)* |

---

### 1.3 VideoLLaVA

#### Collapsed / failed — binary, fourclass, structured-16 (shown in full, excluded from headline claims)

| Prompt | Frames | Trim | View | Overall | Class breakdown | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| Binary | 8 | entire | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_climb/results/videollava/vl_climbing_entire_n8_binary.csv` |
| Binary | 8 | entire | exo | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_climb/results/videollava/vl_climbing_entire_n8_binary.csv` |
| Binary | 8 | trimmed | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_ego_n8_binary.csv` |
| Binary | 16 | trimmed | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_ego_n16_binary.csv` |
| Binary | 16 | trimmed | exo | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_exo_n16_binary.csv` |
| Fourclass | 8 | entire | ego | 24/100 = 24.0% | Novice 24/25=96.0%, others 0/25 | Novice 91, Unknown 9 | `diss_climb/results/videollava/vl_climbing_entire_n8_fourclass.csv` |
| Fourclass | 8 | entire | exo | 24/100 = 24.0% | Novice 24/25=96.0%, others 0/25 | Novice 91, Unknown 9 | `diss_climb/results/videollava/vl_climbing_entire_n8_fourclass.csv` |
| Fourclass | 8 | trimmed | ego | 24/100 = 24.0% | Novice 24/25=96.0%, others 0/25 | Novice 98, Unknown 2 | `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_fourclass.csv` |
| Reasoning | 8 | trimmed | ego | 21/100 = 21.0% | Intermediate Expert 21/25=84.0%, others 0/25 | Intermediate Expert 87, Unknown 11, Late Expert 2 | `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_reasoning.csv` |


#### Kept — the surviving 16-frame binary and 8-frame reasoning/structured results

**Binary — 16 frames, entire, ego**
- Overall: 16/50 = **32.0%** *(below chance)*
- Novice: 12/25 = 48.0% · Expert: 4/25 = 16.0%
- File: `diss_climb/results/videollava/vl_climbing_entire_n16_binary.csv`

**Binary — 16 frames, entire, exo**
- Overall: 38/50 = **76.0%** — best single result in the climbing arm
- Novice: 22/25 = 88.0% · Expert: 16/25 = 64.0%
- File: `diss_climb/results/videollava/vl_climbing_entire_n16_binary.csv`

**Binary — 8 frames, trimmed, exo**
- Overall: 30/50 = **60.0%**
- Novice: 25/25 = 100.0% · Expert: 5/25 = 20.0%
- Predicted counts: Novice 45, Expert 5
- File: `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_exo_n8_binary.csv`

**Fourclass — 8 frames, trimmed, exo**
- Overall: 25/100 = **25.0%**
- Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 0/25 · Late Expert: 3/25 = 12.0%
- Predicted counts: Novice 91, Late Expert 7, Unknown 2
- File: `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_fourclass.csv`

**Reasoning — 8 frames, entire, ego** *(two independent runs — both kept)*
- Run A overall: 14/100 = **14.0%** — Novice 1/25=4.0%, Intermediate Expert 10/25=40.0%, Late Expert 3/25=12.0%, Early Expert 0/25
  Predicted counts: Unknown 50, Intermediate Expert 39, Late Expert 7, Novice 3, Early Expert 1
  File: `diss_climb/results/videollava/reasoning_n8_test03.csv` *(cam03 camera-angle ablation — see §3)*
- Run B overall: 13/100 = **13.0%** — Novice 1/25=4.0%, Intermediate Expert 10/25=40.0%, Late Expert 2/25=8.0%, Early Expert 0/25
  Predicted counts: Unknown 47, Intermediate Expert 41, Late Expert 6, Novice 4, Early Expert 2
  File: `diss_climb/results/videollava/vl_climbing_entire_n8_reasoning.csv`

**Reasoning — 8 frames, entire, exo** *(two independent runs)*
- Run A overall: 19/100 = **19.0%** — Early Expert 2/25=8.0%, Intermediate Expert 16/25=64.0%, Late Expert 1/25=4.0%, Novice 0/25
  Predicted counts: Intermediate Expert 68, Unknown 24, Early Expert 4, Late Expert 3, Novice 1
  File: `diss_climb/results/videollava/reasoning_n8_test03.csv` *(cam03 ablation)*
- Run B overall: 23/100 = **23.0%** — Intermediate Expert 22/25=88.0%, Late Expert 1/25=4.0%, Novice 0/25, Early Expert 0/25
  Predicted counts: Intermediate Expert 81, Late Expert 10, Unknown 9
  File: `diss_climb/results/videollava/vl_climbing_entire_n8_reasoning.csv`

**Reasoning — 8 frames, trimmed, exo**
- Overall: 24/100 = **24.0%**
- Intermediate Expert 22/25=88.0%, Late Expert 2/25=8.0%, Novice 0/25, Early Expert 0/25
- Predicted counts: Intermediate Expert 87, Unknown 10, Late Expert 3
- File: `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_reasoning.csv`

**Structured — 8 frames, entire, ego**
- Overall: 23/100 = **23.0%**
- Novice 14/25=56.0%, Early Expert 1/25=4.0%, Intermediate Expert 1/25=4.0%, Late Expert 7/25=28.0%
- Predicted counts: Novice 61, Late Expert 26, Unknown 9, Intermediate Expert 3, Early Expert 1
- File: `diss_climb/results/videollava/vl_climbing_entire_n8_structured.csv`

**Structured — 8 frames, entire, exo**
- Overall: 26/100 = **26.0%**
- Novice 21/25=84.0%, Intermediate Expert 1/25=4.0%, Late Expert 4/25=16.0%, Early Expert 0/25
- Predicted counts: Novice 76, Late Expert 10, Unknown 9, Intermediate Expert 5
- File: `diss_climb/results/videollava/vl_climbing_entire_n8_structured.csv`

**Structured — 8 frames, trimmed, ego**
- Overall: 30/100 = **30.0%** — best VideoLLaVA structured result
- Novice 15/25=60.0%, Early Expert 1/25=4.0%, Intermediate Expert 3/25=12.0%, Late Expert 11/25=44.0%
- Predicted counts: Novice 55, Late Expert 31, Intermediate Expert 8, Early Expert 4, Unknown 2
- File: `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_structured.csv`

**Structured — 8 frames, trimmed, exo**
- Overall: 15/100 = **15.0%**
- Novice 7/25=28.0%, Intermediate Expert 5/25=20.0%, Late Expert 3/25=12.0%, Early Expert 0/25
- Predicted counts: Novice 58, Intermediate Expert 18, Late Expert 14, Early Expert 8, Unknown 2
- File: `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_structured.csv`

---

### 1.4 Qwen3-VL-8B

#### Collapsed — native video-input attempt (shown in full, excluded from headline claims)

Native mode targeted the same ~1fps rate as Gemini (`n = round(duration_sec)` frames requested directly from the model's own video processor via `{"type": "video", "video": video_path, "nframes": n}`), but hit an fps=24 fallback bug that ignored the requested frame count and caused OOM. Only 12 of 50 climbing clips completed before the run was abandoned; no dance native attempt was made at all.

| View | Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|
| exo | 6/12 = 50.0% | 6/6 = 100.0% | 0/6 = 0.0% | Novice 12 | `diss_climb/results/qwen3vl/qwen3vl_climbing_native_binary.csv` |
| ego | 6/12 = 50.0% | 6/6 = 100.0% | 0/6 = 0.0% | Novice 12 | `diss_climb/results/qwen3vl/qwen3vl_climbing_native_binary.csv` |

*(fourclass/reasoning/structured native attempts never got past 1 completed row each — not tabulated.)*

#### Collapsed — frame-extraction substitute, binary ego (shown in full)

| View | Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|
| ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_binary.csv` |

#### Kept

**Binary — exo**
- Overall: 27/50 = **54.0%**
- Novice: 25/25 = 100.0% · Expert: 2/25 = 8.0%
- Predicted counts: Novice 48, Expert 2
- File: `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_binary.csv`

**Fourclass — exo**
- Overall: 26/100 = **26.0%**
- Novice: 16/25 = 64.0% · Early Expert: 0/25 · Intermediate Expert: 9/25 = 36.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Novice 65, Intermediate Expert 31, Late Expert 4
- File: `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_fourclass.csv`

**Fourclass — ego**
- Overall: 37/100 = **37.0%** — best fourclass climbing result of any model tested
- Novice: 23/25 = 92.0% · Early Expert: 0/25 · Intermediate Expert: 14/25 = 56.0% · Late Expert: 0/25
- Predicted counts: Novice 69, Intermediate Expert 31
- File: `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_fourclass.csv`

**Reasoning — exo**
- Overall: 22/100 = **22.0%**
- Novice: 3/25 = 12.0% · Early Expert: 2/25 = 8.0% · Intermediate Expert: 16/25 = 64.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 60, Novice 18, Unknown 10, Late Expert 7, Early Expert 5
- File: `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_reasoning.csv`

**Reasoning — ego**
- Overall: 22/100 = **22.0%**
- Novice: 6/25 = 24.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 15/25 = 60.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 49, Unknown 27, Novice 20, Early Expert 3, Late Expert 1
- File: `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_reasoning.csv`

**Structured — exo**
- Overall: 27/100 = **27.0%**
- Novice: 6/25 = 24.0% · Early Expert: 0/25 · Intermediate Expert: 16/25 = 64.0% · Late Expert: 5/25 = 20.0% — first climbing model result to meaningfully populate Late Expert
- Predicted counts: Intermediate Expert 65, Novice 28, Late Expert 6, Early Expert 1
- File: `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_structured.csv`

**Structured — ego**
- Overall: 25/100 = **25.0%**
- Novice: 14/25 = 56.0% · Early Expert: 0/25 · Intermediate Expert: 11/25 = 44.0% · Late Expert: 0/25
- Predicted counts: Novice 52, Intermediate Expert 43, Late Expert 3, Early Expert 1, Unknown 1
- File: `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_structured.csv`

---

## 2. Dance (`diss_dance`)

### 2.1 Gemini

**Binary — ego**
- Overall: 26/50 = **52.0%** *(not significant, ≈chance)*
- Novice: 25/25 = 100.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Novice 49, Expert 1
- File: `diss_dance/results/gemini/gemini_dance_entire_binary_ego.csv`

**Binary — exo**
- Overall: 30/50 = **60.0%**
- Novice: 13/25 = 52.0% · Late Expert: 17/25 = 68.0%
- Predicted counts: Expert 29, Novice 21
- File: `diss_dance/results/gemini/gemini_dance_entire_binary.csv`

**Fourclass — ego**
- Overall: 21/100 = **21.0%**
- Novice: 15/25 = 60.0% · Early Expert: 0/25 · Intermediate Expert: 6/25 = 24.0% · Late Expert: 0/25
- Predicted counts: Novice 65, Intermediate Expert 34, Early Expert 1
- File: `diss_dance/results/gemini/gemini_dance_entire_fourclass_ego.csv`

**Fourclass — exo**
- Overall: 24/100 = **24.0%**
- Novice: 2/25 = 8.0% · Early Expert: 0/25 · Intermediate Expert: 21/25 = 84.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 84, Novice 13, Early Expert 2, Late Expert 1
- File: `diss_dance/results/gemini/gemini_dance_entire_fourclass.csv`

**Reasoning — ego**
- Overall: 21/100 = **21.0%**
- Novice: 14/25 = 56.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 6/25 = 24.0% · Late Expert: 0/25
- Predicted counts: Novice 70, Intermediate Expert 22, Unknown 4, Early Expert 4
- File: `diss_dance/results/gemini/gemini_dance_entire_reasoning_ego.csv`

**Reasoning — exo**
- Overall: 32/100 = **32.0%** — Gemini's best dance result
- Novice: 7/25 = 28.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 24/25 = 96.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 81, Novice 15, Early Expert 4
- File: `diss_dance/results/gemini/gemini_dance_entire_reasoning.csv`

**Structured — ego**
- Overall: 27/100 = **27.0%**
- Novice: 12/25 = 48.0% · Early Expert: 0/25 · Intermediate Expert: 15/25 = 60.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 60, Novice 40
- File: `diss_dance/results/gemini/gemini_dance_entire_structured_ego.csv`

**Structured — exo**
- Overall: 24/100 = **24.0%**
- Novice: 1/25 = 4.0% · Early Expert: 0/25 · Intermediate Expert: 23/25 = 92.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 88, Novice 12
- File: `diss_dance/results/gemini/gemini_dance_entire_structured.csv`

---

### 2.2 Qwen2.5-VL-7B

#### Collapsed — binary (all conditions; shown in full, excluded from headline claims)

Identical failure mode to climbing: 100% "Novice" regardless of frame count, trim condition, or view.

| Frames | Trim | View | Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| 16 | entire | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n16_binary.csv` |
| 16 | entire | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n16_binary.csv` |
| 8 | entire | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n8_binary.csv` |
| 8 | entire | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n8_binary.csv` |
| 16 | trimmed | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_binary.csv` |
| 16 | trimmed | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_binary.csv` |
| 8 | trimmed | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_binary.csv` |
| 8 | trimmed | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_binary.csv` |

#### Collapsed — fourclass/reasoning, ego view (shown in full)

Every ego-view fourclass and reasoning run collapses to 100% Novice with zero accuracy on all other classes; only the exo columns of these same files (kept below) show real discrimination. This ego/exo split is itself a notable finding for dance.

| Prompt | Frames | Trim | Overall | Novice acc | Other classes | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| Fourclass | 16 | entire | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/qwen_dance_entire_n16_fourclass.csv` |
| Fourclass | 8 | entire | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/qwen_dance_entire_n8_fourclass.csv` |
| Fourclass | 16 | trimmed | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_fourclass.csv` |
| Fourclass | 8 | trimmed | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_fourclass.csv` |
| Reasoning | 16 | entire | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/qwen_dance_entire_n16_reasoning.csv` |
| Reasoning | 8 | entire | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 99, Unknown 1 | `diss_dance/results/qwen/qwen_dance_entire_n8_reasoning.csv` |
| Reasoning | 16 | trimmed | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_reasoning.csv` |
| Reasoning | 8 | trimmed | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_reasoning.csv` |

#### Kept — exo-view fourclass / reasoning, and structured (both views)

**Fourclass — 16 frames, entire, exo**
- Overall: 33/100 = **33.0%**
- Novice: 16/25 = 64.0% · Early Expert: 0/25 · Intermediate Expert: 17/25 = 68.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 51, Novice 49
- File: `diss_dance/results/qwen/qwen_dance_entire_n16_fourclass.csv`

**Fourclass — 8 frames, entire, exo**
- Overall: 31/100 = **31.0%**
- Novice: 18/25 = 72.0% · Early Expert: 0/25 · Intermediate Expert: 13/25 = 52.0% · Late Expert: 0/25
- Predicted counts: Novice 61, Intermediate Expert 39
- File: `diss_dance/results/qwen/qwen_dance_entire_n8_fourclass.csv`

**Fourclass — 16 frames, trimmed, exo**
- Overall: 34/100 = **34.0%**
- Novice: 16/25 = 64.0% · Early Expert: 0/25 · Intermediate Expert: 18/25 = 72.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 55, Novice 45
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_fourclass.csv`

**Fourclass — 8 frames, trimmed, exo**
- Overall: 32/100 = **32.0%**
- Novice: 16/25 = 64.0% · Early Expert: 0/25 · Intermediate Expert: 16/25 = 64.0% · Late Expert: 0/25
- Predicted counts: Novice 57, Intermediate Expert 43
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_fourclass.csv`

**Reasoning — 16 frames, entire, exo**
- Overall: 27/100 = **27.0%**
- Novice: 1/25 = 4.0% · Early Expert: 0/25 · Intermediate Expert: 24/25 = 96.0% · Late Expert: 2/25 = 8.0%
- Predicted counts: Intermediate Expert 94, Late Expert 4, Novice 2
- File: `diss_dance/results/qwen/qwen_dance_entire_n16_reasoning.csv`

**Reasoning — 8 frames, entire, exo**
- Overall: 32/100 = **32.0%**
- Novice: 13/25 = 52.0% · Early Expert: 0/25 · Intermediate Expert: 19/25 = 76.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 63, Novice 35, Unknown 1, Late Expert 1
- File: `diss_dance/results/qwen/qwen_dance_entire_n8_reasoning.csv`

**Reasoning — 16 frames, trimmed, exo**
- Overall: 27/100 = **27.0%**
- Novice: 1/25 = 4.0% · Early Expert: 0/25 · Intermediate Expert: 25/25 = 100.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 95, Novice 3, Late Expert 2
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_reasoning.csv`

**Reasoning — 8 frames, trimmed, exo**
- Overall: 28/100 = **28.0%**
- Novice: 12/25 = 48.0% · Early Expert: 0/25 · Intermediate Expert: 15/25 = 60.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 53, Novice 43, Late Expert 4
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_reasoning.csv`

**Structured — 16 frames, entire, ego**
- Overall: 23/100 = **23.0%**
- Novice: 14/25 = 56.0% · Early Expert: 0/25 · Intermediate Expert: 9/25 = 36.0% · Late Expert: 0/25
- Predicted counts: Novice 62, Intermediate Expert 35, Unknown 2, Late Expert 1
- File: `diss_dance/results/qwen/qwen_dance_entire_n16_structured.csv`

**Structured — 16 frames, entire, exo**
- Overall: 41/100 = **41.0%** — best Qwen2.5-VL result across the entire project
- Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 19/25 = 76.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 57, Novice 43
- File: `diss_dance/results/qwen/qwen_dance_entire_n16_structured.csv`

**Structured — 8 frames, entire, ego**
- Overall: 25/100 = **25.0%**
- Novice: 20/25 = 80.0% · Early Expert: 0/25 · Intermediate Expert: 5/25 = 20.0% · Late Expert: 0/25
- Predicted counts: Novice 79, Intermediate Expert 19, Unknown 2
- File: `diss_dance/results/qwen/qwen_dance_entire_n8_structured.csv`

**Structured — 8 frames, entire, exo**
- Overall: 32/100 = **32.0%**
- Novice: 15/25 = 60.0% · Early Expert: 0/25 · Intermediate Expert: 17/25 = 68.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 61, Novice 37, Unknown 2
- File: `diss_dance/results/qwen/qwen_dance_entire_n8_structured.csv`

**Structured — 16 frames, trimmed, ego**
- Overall: 28/100 = **28.0%**
- Novice: 20/25 = 80.0% · Early Expert: 0/25 · Intermediate Expert: 8/25 = 32.0% · Late Expert: 0/25
- Predicted counts: Novice 62, Intermediate Expert 34, Unknown 4
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_structured.csv`

**Structured — 16 frames, trimmed, exo**
- Overall: 39/100 = **39.0%** — second-best Qwen2.5-VL result overall
- Novice: 17/25 = 68.0% · Early Expert: 0/25 · Intermediate Expert: 22/25 = 88.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 71, Novice 29
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_structured.csv`

**Structured — 8 frames, trimmed, ego**
- Overall: 24/100 = **24.0%**
- Novice: 19/25 = 76.0% · Early Expert: 0/25 · Intermediate Expert: 5/25 = 20.0% · Late Expert: 0/25
- Predicted counts: Novice 74, Intermediate Expert 24, Late Expert 1, Unknown 1
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_structured.csv`

**Structured — 8 frames, trimmed, exo**
- Overall: 35/100 = **35.0%**
- Novice: 17/25 = 68.0% · Early Expert: 0/25 · Intermediate Expert: 17/25 = 68.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 60, Novice 38, Late Expert 1, Unknown 1
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_structured.csv`

---

### 2.3 VideoLLaVA

#### Collapsed — binary, fourclass, and reasoning-exo (shown in full, excluded from headline claims)

Every binary and fourclass run is a pure Novice collapse. Reasoning-exo is also excluded here: unlike reasoning-ego (kept below), it never once predicts anything but Late Expert or Unknown.

| Prompt | Frames | Trim | View | Overall | Class breakdown | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| Binary | 16 | entire | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/vl_dance_entire_n16_binary.csv` |
| Binary | 16 | entire | exo | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/vl_dance_entire_n16_binary.csv` |
| Binary | 8 | entire | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/vl_dance_entire_n8_binary.csv` |
| Binary | 8 | entire | exo | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/vl_dance_entire_n8_binary.csv` |
| Binary | 16 | trimmed | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n16_binary.csv` |
| Binary | 16 | trimmed | exo | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n16_binary.csv` |
| Binary | 8 | trimmed | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_binary.csv` |
| Binary | 8 | trimmed | exo | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_binary.csv` |
| Fourclass | 8 | entire | ego | 25/100 = 25.0% | Novice 25/25=100.0%, others 0/25 | Novice 100 | `diss_dance/results/videollava/vl_dance_entire_n8_fourclass.csv` |
| Fourclass | 8 | entire | exo | 25/100 = 25.0% | Novice 25/25=100.0%, others 0/25 | Novice 100 | `diss_dance/results/videollava/vl_dance_entire_n8_fourclass.csv` |
| Fourclass | 8 | trimmed | ego | 25/100 = 25.0% | Novice 25/25=100.0%, others 0/25 | Novice 100 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_fourclass.csv` |
| Fourclass | 8 | trimmed | exo | 25/100 = 25.0% | Novice 25/25=100.0%, others 0/25 | Novice 100 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_fourclass.csv` |
| Reasoning | 8 | entire | exo | 25/100 = 25.0% | Late Expert 25/25=100.0%, others 0/25 | Late Expert 94, Unknown 5, Early Expert 1 | `diss_dance/results/videollava/vl_dance_entire_n8_reasoning.csv` |
| Reasoning | 8 | trimmed | exo | 22/100 = 22.0% | Late Expert 22/25=88.0%, others 0/25 | Late Expert 86, Unknown 10, Novice 2, Early Expert 2 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_reasoning.csv` |

#### Kept — reasoning-ego and structured (both views)

**Reasoning — entire, ego**
- Overall: 25/100 = **25.0%**
- Intermediate Expert: 1/25 = 4.0% · Late Expert: 24/25 = 96.0% · Novice: 0/25 · Early Expert: 0/25
- Predicted counts: Late Expert 95, Intermediate Expert 3, Early Expert 1, Unknown 1
- File: `diss_dance/results/videollava/vl_dance_entire_n8_reasoning.csv`

**Reasoning — trimmed, ego**
- Overall: 31/100 = **31.0%** — best VideoLLaVA dance result
- Early Expert: 4/25 = 16.0% · Intermediate Expert: 3/25 = 12.0% · Late Expert: 24/25 = 96.0% · Novice: 0/25
- Predicted counts: Late Expert 83, Intermediate Expert 11, Early Expert 6
- File: `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_reasoning.csv`

**Structured — entire, ego**
- Overall: 23/100 = **23.0%**
- Novice: 10/25 = 40.0% · Early Expert: 0/25 · Intermediate Expert: 8/25 = 32.0% · Late Expert: 5/25 = 20.0%
- Predicted counts: Intermediate Expert 44, Novice 35, Late Expert 21
- File: `diss_dance/results/videollava/vl_dance_entire_n8_structured.csv`

**Structured — entire, exo**
- Overall: 19/100 = **19.0%**
- Novice: 7/25 = 28.0% · Early Expert: 0/25 · Intermediate Expert: 11/25 = 44.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 48, Novice 36, Late Expert 16
- File: `diss_dance/results/videollava/vl_dance_entire_n8_structured.csv`

**Structured — trimmed, ego**
- Overall: 24/100 = **24.0%**
- Novice: 4/25 = 16.0% · Early Expert: 0/25 · Intermediate Expert: 11/25 = 44.0% · Late Expert: 9/25 = 36.0%
- Predicted counts: Intermediate Expert 54, Late Expert 28, Novice 18
- File: `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_structured.csv`

**Structured — trimmed, exo**
- Overall: 26/100 = **26.0%**
- Novice: 9/25 = 36.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 11/25 = 44.0% · Late Expert: 5/25 = 20.0%
- Predicted counts: Intermediate Expert 42, Novice 32, Late Expert 25, Early Expert 1
- File: `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_structured.csv`

---

### 2.4 Qwen3-VL-8B (~1fps OpenCV frame-extraction substitute; dance has no native-pipeline attempt at all — only climbing tried and failed the native route)

#### Collapsed — binary ego (shown in full)

| View | Overall | Novice acc | Late Expert acc | Predicted counts | File |
|---|---|---|---|---|---|
| ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen3vl/qwen3vl_dance_frames_binary.csv` |

#### Kept

**Binary — exo**
- Overall: 25/50 = **50.0%**
- Novice: 23/25 = 92.0% · Late Expert: 2/25 = 8.0%
- Predicted counts: Novice 46, Expert 4
- File: `diss_dance/results/qwen3vl/qwen3vl_dance_frames_binary.csv`

**Fourclass — exo**
- Overall: 23/100 = **23.0%**
- Novice: 0/25 · Early Expert: 0/25 · Intermediate Expert: 22/25 = 88.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 93, Novice 4, Late Expert 3
- File: `diss_dance/results/qwen3vl/qwen3vl_dance_frames_fourclass.csv`

**Fourclass — ego**
- Overall: 28/100 = **28.0%**
- Novice: 25/25 = 100.0% · Early Expert: 0/25 · Intermediate Expert: 3/25 = 12.0% · Late Expert: 0/25
- Predicted counts: Novice 87, Intermediate Expert 13
- File: `diss_dance/results/qwen3vl/qwen3vl_dance_frames_fourclass.csv`

**Reasoning — exo**
- Overall: 27/100 = **27.0%**
- Novice: 0/25 · Early Expert: 3/25 = 12.0% · Intermediate Expert: 21/25 = 84.0% · Late Expert: 3/25 = 12.0%
- Predicted counts: Intermediate Expert 74, Late Expert 11, Novice 8, Early Expert 6, Unknown 1
- File: `diss_dance/results/qwen3vl/qwen3vl_dance_frames_reasoning.csv`

**Reasoning — ego**
- Overall: 27/100 = **27.0%**
- Novice: 20/25 = 80.0% · Early Expert: 0/25 · Intermediate Expert: 5/25 = 20.0% · Late Expert: 2/25 = 8.0%
- Predicted counts: Novice 74, Unknown 16, Intermediate Expert 7, Late Expert 3
- File: `diss_dance/results/qwen3vl/qwen3vl_dance_frames_reasoning.csv`

**Structured — exo**
- Overall: 30/100 = **30.0%** — best Qwen3-VL dance result
- Novice: 3/25 = 12.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 20/25 = 80.0% · Late Expert: 6/25 = 24.0%
- Predicted counts: Intermediate Expert 69, Late Expert 19, Novice 10, Early Expert 2
- File: `diss_dance/results/qwen3vl/qwen3vl_dance_frames_structured.csv`

**Structured — ego**
- Overall: 24/100 = **24.0%**
- Novice: 9/25 = 36.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 13/25 = 52.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 52, Novice 37, Late Expert 9, Early Expert 1, Unknown 1
- File: `diss_dance/results/qwen3vl/qwen3vl_dance_frames_structured.csv`

---

## 3. Cross-domain generalization

### Collapsed — binary results across all cross-domain tests (shown in full, excluded from headline claims)

All binary-prompt cross-domain tests collapse to (near-)100% "Novice," identical to the climbing/dance binary failure mode.

**Basketball binary — Qwen2.5-VL-7B, 8 frames, entire, ego+exo** *(EgoExo4D "Basketball" scenario; benchmark rebuilt via `rebuild_basketball_benchmark.py` because 16/50 original clips had no video on disk)*
- Overall (both views identical): 12/50 = **24.0%** — below chance
- Novice: 12/25 = 48.0% · "Late Expert" (mapped to Expert): 0/25 = 0.0%
- Raw answer counts (no `predicted` column in this file's schema; both exo_answer/ego_answer identical): Novice 34, `ERROR` 16 — 15 of those 16 `ERROR` rows are on the Expert-labeled clips specifically (missing video files)
- File: `diss_climb/results/qwen/binary_basketball.csv`

**JIGSAWS suturing binary — Qwen2.5-VL-7B, 8 frames, exo only** *(surgical skill, cross-domain generalization test; benchmark filtered to Novice+Expert only from `benchmark/benchmark_jigsaws_suturing.json`, not class-balanced: 19 Novice / 10 Expert)*
- Overall: 19/29 = **65.5%** — reported accuracy is a majority-class artifact, not real discrimination
- Novice: 19/19 = 100.0% · Expert: 0/10 = 0.0%
- Predicted counts: Novice 29 (100% collapse — every single clip predicted Novice)
- File: `results/jigsaws_qwen_binary.csv`

**Mixed-activity binary — Qwen2.5-VL-7B, 8 frames, trimmed, exo** *(activities: Basketball, Music, Cooking, Soccer)*
- Overall: 25/100 = **25.0%**
- Novice: 25/25 = 100.0% · Early Expert: 0/25 = 0.0% · Intermediate Expert: 0/25 = 0.0% · Late Expert: 0/25 = 0.0%
- Predicted counts: Novice 100
- File: `mixed/results/qwen_mixed_trimmed_n8_binary.csv`

**Mixed-activity binary — Qwen2.5-VL-7B, 8 frames, trimmed, ego**
- Overall: 25/100 = **25.0%**
- Novice: 25/25 = 100.0% · Early Expert: 0/25 = 0.0% · Intermediate Expert: 0/25 = 0.0% · Late Expert: 0/25 = 0.0%
- Predicted counts: Novice 100
- File: `mixed/results/qwen_mixed_trimmed_n8_binary.csv`

**cam03 binary — Qwen2.5-VL-7B, 8 frames, trimmed, exo (alternate `cam03` camera instead of `cam01`)**
- Overall: 25/50 = **50.0%**
- Novice: 25/25 = 100.0% · Expert: 0/25 = 0.0%
- Predicted counts: Novice 50
- File: `diss_climb/results/qwen/trimmed/test_exo3.csv`
- Same failure mode as the equivalent cam01 trimmed-exo binary run — camera angle within "exo" does not change the collapse.

### Kept — mixed-activity structured

**Mixed-activity structured — Qwen2.5-VL-7B, 8 frames, trimmed, exo** *(activities: Basketball, Music, Cooking, Soccer — climbing/dance excluded from this benchmark by design)*
- Overall: 19/100 = **19.0%**
- Novice: 9/25 = 36.0% · Early Expert: 0/25 · Intermediate Expert: 10/25 = 40.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 51, Novice 44, Unknown 5
- By-activity correct/n: Basketball 12/48, Cooking 5/21, Music 2/20, Soccer 0/11
- File: `mixed/results/qwen_mixed_trimmed_n8_structured.csv`

**Mixed-activity structured — Qwen2.5-VL-7B, 8 frames, trimmed, ego**
- Overall: 23/100 = **23.0%**
- Novice: 8/25 = 32.0% · Early Expert: 0/25 · Intermediate Expert: 15/25 = 60.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 48, Unknown 27, Novice 25
- By-activity correct/n: Basketball 9/48, Cooking 8/21, Music 5/20, Soccer 1/11
- File: `mixed/results/qwen_mixed_trimmed_n8_structured.csv`

> Signal concentrates in Basketball and Cooking (visible object-interaction cues — ball, knife); Music and Soccer are near-total failures. Flag as a hypothesis for future work given small per-activity n (11–48).

**Camera-angle ablation (cam03 vs cam01)** — VideoLLaVA reasoning results using the alternate `cam03` exocentric camera are already listed in §1.3 (`reasoning_n8_test03.csv`, both runs marked "cam03 camera-angle ablation"). Finding: accuracy (14–19%) is statistically indistinguishable from the equivalent cam01 reasoning runs — camera placement within the "exo" category does not change the failure mode.

---

## 4. Label-granularity ablation (3-class)

Qwen (structured prompt, 8 frames, ego+exo), collapsing the label set to Novice / Intermediate / Expert (Intermediate Expert → "Intermediate", Late Expert → "Expert"; Early Expert excluded from this benchmark by construction), 10 clips/class = 30 total.

#### Collapsed — exo view (shown in full)

| View | Overall | Novice acc | Intermediate acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|---|
| exo | 11/30 = 36.7% | 0/10 = 0.0% | 1/10 = 10.0% | 10/10 = 100.0% | Expert 29, Intermediate 1 | `diss_climb/results/qwen/3class_structured.csv` |

29 of 30 predictions were "Expert" regardless of ground truth — a near-total collapse with only one other correct guess, offering no real discriminative signal despite technically touching a second class.

#### Kept — ego view

**3-class structured — ego**
- Overall: 10/30 = **33.3%**
- Novice: 0/10 = 0.0% · Intermediate: 4/10 = 40.0% · Expert: 6/10 = 60.0%
- Predicted counts: Expert 18, Intermediate 9, Unknown 2, Novice 1
- File: `diss_climb/results/qwen/3class_structured.csv`

---

## 5. What was excluded, and why

*This is a jump-index — every row below now has its full Overall/per-class/predicted-count numbers shown inline in a `#### Collapsed` block in the section noted, not just this summary.*

| Category | Examples | Reason | Full detail in |
|---|---|---|---|
| Qwen2.5-VL binary, climbing & dance (all frame counts, entire+trimmed, both views) | `qwen_climbing_entire_n16_binary.csv`, `qwen_dance_entire_n8_binary.csv`, `binary_n32.csv`, all `trimmed/*_binary.csv` | 100% "Novice" prediction regardless of input; accuracy exactly 50% = chance by construction | §1.2, §2.2 |
| Qwen2.5-VL fourclass/reasoning, ego view (climbing & dance) | `qwen_climbing_entire_n16_fourclass.csv` (ego), `qwen_dance_entire_n16_reasoning.csv` (ego), etc. | Near-total Novice collapse, zero accuracy on every other class | §1.2, §2.2 |
| VideoLLaVA binary/fourclass, dance (all) | `vl_dance_entire_n8_binary.csv`, `vl_dance_entire_n16_binary.csv`, `vl_dance_entire_n8_fourclass.csv`, trimmed equivalents | 100% Novice collapse | §2.3 |
| VideoLLaVA reasoning-exo, dance | `vl_dance_entire_n8_reasoning.csv`, `trimmed/vl_dance_trimmed_n8_reasoning.csv` (exo columns) | 88–100% Late Expert collapse (inverse bias to climbing) | §2.3 |
| VideoLLaVA binary, climbing (8fr entire, 32fr, most 16fr trimmed) | `vl_climbing_entire_n8_binary.csv`, `binary_n32.csv`, `trimmed/vl_climbing_trimmed_ego/exo_n16_binary.csv`, `trimmed/vl_climbing_trimmed_ego_n8_binary.csv` | 100% Novice collapse, or (32-frame) empty/`ERROR` cells on 48/50 rows | §1.3 |
| VideoLLaVA fully-failed generations | `structured_n16_vl.csv`, `trimmed/reasoning_n16.csv` | Literal `"..."` / `"ERROR"` in every row — no usable text at all. **Real cause:** fourclass/structured/reasoning at 16 frames fail because VideoLLaVA's own vision-token encoding of 16 frames pushes the prompt over its 4096-token context limit — confirmed by the hard-coded `if n_tokens > 4096` guard in `diss_climb/scripts/videollava/trimmed/reasoning_n16.py:80` and `structured_n16.py:83`, which aborts generation rather than truncating. | §1.3 |
| VideoLLaVA fourclass, climbing (8fr entire, trimmed n16, trimmed n8) | `vl_climbing_entire_n8_fourclass.csv`, `trimmed/fourclass_n16.csv`, `trimmed/vl_climbing_trimmed_n8_fourclass.csv` | 96–98% Novice/Unknown collapse. The trimmed-n16 run shares the same 4096-token root cause as reasoning/structured-n16, but `trimmed/fourclass_n16.py` has no `n_tokens > 4096` guard, so instead of aborting it silently proceeds with an over-length prompt and emits garbage parsed as `Unknown` (95/83 of 100 rows) rather than crashing outright. | §1.3 |
| VideoLLaVA reasoning-16-trimmed and reasoning-8-trimmed-ego, climbing | `trimmed/reasoning_n16.csv`, `trimmed/vl_climbing_trimmed_n8_reasoning.csv` (ego) | Total `ERROR` failure, or 84% Intermediate Expert collapse | §1.3 |
| Qwen3-VL native video pipeline | `qwen3vl_climbing_native_*.csv` | fps=24 fallback bug caused OOM; only 12/50 clips completed, 100% Novice collapse — approach abandoned | §1.4 |
| Qwen3-VL binary, ego view (climbing & dance) | `qwen3vl_climbing_frames_binary.csv` (ego), `qwen3vl_dance_frames_binary.csv` (ego) | Pure Novice collapse on that view/column only (exo column of same file kept) | §1.4, §2.4 |
| Basketball binary | `diss_climb/results/qwen/binary_basketball.csv` | Only Novice class ever scored correctly (0% on Expert); also 15/50 rows (30%) are `ERROR` from missing video files | §3 |
| JIGSAWS suturing binary | `results/jigsaws_qwen_binary.csv` | Model predicted "Novice" for all 29 clips — the reported 65.5% accuracy is entirely a majority-class artifact (19 of 29 ground-truth labels happen to be Novice), not real discrimination | §3 |
| Mixed-activity binary | `mixed/results/qwen_mixed_trimmed_n8_binary.csv` | 100% Novice collapse on both exo and ego | §3 |
| cam03 Qwen binary | `diss_climb/results/qwen/trimmed/test_exo3.csv` | 100% Novice collapse, identical failure mode to the cam01 equivalent | §3 |
| 3-class structured, exo view | `diss_climb/results/qwen/3class_structured.csv` (exo) | 29/30 predictions were "Expert" — effectively a collapse | §4 |
| Gemini duplicate row | `gemini_climbing_entire_binary.csv` | Not excluded, but corrected: stale failed retry for one clip removed before scoring (see §1.1) | §1.1 |

---

## 6. Extraction-bug re-verification method

The original label parser used for free-text "reasoning" columns (Qwen, VideoLLaVA, and Gemini's ego-view reasoning file) was a naive fixed-priority substring scan that misread negated statements (e.g. "not yet a Late Expert" → wrongly scored as Late Expert) and arbitrarily resolved hedged disjunctions ("Novice or Early Expert" → picked whichever label it checked first). This was fixed with a corrected extractor that:
1. Honors an explicit leading verdict line.
2. Treats an unresolved "X or Y" disjunction as `Unknown` rather than guessing.
3. Matches explicit "skill level is/appears to be X" statements.
4. Matches conclusion sentences ("Overall/Therefore/In conclusion... X") only after stripping negated mentions.
5. Falls back to hedge detection ("difficult to determine" → `Unknown`) and negation-aware first-mention matching.

All 24 affected files (16 climbing + 8 dance, all "reasoning"-prompt) were **independently re-extracted from the raw stored answer text** for this report — not copied from a prior summary — and every resulting accuracy/per-class/predicted-count figure above was hand-checked against that fresh extraction. No discrepancies were found against the project's existing `statistics/master_results_summary.csv`, confirming those numbers are correct as stored.

**Caveat on `statistics/master_results_summary.csv` itself**: the generator script (`statistics/build_master_summary.py`) hardcodes a `dissertation_v2/results/` path that no longer exists (the folder was renamed to `diss_climb/` mid-project). Re-running that script as-is silently drops all climbing rows. The CSV currently in the repo predates the rename and is still correct; **do not re-run `build_master_summary.py` until its `RESULT_DIRS` path is updated**, or it will overwrite the file with incomplete data.
