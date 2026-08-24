# Dissertation Results — Full Detail, Collapsed Results Clearly Flagged

*Compiled 2026-08-13 (supersedes the 2026-08-06 version). Every number below was regenerated from the raw result CSVs using the project's unified negation-aware answer parser (`figures-frame_count/rescore.py` / `figures-video/rescore_video.py`, re-run fresh for this pass) — not copy-pasted from the prior summary, and not the same hand-built extractor used in the 2026-08-06 version. Files not covered by that pipeline's manifest (ad-hoc probes, cross-domain tests, and every file added after 2026-08-07) were re-parsed for this report by importing the identical `parse()`/`to_binary_gt()` functions from `rescore.py`, so every number in this document — canonical grid or ad-hoc — now comes from one shared parser. Every result the pipelines produced — including collapsed/degenerate ones — is shown in full detail (Overall accuracy → per-class accuracy → predicted-answer counts → source file), organized under a `#### Collapsed` or `#### Kept` sub-header within each model/activity section so it's clear at a glance which numbers should and shouldn't be cited as evidence of real model capability.*

**What changed since the 2026-08-06 version:**
1. **Numbers shifted** for most "reasoning"-prompt runs (free-text answers) and a handful of "structured"/"fourclass" runs. The 2026-08-06 version's hand extractor and the pipeline's automated negation-aware parser were independently built and, it turns out, disagreed on a non-trivial share of free-text answers — re-running the actual pipeline (rather than trusting the earlier hand-check) is the point of this pass. The largest swing: **VideoLLaVA climbing, trimmed n8 reasoning, ego view, changes from Collapsed (84% single-class) to Kept (two ground-truth classes resolved)** — see §1.3.
2. **New experiments added** since 2026-08-07 are now included: a per-clip best-exocentric-camera probe (§1.2.4), a 64-frame condition (§1.2.5), a ~4×-scale generalization check on 389 clips (§1.2.6), and a text-only "commentary" control that asks whether skill level is recoverable from expert commentary language alone with no video at all (§7).
3. The label-granularity (3-class) ablation in §4 also changed materially under re-parsing — its "Collapsed" exo run drops from 100% Expert-class recall to 80%, revealing 7 previously-miscounted answers.
4. Everything else (Gemini binary/fourclass/structured/reasoning, both activities; Qwen3-VL fourclass/structured/binary; most of the cross-domain §3 results) reproduced **exactly**, confirming the 2026-08-06 hand-verification was correct for those runs.
5. **2026-08-18 update:** the 64-frame condition, previously climbing-only (§1.2.5), now also has Qwen2.5-VL dance results (§2.2.4) — entire clip, all four prompts, both views. The §2.2 "Corrections" note claiming "no 64-frame dance runs exist" is superseded.

**Exclusion rule (governs what counts as "Kept" vs "Collapsed", not what's shown):** a result is marked **Collapsed** if the model's predicted-label distribution is a **near-total collapse to a single label** — formally, if at most one ground-truth class has non-zero accuracy (i.e. the model only ever gets the majority/default class right and is never correct on any other class). This was applied uniformly and reproducibly across every file, not judged case-by-case (one narrow exception, carried over from the prior version and flagged inline, is the 3-class exo run in §4, which is called Collapsed on prediction-distribution grounds even though it technically touches two ground-truth classes). Collapsed results are not deleted — they're tabulated in full under their own sub-header — but should not be cited as evidence of discriminative skill judgment; only "Kept" results should be used for headline claims. Chance baselines: **50%** for binary (25/25 split), **25%** for four-class (25 per class, except the 389-clip §1.2.6 benchmark at ~100/class).

Two extraction bugs were found and corrected in the 2026-08-06 version and remain fixed here (see §7):
1. **Free-text "reasoning" answers** (Qwen, VideoLLaVA, Gemini-ego) — the original label parser mis-scored negated/hedged/disjunctive sentences ("not yet a Late Expert" being read as "Late Expert"). All reasoning-file numbers below use the pipeline's negation-aware extractor.
2. **`gemini_climbing_entire_binary.csv` duplicate row** — one clip (`97b78075…`) had a stale failed retry left in the file alongside its successful retry. Deduplicated (kept the successful retry) before scoring — see §1.1.

**Video-input method — two paradigms, not one:** Qwen2.5-VL and VideoLLaVA use **fixed frame-count sampling** (8/16/32/64 frames extracted uniformly regardless of clip length). Gemini and Qwen3-VL are the two models run at a **clip-duration-proportional ~1fps rate** instead — but only Gemini's numbers come from genuine native video ingestion (raw file uploaded directly to the API); Qwen3-VL's native pipeline hit an fps=24 OOM bug and was abandoned after 12/50 clips, so its reported "frames" numbers come from a manual OpenCV substitute built to reproduce the same ~1fps target rate by hand, not from true native ingestion. See `PROJECT_REPORT.md` §"Two distinct video-input paradigms" for the full detail. Practical implication: frame-count ablation (8 vs 16 vs 32 vs 64) only exists for Qwen2.5-VL/VideoLLaVA — Gemini and Qwen3-VL were never run at a fixed frame count at all.

---

## Table of contents
1. [Climbing (`diss_climb`)](#1-climbing-diss_climb)
2. [Dance (`diss_dance`)](#2-dance-diss_dance)
3. [Cross-domain generalization](#3-cross-domain-generalization)
4. [Label-granularity ablation (3-class)](#4-label-granularity-ablation-3-class)
5. [What was excluded, and why](#5-what-was-excluded-and-why)
6. [Text-only commentary control](#6-text-only-commentary-control)
7. [Extraction-bug re-verification method](#7-extraction-bug-re-verification-method)

---

## 1. Climbing (`diss_climb`)

### 1.1 Gemini

*Unchanged from the 2026-08-06 version — reproduced exactly by the fresh re-parse. No corrections needed.*

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
- Overall: 26/100 = **26.0%**
- Novice: 5/25 = 20.0% · Early Expert: 0/25 · Intermediate Expert: 20/25 = 80.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 79, Novice 14, Late Expert 6, Unknown 1
- File: `diss_climb/results/gemini/gemini_climbing_entire_reasoning_ego.csv`

**Reasoning — exo**
- Overall: 28/100 = **28.0%**
- Novice: 3/25 = 12.0% · Early Expert: 0/25 · Intermediate Expert: 24/25 = 96.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 90, Novice 6, Late Expert 3, Early Expert 1
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

> Note: Gemini never once predicts Early Expert or Late Expert with meaningful frequency on climbing — every fourclass/reasoning/structured run collapses the 4-class scale to an effective Novice-vs-Intermediate-Expert judgment. Kept because it still resolves 2–3 of the 4 classes with real signal.

> **Correction vs. 2026-08-06:** the reasoning-ego numbers above (26.0%, not 28.0%) differ from the prior version by one clip's worth of accuracy after re-parsing; reasoning-exo is unchanged. Minor, does not change any qualitative conclusion.

---

### 1.2 Qwen2.5-VL-7B

#### Collapsed — binary (all conditions; shown in full, excluded from headline claims)

*Unchanged from 2026-08-06.* Every binary-prompt climbing run predicts "Novice" for effectively 100% of clips, regardless of frame count, trim condition, or view. Accuracy is exactly 50% in every case = chance by construction (Novice is exactly half the ground truth).

| Frames | Trim | View | Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| 8 | entire | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 48, Unknown 2 | `diss_climb/results/qwen/qwen_climbing_entire_n8_binary.csv` |
| 8 | entire | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 48, Unknown 2 | `diss_climb/results/qwen/qwen_climbing_entire_n8_binary.csv` |
| 16 | entire | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 48, Unknown 2 | `diss_climb/results/qwen/qwen_climbing_entire_n16_binary.csv` |
| 16 | entire | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 48, Unknown 2 | `diss_climb/results/qwen/qwen_climbing_entire_n16_binary.csv` |
| 8 | trimmed | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_ego_n8_binary.csv` |
| 8 | trimmed | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_exo_n8_binary.csv` |
| 16 | trimmed | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_ego_n16_binary.csv` |
| 16 | trimmed | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_exo_n16_binary.csv` |

#### Collapsed — structured, entire n8, both views (separate run from `structured_eval.csv` below)

*Unchanged from 2026-08-06.*

| View | Overall | Novice acc | Other classes | Predicted counts | File |
|---|---|---|---|---|---|
| ego | 5/100 = 5.0% | 5/25 = 20.0% | all 0/25 | Unknown 73, Novice 21, Intermediate Expert 6 | `diss_climb/results/qwen/qwen_climbing_entire_n8_structured.csv` |
| exo | 14/100 = 14.0% | 14/25 = 56.0% | all 0/25 | Novice 68, Unknown 31, Intermediate Expert 1 | `diss_climb/results/qwen/qwen_climbing_entire_n8_structured.csv` |

#### Collapsed — fourclass/reasoning, ego view (updated numbers)

Ego-view fourclass/reasoning collapses to (near-)100% Novice, zero accuracy on every other class, at both frame counts and both trim conditions. **The `_patched` re-extraction file used in the 2026-08-06 version for one exo run is now redundant** — the unified parser resolves the un-patched `qwen_climbing_entire_n16_reasoning.csv` directly (see Kept, below) — so it is no longer cited separately.

| Prompt | Frames | Trim | Overall | Novice acc | Other classes | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| Fourclass | 16 | entire | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 98, Intermediate Expert 2 | `diss_climb/results/qwen/qwen_climbing_entire_n16_fourclass.csv` |
| Reasoning | 8 | entire | 24/100 = 24.0% | 24/25 = 96.0% | all 0/25 | Novice 90, Unknown 9, Intermediate Expert 1 | `diss_climb/results/qwen/qwen_climbing_entire_n8_reasoning.csv` |
| Reasoning | 16 | entire | 24/100 = 24.0% | 24/25 = 96.0% | all 0/25 | Novice 90, Unknown 9, Early Expert 1 | `diss_climb/results/qwen/qwen_climbing_entire_n16_reasoning.csv` |
| Reasoning | 8 | trimmed | 23/100 = 23.0% | 23/25 = 92.0% | all 0/25 | Novice 89, Unknown 10, Intermediate Expert 1 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n8_reasoning.csv` |
| Reasoning | 16 | trimmed | 23/100 = 23.0% | 23/25 = 92.0% | all 0/25 | Novice 89, Unknown 10, Intermediate Expert 1 | `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n16_reasoning.csv` |

#### Kept — fourclass / reasoning / structured (exo view, plus both views of structured)

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

**Fourclass — 16 frames, entire, exo**
- Overall: 27/100 = **27.0%**
- Novice: 23/25 = 92.0% · Early Expert: 0/25 · Intermediate Expert: 4/25 = 16.0% · Late Expert: 0/25
- Predicted counts: Novice 89, Intermediate Expert 11
- File: `diss_climb/results/qwen/qwen_climbing_entire_n16_fourclass.csv`

**Fourclass — 8 frames, trimmed, exo/ego** *(`fourclass_trimmed.csv` — frame count ambiguous, both scripts write the same path; treat with caution)*
- ego: Overall 26/100 = **26.0%** — Novice 24/25=96.0%, Intermediate Expert 2/25=8.0%, others 0/25. Predicted counts: Novice 93, Intermediate Expert 5, Unknown 2
- exo: Overall 24/100 = **24.0%** — Novice 22/25=88.0%, Intermediate Expert 2/25=8.0%, others 0/25. Predicted counts: Novice 88, Intermediate Expert 10, Unknown 2
- File: `diss_climb/results/qwen/trimmed/fourclass_trimmed.csv`

**Reasoning — 8 frames, entire, exo**
- Overall: 18/100 = **18.0%**
- Novice: 12/25 = 48.0% · Early Expert: 3/25 = 12.0% · Intermediate Expert: 3/25 = 12.0% · Late Expert: 0/25
- Predicted counts: Novice 69, Intermediate Expert 16, Unknown 9, Early Expert 6
- File: `diss_climb/results/qwen/qwen_climbing_entire_n8_reasoning.csv`

**Reasoning — 16 frames, entire, exo** *(now resolves all four classes directly, no patched variant needed)*
- Overall: 18/100 = **18.0%**
- Novice: 6/25 = 24.0% · Early Expert: 4/25 = 16.0% · Intermediate Expert: 8/25 = 32.0% · Late Expert: 0/25
- Predicted counts: Novice 48, Intermediate Expert 35, Unknown 9, Early Expert 7, Late Expert 1
- File: `diss_climb/results/qwen/qwen_climbing_entire_n16_reasoning.csv`

**Reasoning — 8 frames, trimmed, exo**
- Overall: 16/100 = **16.0%**
- Novice: 7/25 = 28.0% · Early Expert: 4/25 = 16.0% · Intermediate Expert: 5/25 = 20.0% · Late Expert: 0/25
- Predicted counts: Novice 55, Intermediate Expert 29, Unknown 10, Early Expert 6
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n8_reasoning.csv`

**Reasoning — 16 frames, trimmed, exo**
- Overall: 23/100 = **23.0%**
- Novice: 3/25 = 12.0% · Early Expert: 12/25 = 48.0% · Intermediate Expert: 8/25 = 32.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 36, Novice 35, Early Expert 19, Unknown 10
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n16_reasoning.csv`

**Structured — 8 frames, entire, ego** *(separate run from the canonical-grid file above — same clip set, no `Skill Level:` template collapse)*
- Overall: 19/100 = **19.0%**
- Novice: 16/25 = 64.0% · Early Expert: 0/25 · Intermediate Expert: 3/25 = 12.0% · Late Expert: 0/25
- Predicted counts: Novice 71, Intermediate Expert 20, Unknown 9
- File: `diss_climb/results/qwen/structured_eval.csv`

**Structured — 8 frames, entire, exo**
- Overall: 24/100 = **24.0%**
- Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 2/25 = 8.0% · Late Expert: 0/25
- Predicted counts: Novice 87, Unknown 9, Intermediate Expert 4
- File: `diss_climb/results/qwen/structured_eval.csv`

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

**Structured — 8 frames, trimmed, ego**
- Overall: 19/100 = **19.0%**
- Novice: 13/25 = 52.0% · Early Expert: 0/25 · Intermediate Expert: 6/25 = 24.0% · Late Expert: 0/25
- Predicted counts: Novice 74, Intermediate Expert 24, Unknown 2
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n8_structured.csv`

**Structured — 8 frames, trimmed, exo**
- Overall: 22/100 = **22.0%**
- Novice: 18/25 = 72.0% · Early Expert: 0/25 · Intermediate Expert: 4/25 = 16.0% · Late Expert: 0/25
- Predicted counts: Novice 83, Intermediate Expert 15, Unknown 2
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n8_structured.csv`

**Structured — 16 frames, trimmed, ego**
- Overall: 26/100 = **26.0%**
- Novice: 8/25 = 32.0% · Early Expert: 0/25 · Intermediate Expert: 18/25 = 72.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 59, Novice 39, Unknown 2
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n16_structured.csv`

**Structured — 16 frames, trimmed, exo**
- Overall: 18/100 = **18.0%**
- Novice: 9/25 = 36.0% · Early Expert: 0/25 · Intermediate Expert: 9/25 = 36.0% · Late Expert: 0/25
- Predicted counts: Novice 53, Intermediate Expert 43, Unknown 4
- File: `diss_climb/results/qwen/trimmed/qwen_climbing_trimmed_n16_structured.csv`

> **Corrections vs. 2026-08-06 in this subsection:** reasoning-exo numbers moved at every frame/trim combination (typically ±1–5 points; largest: trimmed n16 exo 27%→23%, structured trimmed n16 exo 17%→18%). Fourclass/structured numbers are essentially unchanged. The "patched re-extraction" special case for n16-entire-exo reasoning no longer exists as a separate artefact — the unified parser resolves the original file into the same 4-class-spread result directly.

---

#### 1.2.4 New: per-clip best-exocentric-camera check (`bestexo`)

*New since 2026-08-07.* Rather than fixing the exo view to `cam01`, this run selects the take's annotated `best_exo` camera per clip (distribution: cam03 ×23, cam02 ×21, cam01 ×5, cam04 ×1) to test whether the earlier exo results were being handicapped by a suboptimal fixed camera choice. 8 frames, entire clip, binary + reasoning only (no fourclass/structured run).

##### Collapsed — binary

| Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|
| 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/qwen_climbing_bestexo_n8_binary.csv` |

Identical collapse to every fixed-camera exo binary run above — per-clip optimal camera selection does not rescue the binary prompt.

##### Kept — reasoning

**Reasoning — best-exo**
- Overall: 24/100 = **24.0%**
- Novice: 9/25 = 36.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 14/25 = 56.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 58, Novice 26, Unknown 9, Early Expert 6, Late Expert 1
- File: `diss_climb/results/qwen/qwen_climbing_bestexo_n8_reasoning.csv`

This sits within the same 16–23% band as the fixed-`cam01` exo reasoning runs above — camera *selection* (as opposed to camera *placement*, tested separately in §3 via cam03) does not measurably change the failure mode.

#### 1.2.5 New: 64-frame condition (`n64`)

*New since 2026-08-07, extends the frame-count ablation beyond the 8/16 pair covered by the canonical pipeline (32-frame was already out of scope — see §5).* Entire clip, climbing only.

##### Collapsed — binary

| View | Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|
| ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/qwen_climbing_entire_n64_binary.csv` |
| exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen/qwen_climbing_entire_n64_binary.csv` |

Binary still collapses to 100% Novice at 64 frames — the frame-count ablation's core finding (more frames does not fix the binary collapse) extends to 8×.

##### Kept — fourclass / reasoning / structured

**Fourclass — 64 frames, ego**
- Overall: 26/100 = **26.0%**
- Novice: 25/25 = 100.0% · Early Expert: 0/25 · Intermediate Expert: 1/25 = 4.0% · Late Expert: 0/25
- Predicted counts: Novice 97, Intermediate Expert 3
- File: `diss_climb/results/qwen/qwen_climbing_entire_n64_fourclass.csv`

**Fourclass — 64 frames, exo**
- Overall: 25/100 = **25.0%**
- Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 3/25 = 12.0% · Late Expert: 0/25
- Predicted counts: Novice 85, Intermediate Expert 15
- File: `diss_climb/results/qwen/qwen_climbing_entire_n64_fourclass.csv`

**Reasoning — 64 frames, ego**
- Overall: 29/100 = **29.0%** — best 64-frame result
- Novice: 10/25 = 40.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 18/25 = 72.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 53, Novice 42, Early Expert 4, Late Expert 1
- File: `diss_climb/results/qwen/qwen_climbing_entire_n64_reasoning.csv`

**Reasoning — 64 frames, exo**
- Overall: 32/100 = **32.0%** — highest single Qwen2.5-VL climbing accuracy in the project, at 8× the frames used everywhere else
- Novice: 0/25 · Early Expert: 11/25 = 44.0% · Intermediate Expert: 20/25 = 80.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 66, Early Expert 25, Novice 6, Late Expert 3
- File: `diss_climb/results/qwen/qwen_climbing_entire_n64_reasoning.csv`

**Structured — 64 frames, ego**
- Overall: 20/100 = **20.0%**
- Novice: 0/25 · Early Expert: 0/25 · Intermediate Expert: 19/25 = 76.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 86, Novice 10, Late Expert 4
- File: `diss_climb/results/qwen/qwen_climbing_entire_n64_structured.csv`

**Structured — 64 frames, exo**
- Overall: 21/100 = **21.0%**
- Novice: 3/25 = 12.0% · Early Expert: 0/25 · Intermediate Expert: 18/25 = 72.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 79, Novice 20, Late Expert 1
- File: `diss_climb/results/qwen/qwen_climbing_entire_n64_structured.csv`

**Reading this against the 8/16-frame ablation:** the highest overall number in the whole 8/16/64-frame family is the 64-frame reasoning-exo run (32%, vs. 18% at both 8 and 16 frames) — the one genuinely interesting frame-count effect in the project, and it's on the free-text prompt, not fourclass/structured (which stay flat at 20–26% same as 8/16 frames). Note the collapse *pattern* also shifts at 64 frames: reasoning-exo goes from a Novice-leaning collapse at 8/16 frames to a Novice-*never*-predicted, Early/Intermediate-dominant one — more frames changes what the model defaults to, not whether it defaults.

#### 1.2.6 New: large-scale generalization check (`benchmark_400`, n=389)

*New since 2026-08-08.* The canonical grid uses 25 clips/class (100 total, four-class). This run rebuilds the benchmark at ~100 clips/class (389 total — Late Expert capped at 89 by data availability rather than the intended 100) to check whether the small-n findings above are a sampling artefact. 8 frames, entire clip, reasoning + structured only (no binary/fourclass at this scale).

##### Collapsed — reasoning, ego

| Overall | Novice acc | Other classes | Predicted counts | File |
|---|---|---|---|---|
| 99/389 = 25.4% | 99/100 = 99.0% | all 0/100 or 0/89 | Novice 382, Intermediate Expert 5, Early Expert 2 | `diss_climb/results/qwen/qwen_climbing_entire_n8_reasoning_400.csv` |

##### Kept

**Reasoning — 8 frames, entire, exo, n=389**
- Overall: 65/389 = **16.7%**
- Novice: 45/100 = 45.0% · Early Expert: 7/100 = 7.0% · Intermediate Expert: 13/100 = 13.0% · Late Expert: 0/89 = 0.0%
- Predicted counts: Novice 283, Intermediate Expert 80, Early Expert 24, Late Expert 2
- File: `diss_climb/results/qwen/qwen_climbing_entire_n8_reasoning_400.csv`

**Structured — 8 frames, entire, ego, n=389**
- Overall: 99/389 = **25.4%**
- Novice: 71/100 = 71.0% · Early Expert: 0/100 · Intermediate Expert: 28/100 = 28.0% · Late Expert: 0/89
- Predicted counts: Novice 284, Intermediate Expert 103, Unknown 2
- File: `diss_climb/results/qwen/qwen_climbing_entire_n8_structured_400.csv`

**Structured — 8 frames, entire, exo, n=389**
- Overall: 91/389 = **23.4%**
- Novice: 87/100 = 87.0% · Early Expert: 0/100 · Intermediate Expert: 4/100 = 4.0% · Late Expert: 0/89
- Predicted counts: Novice 359, Intermediate Expert 28, Unknown 2
- File: `diss_climb/results/qwen/qwen_climbing_entire_n8_structured_400.csv`

**What this confirms:** the same ego-collapses/exo-partially-resolves pattern seen throughout the 25-clip/class grid reproduces almost exactly at ~4× the sample size (reasoning-exo 16.7% here vs. 18.0% at n=100; structured-exo 23.4% here vs. 24.0% at n=100) — the small-n findings in the rest of this document are not a sampling artefact.

---

### 1.3 VideoLLaVA

#### Collapsed / failed — binary, fourclass, structured-16, structured-trimmed-16, reasoning-trimmed-16 (shown in full, excluded from headline claims)

| Prompt | Frames | Trim | View | Overall | Class breakdown | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| Binary | 8 | entire | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 48, Unknown 2 | `diss_climb/results/videollava/vl_climbing_entire_n8_binary.csv` |
| Binary | 8 | entire | exo | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 48, Unknown 2 | `diss_climb/results/videollava/vl_climbing_entire_n8_binary.csv` |
| Binary | 8 | trimmed | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_ego_n8_binary.csv` |
| Binary | 16 | trimmed | ego | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_ego_n16_binary.csv` |
| Binary | 16 | trimmed | exo | 25/50 = 50.0% | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_exo_n16_binary.csv` |
| Fourclass | 8 | entire | ego | 24/100 = 24.0% | Novice 24/25=96.0%, others 0/25 | Novice 91, Unknown 9 | `diss_climb/results/videollava/vl_climbing_entire_n8_fourclass.csv` |
| Fourclass | 8 | entire | exo | 24/100 = 24.0% | Novice 24/25=96.0%, others 0/25 | Novice 91, Unknown 9 | `diss_climb/results/videollava/vl_climbing_entire_n8_fourclass.csv` |
| Fourclass | 8 | trimmed | ego | 24/100 = 24.0% | Novice 24/25=96.0%, others 0/25 | Novice 98, Unknown 2 | `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_fourclass.csv` |
| Structured | 16 | entire | ego+exo | 0/100 = 0.0% both | total generation failure | Unknown 100 both | `diss_climb/results/videollava/structured_n16_vl.csv` |
| Fourclass | 16 | trimmed | ego+exo | 2/100 = 2.0% both | Novice 2/25=8.0%, others 0/25 | Unknown 95/83, Novice 5/17 | `diss_climb/results/videollava/trimmed/fourclass_n16.csv` |
| Reasoning | 16 | trimmed | ego+exo | 0/100 = 0.0% both | total generation failure | Unknown 100 both | `diss_climb/results/videollava/trimmed/reasoning_n16.csv` |

*(16-frame structured/reasoning failures and the trimmed-16 fourclass near-failure all trace to the same 4096-token context overflow — see §5.)*

#### Kept — the surviving 16-frame binary, structured, and reasoning results

**Binary — 16 frames, entire, ego**
- Overall: 16/50 = **32.0%** *(below chance)*
- Novice: 12/25 = 48.0% · Expert: 4/25 = 16.0%
- Predicted counts: Novice 31, Expert 17, Unknown 2
- File: `diss_climb/results/videollava/vl_climbing_entire_n16_binary.csv`

**Binary — 16 frames, entire, exo**
- Overall: 38/50 = **76.0%** — best single result in the climbing arm
- Novice: 22/25 = 88.0% · Expert: 16/25 = 64.0%
- Predicted counts: Novice 29, Expert 19, Unknown 2
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
- Cam01 (main file): 16/100 = **16.0%** — Novice 1/25=4.0%, Early Expert 1/25=4.0%, Intermediate Expert 10/25=40.0%, Late Expert 4/25=16.0%
  Predicted counts: Intermediate Expert 42, Unknown 32, Late Expert 16, Early Expert 5, Novice 5
  File: `diss_climb/results/videollava/vl_climbing_entire_n8_reasoning_fixed.csv`
- Cam03 (camera-angle ablation — see §3): 15/100 = **15.0%** — Novice 1/25=4.0%, Intermediate Expert 9/25=36.0%, Late Expert 5/25=20.0%, Early Expert 0/25
  Predicted counts: Intermediate Expert 40, Unknown 33, Late Expert 18, Novice 6, Early Expert 3
  File: `diss_climb/results/videollava/reasoning_n8_test03.csv`

**Reasoning — 8 frames, entire, exo** *(two independent runs)*
- Cam01 (main file): 21/100 = **21.0%** — Intermediate Expert 20/25=80.0%, Late Expert 1/25=4.0%, Novice 0/25, Early Expert 0/25
  Predicted counts: Intermediate Expert 77, Late Expert 14, Unknown 9
  File: `diss_climb/results/videollava/vl_climbing_entire_n8_reasoning_fixed.csv`
- Cam03 (ablation): 21/100 = **21.0%** — Early Expert 2/25=8.0%, Intermediate Expert 11/25=44.0%, Late Expert 8/25=32.0%, Novice 0/25
  Predicted counts: Intermediate Expert 44, Late Expert 30, Unknown 19, Early Expert 5, Novice 2
  File: `diss_climb/results/videollava/reasoning_n8_test03.csv`

**Reasoning — 8 frames, trimmed, ego** *(⚠ status change from 2026-08-06: now Kept, was Collapsed)*
- Overall: 21/100 = **21.0%**
- Novice: 0/25 · Early Expert: 0/25 · Intermediate Expert: 18/25 = 72.0% · Late Expert: 3/25 = 12.0%
- Predicted counts: Intermediate Expert 72, Late Expert 17, Unknown 11
- File: `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_reasoning.csv`
- The 2026-08-06 version reported this run as an 84%-Intermediate-Expert single-class collapse. Re-parsing with the unified extractor recovers 3 correct Late-Expert answers the hand extractor missed, which is enough under the exclusion rule (two ground-truth classes now show non-zero accuracy) to move this out of Collapsed. Treat as a low-confidence Kept result — it barely clears the bar and the underlying signal (Intermediate Expert dominating predictions regardless of true class) hasn't changed.

**Reasoning — 8 frames, trimmed, exo**
- Overall: 22/100 = **22.0%**
- Intermediate Expert: 20/25=80.0%, Late Expert: 2/25=8.0%, Novice: 0/25, Early Expert: 0/25
- Predicted counts: Intermediate Expert 81, Unknown 10, Late Expert 9
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
- Novice 15/25=60.0%, Early Expert 1/25=4.0%, Intermediate Expert 4/25=16.0%, Late Expert 10/25=40.0%
- Predicted counts: Novice 55, Late Expert 29, Intermediate Expert 10, Early Expert 4, Unknown 2
- File: `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_structured.csv`

**Structured — 8 frames, trimmed, exo**
- Overall: 15/100 = **15.0%**
- Novice 7/25=28.0%, Intermediate Expert 5/25=20.0%, Late Expert 3/25=12.0%, Early Expert 0/25
- Predicted counts: Novice 58, Intermediate Expert 18, Late Expert 14, Early Expert 8, Unknown 2
- File: `diss_climb/results/videollava/trimmed/vl_climbing_trimmed_n8_structured.csv`

> **Corrections vs. 2026-08-06 in this subsection:** reasoning numbers shifted at every entry (entire-ego 13%→16%, entire-exo 23%→21%, cam03-exo 19%→21%, cam03-ego 14%→15%, trimmed-exo 24%→22%), and **trimmed-ego reasoning moves from Collapsed to Kept** (see above). Binary/fourclass/structured numbers are unchanged.

---

### 1.4 Qwen3-VL-8B

#### Collapsed — native video-input attempt (shown in full, excluded from headline claims)

*Unchanged from 2026-08-06.* Native mode targeted the same ~1fps rate as Gemini (`n = round(duration_sec)` frames requested directly from the model's own video processor via `{"type": "video", "video": video_path, "nframes": n}`), but hit an fps=24 fallback bug that ignored the requested frame count and caused OOM. Only 12 of 50 climbing clips completed before the run was abandoned; no dance native attempt was made at all.

| View | Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|
| exo | 6/12 = 50.0% | 6/6 = 100.0% | 0/6 = 0.0% | Novice 12 | `diss_climb/results/qwen3vl/qwen3vl_climbing_native_binary.csv` |
| ego | 6/12 = 50.0% | 6/6 = 100.0% | 0/6 = 0.0% | Novice 12 | `diss_climb/results/qwen3vl/qwen3vl_climbing_native_binary.csv` |

*(fourclass/reasoning/structured native attempts produced empty output files — not tabulated.)*

#### Collapsed — frame-extraction substitute, binary ego (shown in full)

*Unchanged.*

| View | Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|
| ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_binary.csv` |

#### Kept

*Fourclass and structured numbers below are unchanged from 2026-08-06; reasoning numbers shifted slightly (22%→23% both views).*

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
- Overall: 37/100 = **37.0%** — best fourclass climbing result of any model tested (still short of the 64-frame Qwen2.5-VL reasoning-exo result at 32% on a harder task — this remains the best *fourclass-specifically* result)
- Novice: 23/25 = 92.0% · Early Expert: 0/25 · Intermediate Expert: 14/25 = 56.0% · Late Expert: 0/25
- Predicted counts: Novice 69, Intermediate Expert 31
- File: `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_fourclass.csv`

**Reasoning — exo**
- Overall: 23/100 = **23.0%**
- Novice: 6/25 = 24.0% · Early Expert: 2/25 = 8.0% · Intermediate Expert: 13/25 = 52.0% · Late Expert: 2/25 = 8.0%
- Predicted counts: Intermediate Expert 50, Novice 30, Unknown 10, Late Expert 7, Early Expert 3
- File: `diss_climb/results/qwen3vl/qwen3vl_climbing_frames_reasoning.csv`

**Reasoning — ego**
- Overall: 23/100 = **23.0%**
- Novice: 6/25 = 24.0% · Early Expert: 2/25 = 8.0% · Intermediate Expert: 13/25 = 52.0% · Late Expert: 2/25 = 8.0%
- Predicted counts: Intermediate Expert 46, Unknown 26, Novice 17, Late Expert 7, Early Expert 4
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

*Unchanged from the 2026-08-06 version — reproduced exactly.*

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
- Overall: 23/100 = **23.0%**
- Novice: 14/25 = 56.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 7/25 = 28.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Novice 68, Intermediate Expert 24, Early Expert 4, Late Expert 2, Unknown 2
- File: `diss_dance/results/gemini/gemini_dance_entire_reasoning_ego.csv`

**Reasoning — exo**
- Overall: 32/100 = **32.0%** — Gemini's best dance result
- Novice: 7/25 = 28.0% · Early Expert: 1/25 = 4.0% · Intermediate Expert: 23/25 = 92.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 78, Novice 16, Early Expert 4, Late Expert 2
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

> **Correction vs. 2026-08-06:** reasoning-ego moves from 21.0% to 23.0% (predicted counts also gain a Late Expert 2 previously mis-parsed as something else). All other Gemini-dance numbers are unchanged.

---

### 2.2 Qwen2.5-VL-7B

#### Collapsed — binary (all conditions; shown in full, excluded from headline claims)

*Unchanged.* Identical failure mode to climbing: 100% "Novice" regardless of frame count, trim condition, or view.

| Frames | Trim | View | Overall | Novice acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| 8 | entire | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n8_binary.csv` |
| 8 | entire | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n8_binary.csv` |
| 16 | entire | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n16_binary.csv` |
| 16 | entire | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n16_binary.csv` |
| 8 | trimmed | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_binary.csv` |
| 8 | trimmed | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_binary.csv` |
| 16 | trimmed | ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_binary.csv` |
| 16 | trimmed | exo | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_binary.csv` |

#### Collapsed — fourclass/reasoning, ego view (shown in full)

*Unchanged.* Every ego-view fourclass and reasoning run collapses to 100% Novice with zero accuracy on all other classes; only the exo columns of these same files (kept below) show real discrimination.

| Prompt | Frames | Trim | Overall | Novice acc | Other classes | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| Fourclass | 8 | entire | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/qwen_dance_entire_n8_fourclass.csv` |
| Fourclass | 16 | entire | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/qwen_dance_entire_n16_fourclass.csv` |
| Fourclass | 8 | trimmed | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_fourclass.csv` |
| Fourclass | 16 | trimmed | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_fourclass.csv` |
| Reasoning | 8 | entire | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 99, Intermediate Expert 1 | `diss_dance/results/qwen/qwen_dance_entire_n8_reasoning.csv` |
| Reasoning | 16 | entire | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/qwen_dance_entire_n16_reasoning.csv` |
| Reasoning | 8 | trimmed | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_reasoning.csv` |
| Reasoning | 16 | trimmed | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_reasoning.csv` |

#### Kept — exo-view fourclass / reasoning, and structured (both views)

**Fourclass — 8 frames, entire, exo**
- Overall: 31/100 = **31.0%**
- Novice: 18/25 = 72.0% · Early Expert: 0/25 · Intermediate Expert: 13/25 = 52.0% · Late Expert: 0/25
- Predicted counts: Novice 61, Intermediate Expert 39
- File: `diss_dance/results/qwen/qwen_dance_entire_n8_fourclass.csv`

**Fourclass — 16 frames, entire, exo**
- Overall: 33/100 = **33.0%**
- Novice: 16/25 = 64.0% · Early Expert: 0/25 · Intermediate Expert: 17/25 = 68.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 51, Novice 49
- File: `diss_dance/results/qwen/qwen_dance_entire_n16_fourclass.csv`

**Fourclass — 8 frames, trimmed, exo**
- Overall: 32/100 = **32.0%**
- Novice: 16/25 = 64.0% · Early Expert: 0/25 · Intermediate Expert: 16/25 = 64.0% · Late Expert: 0/25
- Predicted counts: Novice 57, Intermediate Expert 43
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_fourclass.csv`

**Fourclass — 16 frames, trimmed, exo**
- Overall: 34/100 = **34.0%**
- Novice: 16/25 = 64.0% · Early Expert: 0/25 · Intermediate Expert: 18/25 = 72.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 55, Novice 45
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_fourclass.csv`

**Reasoning — 8 frames, entire, exo**
- Overall: 32/100 = **32.0%**
- Novice: 10/25 = 40.0% · Early Expert: 0/25 · Intermediate Expert: 21/25 = 84.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 66, Novice 30, Late Expert 4
- File: `diss_dance/results/qwen/qwen_dance_entire_n8_reasoning.csv`

**Reasoning — 16 frames, entire, exo**
- Overall: 26/100 = **26.0%**
- Novice: 0/25 · Early Expert: 0/25 · Intermediate Expert: 25/25 = 100.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 96, Late Expert 3, Novice 1
- File: `diss_dance/results/qwen/qwen_dance_entire_n16_reasoning.csv`

**Reasoning — 8 frames, trimmed, exo**
- Overall: 27/100 = **27.0%**
- Novice: 11/25 = 44.0% · Early Expert: 0/25 · Intermediate Expert: 14/25 = 56.0% · Late Expert: 2/25 = 8.0%
- Predicted counts: Intermediate Expert 56, Novice 40, Late Expert 4
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n8_reasoning.csv`

**Reasoning — 16 frames, trimmed, exo**
- Overall: 24/100 = **24.0%**
- Novice: 1/25 = 4.0% · Early Expert: 0/25 · Intermediate Expert: 23/25 = 92.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 91, Late Expert 6, Novice 3
- File: `diss_dance/results/qwen/trimmed/qwen_dance_trimmed_n16_reasoning.csv`

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

**Structured — 16 frames, entire, ego**
- Overall: 23/100 = **23.0%**
- Novice: 14/25 = 56.0% · Early Expert: 0/25 · Intermediate Expert: 9/25 = 36.0% · Late Expert: 0/25
- Predicted counts: Novice 62, Intermediate Expert 35, Unknown 2, Late Expert 1
- File: `diss_dance/results/qwen/qwen_dance_entire_n16_structured.csv`

**Structured — 16 frames, entire, exo**
- Overall: 41/100 = **41.0%** — best Qwen2.5-VL result across the entire fixed-frame family (8/16/64)
- Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 19/25 = 76.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 57, Novice 43
- File: `diss_dance/results/qwen/qwen_dance_entire_n16_structured.csv`

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

> **Corrections vs. 2026-08-06 in this subsection:** fourclass/structured numbers are unchanged. Reasoning-exo numbers shifted at the trimmed conditions only (trimmed n8: 28%→27%; trimmed n16: 27%→24%); entire-condition reasoning-exo (32%/26%) is unchanged. No bestexo dance runs exist — that addition is climbing-only (§1.2.4). The 64-frame condition *was* climbing-only as of 2026-08-13; dance n64 results were added 2026-08-18 and are covered in §2.2.4 below.

#### 2.2.4 New (2026-08-18): 64-frame condition (`n64`)

*Mirrors §1.2.5 for dance — extends the frame-count ablation beyond the 8/16 pair to 64 frames.* Entire clip, dance only, Qwen2.5-VL.

##### Collapsed — binary, and ego-view fourclass/reasoning

| Prompt | View | Overall | Novice acc | Other classes | Predicted counts | File |
|---|---|---|---|---|---|---|
| Binary | ego | 25/50 = 50.0% | 25/25 = 100.0% | Expert 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n64_binary.csv` |
| Binary | exo | 25/50 = 50.0% | 25/25 = 100.0% | Expert 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen/qwen_dance_entire_n64_binary.csv` |
| Fourclass | ego | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/qwen_dance_entire_n64_fourclass.csv` |
| Reasoning | ego | 25/100 = 25.0% | 25/25 = 100.0% | all 0/25 | Novice 100 | `diss_dance/results/qwen/qwen_dance_entire_n64_reasoning.csv` |

Same failure mode as climbing at 64 frames: binary and ego-view fourclass/reasoning collapse to 100% Novice regardless of frame count.

##### Kept — exo-view fourclass/reasoning, and structured (both views)

**Fourclass — 64 frames, exo**
- Overall: 31/100 = **31.0%**
- Novice: 23/25 = 92.0% · Early Expert: 0/25 · Intermediate Expert: 8/25 = 32.0% · Late Expert: 0/25
- Predicted counts: Novice 81, Intermediate Expert 19
- File: `diss_dance/results/qwen/qwen_dance_entire_n64_fourclass.csv`

**Reasoning — 64 frames, exo**
- Overall: 20/100 = **20.0%**
- Novice: 9/25 = 36.0% · Early Expert: 9/25 = 36.0% · Intermediate Expert: 2/25 = 8.0% · Late Expert: 0/25
- Predicted counts: Early Expert 47, Novice 44, Intermediate Expert 5, Late Expert 4
- File: `diss_dance/results/qwen/qwen_dance_entire_n64_reasoning.csv`

**Structured — 64 frames, ego**
- Overall: 22/83 valid = **26.5%** (83/100 valid — 17 unparseable)
- Novice: 11/25 = 44.0% · Early Expert: 0/25 · Intermediate Expert: 11/25 = 44.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 44, Novice 39
- File: `diss_dance/results/qwen/qwen_dance_entire_n64_structured.csv`

**Structured — 64 frames, exo**
- Overall: 27/100 = **27.0%**
- Novice: 7/25 = 28.0% · Early Expert: 0/25 · Intermediate Expert: 15/25 = 60.0% · Late Expert: 5/25 = 20.0%
- Predicted counts: Intermediate Expert 72, Novice 15, Late Expert 13
- File: `diss_dance/results/qwen/qwen_dance_entire_n64_structured.csv`

**Reading against the 8/16-frame dance ablation:** unlike climbing (where 64-frame reasoning-exo was the project's best single result, 32%), dance's 64-frame numbers do not beat its own 8/16-frame family — best dance fourclass-family result stays the 16-frame structured-exo run at 41% (§2.2), and 64-frame fourclass-exo (31%) and reasoning-exo (20%) both sit at or below their 8/16-frame counterparts (33%/26% and 32%/26% respectively). Frame count does not help dance the way it marginally helped climbing's reasoning prompt.

---

### 2.3 VideoLLaVA

#### Collapsed — binary, fourclass, and reasoning-exo (shown in full, excluded from headline claims)

*Unchanged from 2026-08-06.* Every binary and fourclass run is a pure Novice collapse. Reasoning-exo is also excluded here: unlike reasoning-ego (kept below), it never once predicts anything but Late Expert or Unknown.

| Prompt | Frames | Trim | View | Overall | Class breakdown | Predicted counts | File |
|---|---|---|---|---|---|---|---|
| Binary | 8/16 | entire | ego/exo | 25/50 = 50.0% (all 4) | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/vl_dance_entire_n{8,16}_binary.csv` |
| Binary | 8/16 | trimmed | ego/exo | 25/50 = 50.0% (all 4) | Novice 25/25=100.0%, Expert 0/25=0.0% | Novice 50 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n{8,16}_binary.csv` |
| Fourclass | 8 | entire | ego/exo | 25/100 = 25.0% (both) | Novice 25/25=100.0%, others 0/25 | Novice 100 | `diss_dance/results/videollava/vl_dance_entire_n8_fourclass.csv` |
| Fourclass | 8 | trimmed | ego/exo | 25/100 = 25.0% (both) | Novice 25/25=100.0%, others 0/25 | Novice 100 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_fourclass.csv` |
| Reasoning | 8 | entire | exo | 25/100 = 25.0% | Late Expert 25/25=100.0%, others 0/25 | Late Expert 93, Unknown 4, Intermediate Expert 2, Early Expert 1 | `diss_dance/results/videollava/vl_dance_entire_n8_reasoning.csv` |
| Reasoning | 8 | trimmed | exo | 22/100 = 22.0% | Late Expert 22/25=88.0%, others 0/25 | Late Expert 85, Unknown 9, Novice 3, Early Expert 2, Intermediate Expert 1 | `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_reasoning.csv` |

#### Kept — reasoning-ego and structured (both views)

*All numbers below are unchanged from 2026-08-06.*

**Reasoning — entire, ego**
- Overall: 25/100 = **25.0%**
- Intermediate Expert: 3/25 = 12.0% · Late Expert: 24/25 = 96.0% · Novice: 0/25 · Early Expert: 1/25 = 4.0%
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

*Unchanged.*

| View | Overall | Novice acc | Late Expert acc | Predicted counts | File |
|---|---|---|---|---|---|
| ego | 25/50 = 50.0% | 25/25 = 100.0% | 0/25 = 0.0% | Novice 50 | `diss_dance/results/qwen3vl/qwen3vl_dance_frames_binary.csv` |

#### Kept

*Fourclass and structured numbers below are unchanged from 2026-08-06. Both reasoning entries changed materially — this is the largest revision in the dance arm.*

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

**Reasoning — exo** *(⚠ updated: was 27.0% with an Intermediate-dominant distribution)*
- Overall: 26/100 = **26.0%**
- Novice: 2/25 = 8.0% · Early Expert: 4/25 = 16.0% · Intermediate Expert: 13/25 = 52.0% · Late Expert: 7/25 = 28.0%
- Predicted counts: Intermediate Expert 44, Late Expert 28, Novice 16, Early Expert 11, Unknown 1
- File: `diss_dance/results/qwen3vl/qwen3vl_dance_frames_reasoning.csv`

**Reasoning — ego** *(⚠ updated: was 27.0%)*
- Overall: 26/100 = **26.0%**
- Novice: 19/25 = 76.0% · Early Expert: 0/25 · Intermediate Expert: 5/25 = 20.0% · Late Expert: 2/25 = 8.0%
- Predicted counts: Novice 71, Unknown 16, Intermediate Expert 10, Late Expert 3
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

*All numbers in this section reproduced exactly under re-parsing, with the single exception of the cam03 reasoning ablation (VideoLLaVA), which is flagged below and cross-referenced from §1.3.*

### Collapsed — binary results across all cross-domain tests (shown in full, excluded from headline claims)

All binary-prompt cross-domain tests collapse to (near-)100% "Novice," identical to the climbing/dance binary failure mode.

**Basketball binary — Qwen2.5-VL-7B, 8 frames, entire, ego+exo** *(EgoExo4D "Basketball" scenario; benchmark rebuilt via `rebuild_basketball_benchmark.py` because 16/50 original clips had no video on disk)*
- Overall (both views identical): 12/50 = **24.0%** — below chance
- Novice: 12/25 = 48.0% · "Late Expert" (mapped to Expert): 0/25 = 0.0%
- Predicted counts (both views identical): Novice 34, Unknown 16 — 15 of those 16 unparseable/missing-video rows are on the Expert-labeled clips specifically
- File: `diss_climb/results/qwen/binary_basketball.csv`

**JIGSAWS suturing binary — Qwen2.5-VL-7B, 8 frames, exo only** *(surgical skill, cross-domain generalization test; benchmark filtered to Novice+Expert only from `benchmark/benchmark_jigsaws_suturing.json`, not class-balanced: 19 Novice / 10 Expert)*
- Overall: 19/29 = **65.5%** — reported accuracy is a majority-class artifact, not real discrimination
- Novice: 19/19 = 100.0% · Expert: 0/10 = 0.0%
- Predicted counts: Novice 29 (100% collapse — every single clip predicted Novice)
- File: `results/jigsaws_qwen_binary.csv`

**Mixed-activity binary — Qwen2.5-VL-7B, 8 frames, trimmed, exo/ego** *(activities: Basketball, Music, Cooking, Soccer)*
- Overall (both views identical): 25/100 = **25.0%**
- Novice: 25/25 = 100.0% · Early/Intermediate/Late Expert: 0/25 each = 0.0%
- Predicted counts: Novice 100
- File: `mixed/results/qwen_mixed_trimmed_n8_binary.csv`

**cam03 binary — Qwen2.5-VL-7B, 8 frames, trimmed, exo (alternate `cam03` camera instead of `cam01`)**
- Overall: 25/50 = **50.0%**
- Novice: 25/25 = 100.0% · Expert: 0/25 = 0.0%
- Predicted counts: Novice 50
- File: `diss_climb/results/qwen/trimmed/test_exo3.csv`
- Same failure mode as the equivalent cam01 trimmed-exo binary run — camera angle within "exo" does not change the collapse. (The per-clip best-exo-camera probe in §1.2.4 reaches the same conclusion by a different route.)

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

**Camera-angle ablation (cam03 vs cam01)** — VideoLLaVA reasoning results using the alternate `cam03` exocentric camera (`reasoning_n8_test03.csv`, both runs, listed in full in §1.3): exo 21.0% (⚠ was 19.0%), ego 15.0% (⚠ was 14.0%). Finding unchanged under re-parsing: accuracy remains statistically indistinguishable from the equivalent cam01 reasoning runs (21% and 16% — see §1.3) — camera placement within the "exo" category does not change the failure mode.

---

## 4. Label-granularity ablation (3-class)

Qwen (structured prompt, 8 frames, ego+exo), collapsing the label set to Novice / Intermediate / Expert (Intermediate Expert → "Intermediate", Late Expert → "Expert"; Early Expert excluded from this benchmark by construction, and any Early-Expert-shaped prediction is scored Unknown rather than folded into a class it can't legally answer), 10 clips/class = 30 total.

**⚠ Both rows in this section changed materially under re-parsing — this is the largest correction in the whole document.**

#### Collapsed — exo view (numbers changed)

| View | Overall | Novice acc | Intermediate acc | Expert acc | Predicted counts | File |
|---|---|---|---|---|---|---|
| exo | 9/30 = 30.0% | 0/10 = 0.0% | 1/10 = 10.0% | 8/10 = 80.0% | Expert 22, Unknown 7, Intermediate 1 | `diss_climb/results/qwen/3class_structured.csv` |

The 2026-08-06 version reported Expert recall at 10/10 = 100.0% with 29/30 predictions "Expert"; re-parsing finds 7 of those answers don't actually resolve cleanly to "Expert" once negation/hedging is handled (Expert correct drops to 8/10, predicted-Expert count drops from 29 to 22). The qualitative conclusion is unchanged and, if anything, strengthened: this is still overwhelmingly a one-label response (22/30 raw predictions), still Collapsed on the same prediction-distribution grounds as the 2026-08-06 version — it just isn't quite as clean a collapse as previously reported.

#### Kept — ego view (numbers changed)

**3-class structured — ego**
- Overall: 8/30 = **26.7%**
- Novice: 0/10 = 0.0% · Intermediate: 4/10 = 40.0% · Expert: 4/10 = 40.0%
- Predicted counts: Expert 15, Intermediate 9, Unknown 5, Novice 1
- File: `diss_climb/results/qwen/3class_structured.csv`

The 2026-08-06 version reported Expert recall at 6/10 = 60.0% with only 2 Unknowns; re-parsing drops Expert-correct to 4/10 and triples the Unknown count to 5, because several ego-view answers that superficially mention "Expert" turn out to be hedged or comparative constructions the improved negation handling correctly declines to score. Directionally this makes the 3-class ablation's headline conclusion *stronger*, not weaker: even collapsing to 3 classes, the model resolves fewer of them correctly than previously credited.

---

## 5. What was excluded, and why

*This is a jump-index — every row below now has its full Overall/per-class/predicted-count numbers shown inline in a `#### Collapsed` block in the section noted, not just this summary.*

| Category | Examples | Reason | Full detail in |
|---|---|---|---|
| Qwen2.5-VL binary, climbing & dance (all frame counts, entire+trimmed, both views) | `qwen_climbing_entire_n16_binary.csv`, `qwen_dance_entire_n8_binary.csv`, `binary_n32.csv`, `qwen_climbing_entire_n64_binary.csv`, `qwen_dance_entire_n64_binary.csv`, all `trimmed/*_binary.csv` | 100% "Novice" prediction regardless of input; accuracy exactly 50% = chance by construction | §1.2, §1.2.5, §2.2, §2.2.4 |
| Qwen2.5-VL fourclass/reasoning, ego view (climbing & dance) | `qwen_climbing_entire_n16_fourclass.csv` (ego), `qwen_dance_entire_n16_reasoning.csv` (ego), `qwen_dance_entire_n64_fourclass.csv`/`_reasoning.csv` (ego), etc. | Near-total Novice collapse, zero accuracy on every other class | §1.2, §2.2, §2.2.4 |
| VideoLLaVA binary/fourclass, dance (all) | `vl_dance_entire_n8_binary.csv`, `vl_dance_entire_n16_binary.csv`, `vl_dance_entire_n8_fourclass.csv`, trimmed equivalents | 100% Novice collapse | §2.3 |
| VideoLLaVA reasoning-exo, dance | `vl_dance_entire_n8_reasoning.csv`, `trimmed/vl_dance_trimmed_n8_reasoning.csv` (exo columns) | 88–100% Late Expert collapse (inverse bias to climbing) | §2.3 |
| VideoLLaVA binary, climbing (8fr entire, 32fr, most 16fr trimmed) | `vl_climbing_entire_n8_binary.csv`, `binary_n32.csv`, `trimmed/vl_climbing_trimmed_ego/exo_n16_binary.csv`, `trimmed/vl_climbing_trimmed_ego_n8_binary.csv` | 100% Novice collapse, or (32-frame, both models) total non-response | §1.3 |
| VideoLLaVA fully-failed generations | `structured_n16_vl.csv`, `trimmed/reasoning_n16.csv` | Literal `"..."` / `"ERROR"` in every row — no usable text at all. **Real cause:** fourclass/structured/reasoning at 16 frames fail because VideoLLaVA's own vision-token encoding of 16 frames pushes the prompt over its 4096-token context limit — confirmed by the hard-coded `if n_tokens > 4096` guard in `diss_climb/scripts/videollava/trimmed/reasoning_n16.py:80` and `structured_n16.py:83`, which aborts generation rather than truncating. | §1.3 |
| VideoLLaVA fourclass, climbing (8fr entire, trimmed n16, trimmed n8) | `vl_climbing_entire_n8_fourclass.csv`, `trimmed/fourclass_n16.csv`, `trimmed/vl_climbing_trimmed_n8_fourclass.csv` | 96–98% Novice/Unknown collapse. The trimmed-n16 run shares the same 4096-token root cause as reasoning/structured-n16, but `trimmed/fourclass_n16.py` has no `n_tokens > 4096` guard, so instead of aborting it silently proceeds with an over-length prompt and emits garbage parsed as `Unknown` (95/83 of 100 rows) rather than crashing outright. | §1.3 |
| VideoLLaVA reasoning-16-trimmed, climbing | `trimmed/reasoning_n16.csv` | Total generation failure (same 4096-token cause) | §1.3 |
| Qwen3-VL native video pipeline | `qwen3vl_climbing_native_*.csv` | fps=24 fallback bug caused OOM; only 12/50 clips completed, 100% Novice collapse — approach abandoned | §1.4 |
| Qwen3-VL binary, ego view (climbing & dance) | `qwen3vl_climbing_frames_binary.csv` (ego), `qwen3vl_dance_frames_binary.csv` (ego) | Pure Novice collapse on that view/column only (exo column of same file kept) | §1.4, §2.4 |
| Basketball binary | `diss_climb/results/qwen/binary_basketball.csv` | Only Novice class ever scored correctly (0% on Expert); also 16/50 rows (32%) are unparseable/`ERROR` from missing video files | §3 |
| JIGSAWS suturing binary | `results/jigsaws_qwen_binary.csv` | Model predicted "Novice" for all 29 clips — the reported 65.5% accuracy is entirely a majority-class artifact (19 of 29 ground-truth labels happen to be Novice), not real discrimination | §3 |
| Mixed-activity binary | `mixed/results/qwen_mixed_trimmed_n8_binary.csv` | 100% Novice collapse on both exo and ego | §3 |
| cam03 Qwen binary | `diss_climb/results/qwen/trimmed/test_exo3.csv` | 100% Novice collapse, identical failure mode to the cam01 equivalent | §3 |
| best-exo-camera Qwen binary | `diss_climb/results/qwen/qwen_climbing_bestexo_n8_binary.csv` | 100% Novice collapse — per-clip optimal camera selection doesn't change the binary failure mode | §1.2.4 |
| 64-frame Qwen binary | `diss_climb/results/qwen/qwen_climbing_entire_n64_binary.csv`, `diss_dance/results/qwen/qwen_dance_entire_n64_binary.csv` | 100% Novice collapse, extends the finding to 8× the standard frame count, climbing and dance alike | §1.2.5, §2.2.4 |
| Qwen3-VL dance binary, ego view | `qwen3vl_dance_frames_binary.csv` (ego) | Pure Novice collapse | §2.4 |
| 3-class structured, exo view | `diss_climb/results/qwen/3class_structured.csv` (exo) | 22/30 predictions were "Expert" — a near-total collapse on prediction-distribution grounds even though it technically touches a second class | §4 |
| Gemini duplicate row | `gemini_climbing_entire_binary.csv` | Not excluded, but corrected: stale failed retry for one clip removed before scoring (see §1.1) | §1.1 |
| Text-only commentary binary (Qwen) | `commentary/qwen_binary.py` | Script/SLURM job exist but were never run — no `commentary_binary.csv` output on disk to score | §6 |

---

## 6. Text-only commentary control

*New section — the underlying result files (`commentary_4class.csv`, `videollava_commentary.csv`) predate even the 2026-08-06 version (generated 2026-07-20) but were never written up in this document. A companion `qwen_binary.py`/`.sh` pair for a text-only binary variant was added 2026-08-13 but has not been run (no `commentary_binary.csv` exists on disk — see §5).*

**Question:** given only the expert's written coaching commentary about a climbing attempt — no video, no frames, nothing visual — can a VLM's language backbone recover the climber's skill level from language alone? If text succeeds where video fails, the project's failure mode is specifically about visual grounding; if text also fails, skill level may not be reliably recoverable per-clip in any modality this project tests. Both runs below are single-view (no ego/exo distinction — there is no video), 4-class, 25 clips/class = 100 total.

#### Kept

**Qwen2.5-VL-7B — text-only, 4-class**
- Overall: 26/100 = **26.0%**
- Novice: 1/25 = 4.0% · Early Expert: 7/25 = 28.0% · Intermediate Expert: 18/25 = 72.0% · Late Expert: 0/25
- Predicted counts: Intermediate Expert 67, Early Expert 30, Novice 3
- File: `commentary/commentary_4class.csv`

**VideoLLaVA — text-only, 4-class**
- Overall: 33/100 = **33.0%**
- Novice: 15/25 = 60.0% · Early Expert: 0/25 · Intermediate Expert: 17/25 = 68.0% · Late Expert: 1/25 = 4.0%
- Predicted counts: Intermediate Expert 54, Novice 35, Unknown 10, Late Expert 1
- File: `commentary/videollava_commentary.csv`

**Reading:** both text-only numbers (26%, 33%) sit at or slightly above the *video*-based fourclass/structured numbers for the same models on climbing (Qwen2.5-VL video best ≈27% at n16 fourclass-exo before the n64 result at §1.2.5; VideoLLaVA video best ≈30% structured-trimmed-ego). VideoLLaVA's text-only run is in fact its second-best fourclass-family result of any modality. Text does not fail where video succeeds, nor does it clearly outperform video — the two modalities land in the same ~20–33% band. This is consistent with the project's throughline: the bottleneck is not specifically visual grounding (text-only commentary, describing exactly what would need to be seen, does no better), which points toward a more fundamental limitation in mapping either modality onto this specific four-class proficiency scale, rather than a vision-specific failure. Both models still show the collapse-to-Novice / collapse-to-Intermediate-Expert asymmetry seen throughout the video results — Qwen2.5-VL text-only virtually never predicts Novice (3/100) just as it never predicts it on egocentric video; VideoLLaVA text-only predicts Novice far more (35/100), mirroring its video-side Novice-leaning bias on non-collapsed runs.

**Caveat:** a text-only binary (Novice vs. Expert) variant exists as code (`commentary/qwen_binary.py`) but was never executed — see §5. Running it would be a natural next step to directly compare against the video binary results in §1.1–§1.4, where the binary prompt is the most collapse-prone format.

---

## 7. Extraction-bug re-verification method

The original label parser used for free-text "reasoning" columns (Qwen, VideoLLaVA, and Gemini's ego-view reasoning file) was a naive fixed-priority substring scan that misread negated statements (e.g. "not yet a Late Expert" → wrongly scored as Late Expert) and arbitrarily resolved hedged disjunctions ("Novice or Early Expert" → picked whichever label it checked first). The pipeline's negation-aware extractor (`figures-frame_count/rescore.py::find_mentions`/`parse_fourclass`/`parse_binary`) fixes this by:
1. Scanning all label mentions longest-phrase-first, dropping any mention preceded within 30 characters by a negation/aspiration cue (`not`, `rather than`, `on the way to`, `progressing toward`, etc.).
2. For the structured prompt, preferring the text after the last `Skill Level:` marker if present.
3. When multiple non-negated labels remain, preferring the one in a sentence carrying an explicit verdict cue ("appears to be", "in conclusion", "skill level", ...); failing that, majority vote across mentions, tie-broken by first mention.
4. Treating a genuinely unresolved case (no mentions, or a raw `ERROR`/empty answer) as `Unknown` rather than guessing.

**This pass regenerated both pipelines from scratch** (`.venv-stats/bin/python figures-frame_count/rescore.py` and `figures-video/rescore_video.py`, both re-run fresh rather than trusting cached `master_summary.csv` files) and additionally re-parsed every file *outside* the pipeline's manifest — 3-class ablation, cross-domain probes, cam03/bestexo/n64/benchmark_400/commentary — by importing the identical `parse()`/`to_binary_gt()` functions, so every number in this document (canonical grid or ad-hoc) is now produced by one shared parser rather than two independently-written ones.

**Net effect vs. the 2026-08-06 hand-verification:** ~35 individual result entries changed (nearly all "reasoning"-prompt or free-text-adjacent), typically by 1–5 percentage points; two results changed enough to flip Collapsed↔Kept status (§1.3 VideoLLaVA trimmed-ego reasoning, §4 3-class exo/ego both moved in the "more Unknown, less credited" direction). No qualitative conclusion in the 2026-08-06 version is reversed by this pass — Gemini binary-ego climbing (76%) remains the only unambiguous above-chance result in the project, and every collapse-vs-kept classification that changed moved toward *more* caution (fewer classes credited), not less.

**Caveat on `statistics/master_results_summary.csv`** (the older, pre-`figures-*` summary referenced in the 2026-08-06 version): the generator script (`statistics/build_master_summary.py`) hardcodes a `dissertation_v2/results/` path that no longer exists (the folder was renamed to `diss_climb/` mid-project). Re-running that script as-is silently drops all climbing rows. **Do not use it or re-run it** — `figures-frame_count/master_summary.csv` and `figures-video/master_summary.csv` (regenerated for this pass) are now the authoritative machine-readable summaries; this document is the authoritative narrative/detail view built on top of them.
