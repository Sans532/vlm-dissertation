# Chapter: Results and Discussion — Complete Source Material

*Compiled 2026-08-13 for thesis writing. This file gathers **every result the project produced**, the sequential hypotheses that drove them, the figures and tables to insert, and every problem/challenge encountered. All numbers were re-verified against raw CSVs (see `DISSERTATION_RESULTS.md`) and, where noted, re-scored by the negation-aware parser in `figures-frame_count/rescore.py` / `figures-video/rescore_video.py`.*

**⚠ Reporting conventions used throughout (state these at the start of the chapter):**

1. **Binary tasks have n = 50 (chance = 50%); four-class tasks (fourclass / reasoning / structured) have n = 100 (chance = 25%).** The two are *never* directly comparable — 50% on binary is exactly as uninformative as 25% on four-class. **All tables below therefore keep binary results in separate tables from four-class results**, and the headline figures (fig01 / vfig01) plot *accuracy above each task's own chance* so both share one zero line. Do the same in the thesis.
2. Every accuracy is paired with the **per-class breakdown and the predicted-label distribution**. A bare accuracy number is meaningless in this project because of pervasive single-label collapse (§4). This convention is itself a methodological contribution — say so explicitly.
3. "**Collapsed**" = at most one ground-truth class has non-zero accuracy (the model effectively always outputs one label). Collapsed results are reported (they are findings about model behaviour) but never cited as evidence of discriminative skill judgment.
4. Views: **exo** = third-person camera (cam01 unless stated), **ego** = Aria headset first-person. Gemini's exo files are labelled "single" in the significance table.
5. Significance: binomial test for binary; chi-squared goodness-of-fit vs uniform for four-class. **Caveat to state**: chi-squared "significance" here mostly reflects the model's lopsided *output distribution*, not accuracy above chance — e.g. a 100%-Novice collapse at 25% accuracy is "significant" (p < 1e-40) precisely because it is degenerate. Interpret four-class p-values only alongside the collapse check.

---

## 1. Chapter narrative: the hypotheses in the order they occurred

Present the Results chapter as an escalating investigation, not a flat grid. The eleven hypotheses, chronologically (reconstructed from commit history and script comments):

| # | Hypothesis | Outcome (one line) |
|---|---|---|
| H1 | **Baseline feasibility** — can Qwen2.5-VL / VideoLLaVA classify climbing skill (Novice vs Expert) from sampled frames at all? | No: binary collapses to 100% "Novice" (50% = chance by construction). |
| H2 | **Frame budget** — does 8→16→32 frames help? | No consistent gain; VideoLLaVA *breaks* beyond 8 frames (context overflow, §7.2). |
| H3 | **Ego vs exo viewpoint** — does camera perspective matter? | Yes, substantially — Qwen2.5-VL dance structured: ~25% ego (collapsed) → 41% exo; Gemini climbing binary: 76% ego vs 60% exo. |
| H4 | **Diagnostics — why the "Novice" default?** Per-frame probing of a known Late-Expert clip. | Early frames (setup, standing, looking at a QR code) are scored "Novice"; only frames of active climbing get scored higher → motivated H5. |
| H5 | **Trimming** — does cutting to the annotated task window (`task_start_sec`–`task_end_sec`) fix the Novice bias? | Only partially; binary stays collapsed, small/inconsistent shifts in four-class formats. |
| H6 | **Prompt format** — one-word vs fourclass vs free-text reasoning vs structured template? | Yes: binary is the most collapse-prone; structured shows the most class spread (best Qwen2.5-VL run is structured). |
| H7 | **Cross-model** — proprietary cloud (Gemini) vs open-weight local? | Gemini is far more reliable (no failures, ≤1% invalid output) and holds the only genuine above-chance result (76% binary climbing ego). |
| H8 | **Camera angle within exo** — cam01 vs cam03? | No effect; same collapse/weak-reasoning pattern → failure is about information/format, not camera placement. |
| H9 | **Cross-domain generalization** — basketball, JIGSAWS surgery, mixed activities? | Essentially no transfer; apparent successes (JIGSAWS 65.5%) are majority-class artifacts. |
| H10 | **Label granularity** — is 4 classes too fine? 3-class ablation. | Collapsing to 3 classes does not rescue performance; mid "Expert" tiers remain unresolved. |
| H11 | **A second model in the video-proportional paradigm** — Qwen3-VL-8B was added as an open-weight comparator for Gemini at the same ~1fps sampling rate (its native pipeline was explicitly configured to match Gemini's fps=1 setting), which simultaneously provides a generation comparison against Qwen2.5-VL-7B. | Native pipeline failed (fps=24 OOM bug, abandoned at 12/50 clips), so the Gemini comparison is same-*rate*, not same-ingestion-mechanism. Key answer: Gemini's binary signal does **not** come from the ~1fps paradigm — Qwen3-VL at the same rate still collapses (50–54% binary). Secondary answer: modest generational gains over Qwen2.5-VL — best four-class climbing result (37% ego) and first meaningful Late-Expert predictions — with the *collapse target shifting* from "Novice" to "Intermediate Expert". |

A twelfth, unnumbered control belongs at the end of the narrative: the **text-only commentary experiment** (§6.5) — if the model can't recover skill level even from expert *language*, the failure isn't purely visual grounding.

---

## 2. The two video-input paradigms (frame this before any results)

This is a deliberate methodological axis, not a per-model detail:

1. **Fixed frame-count sampling** — Qwen2.5-VL-7B and VideoLLaVA. Manual OpenCV extraction of exactly 8 / 16 / (Qwen only) 32 / (Qwen only, added 2026-08-18 for dance; climbing since 2026-08-09) 64 uniformly-spaced frames **regardless of clip duration**. The only paradigm where the frame-count ablation (H2) is meaningful. Covered by figure set `figures-frame_count/` — `fig10` (Qwen 64-frame, climbing+dance, all prompts) and `fig11` (8/16/64 comparison against the standard frame-count-effect figure, both activities). *Note: this chapter predates the 64-frame condition and the ablation write-up in §6.5 below reflects only 8/16; see `DISSERTATION_RESULTS.md` §1.2.5/§2.2.4 and `RESULTS_SUMMARY_TABLES.md` §5 for the current 8/16/64 numbers.*
2. **Native / ~1fps proportional input** — Gemini (true native: raw video file uploaded to the API, internal ~1fps sampling) and Qwen3-VL-8B. **Qwen3-VL was added deliberately as an open-weight comparator for Gemini within this paradigm** — its native script set `nframes = round(duration_sec)` explicitly "matching Gemini's fps=1 setting" — so that any Gemini advantage could be attributed either to the input paradigm or to the model itself. The native attempt failed (see §7.3); all reported Qwen3-VL numbers come from a manual OpenCV substitute reproducing the same ~1fps rate (4–263 frames per clip, median 26 climbing / 64 dance). Covered by figure set `figures-video/`.

**Caveat to state verbatim**: Gemini and Qwen3-VL are comparable in *sampling rate*, not in *ingestion mechanism*. There is no 8-vs-16-vs-32 comparison for either of them, and no Qwen3-VL native run for dance at all.

**What the pairing establishes**: since Qwen3-VL at the same ~1fps rate still collapses on binary (50–54%) while Gemini reaches 76%, the project's one genuine above-chance result is attributable to Gemini the model, not to the video-proportional input paradigm.

Both figure sets re-parse every raw answer with the same parser, so numbers across the two sets are directly comparable.

---

## 3. Main results — Climbing and Dance

### 3.1 BINARY task (n = 50, chance = 50%) — report separately from everything else

**Table B1 — Binary results, climbing (Novice vs Expert, 25/25 balanced).** Suggested thesis table.

| Model | Input | Trim | View | Accuracy | Novice recall | Expert recall | Predicted labels | Verdict |
|---|---|---|---|---|---|---|---|---|
| Gemini | native video | entire | ego | **76.0%** (38/50) | 84% | 68% | Novice 29, Expert 21 | **Genuine signal — only unambiguous above-chance result in the project** (binomial p = 0.0003) |
| Gemini | native video | entire | exo | 60.0% (30/50) | 52% | 68% | Expert 29, Novice 21 | Not significant (p ≈ 0.26) but non-collapsed |
| VideoLLaVA | 16 frames | entire | exo | **76.0%** (38/50) | 88% | 64% | mixed | Above chance (p < 0.001) **but an isolated artefact**: the same clips on ego score 32% (below chance) — treat with suspicion |
| VideoLLaVA | 16 frames | entire | ego | 32.0% (16/50) | 48% | 16% | mixed | Below chance |
| VideoLLaVA | 8 frames | trimmed | exo | 60.0% (30/50) | 100% | 20% | Novice 45, Expert 5 | Weak, near-collapse |
| Qwen3-VL | ~1fps frames | entire | exo | 54.0% (27/50) | 100% | 8% | Novice 48, Expert 2 | Near-collapse |
| Qwen2.5-VL | 8/16/32 fr | entire+trimmed | ego+exo | 50.0% in **every** run | 100% | 0% | Novice 50/50 | **Total collapse — answers "Novice" on 100% of clips in all 8+ conditions, p = 1.0** |
| VideoLLaVA | 8 fr entire; 16 fr trimmed | both | mostly | 50.0% | 100% | 0% | Novice 50/50 | Collapse |
| VideoLLaVA | 32 frames | entire | both | — | — | — | 48/50 ERROR/empty | Run failed outright (§7.2) |
| Qwen3-VL | ~1fps frames | entire | ego | 50.0% | 100% | 0% | Novice 50 | Collapse |
| Qwen3-VL | native (attempt) | entire | both | 6/12 = 50% | 100% | 0% | Novice 12/12 | Aborted at 12/50 clips (§7.3) |

**Table B2 — Binary results, dance.**

| Model | Input | View | Accuracy | Novice recall | Expert recall | Predicted labels | Verdict |
|---|---|---|---|---|---|---|---|
| Gemini | native | exo | 60.0% (30/50) | 52% | 68% | Expert 29, Novice 21 | Not significant (p = 0.20), non-collapsed |
| Gemini | native | ego | 52.0% (26/50) | 100% | 4% | Novice 49, Expert 1 | ≈ chance, near-collapse — inverse of climbing, where ego *helped* Gemini |
| Qwen3-VL | ~1fps | exo | 50.0% (25/50) | 92% | 8% | Novice 46, Expert 4 | Near-collapse |
| Qwen3-VL | ~1fps | ego | 50.0% | 100% | 0% | Novice 50 | Collapse |
| Qwen2.5-VL | all 8 conditions | both | 50.0% | 100% | 0% | Novice 50 | Collapse in every run |
| VideoLLaVA | all 8 conditions | both | 50.0% | 100% | 0% | Novice 50 | Collapse in every run |

**Discussion points for the binary subsection:**
- Because the set is class-balanced, accuracy = midpoint of the two recalls; every degenerate strategy lands on exactly 50%. **vfig02** visualises this: points on the recall-vs-recall diagonal all score 50%; only Gemini climbing-ego, at (84, 68), sits meaningfully off the corner.
- Gemini binary climbing-ego (76%, p = 0.0003) is **the single genuine above-chance result across the entire project** (both rescoring pipelines agree). VideoLLaVA's 76% exo/n16 coexists with 32% on ego for identical clips — cite it only with that caveat.
- The Qwen2.5-VL result deserves its own paragraph: 100% "Novice" in *every* binary condition — 8/16/32 frames, entire and trimmed, ego and exo, climbing and dance, plus cam03, basketball, JIGSAWS and mixed. Its 50% is base rate, not judgment.

### 3.2 FOUR-CLASS tasks (n = 100, chance = 25%): fourclass, reasoning, structured prompts

**Table F1 — Climbing, best/representative four-class results (Kept, non-collapsed).**

| Model | Prompt | Frames | Trim | View | Acc | Per-class (N/E/I/L) | Note |
|---|---|---|---|---|---|---|---|
| Qwen3-VL | fourclass | ~1fps | entire | ego | **37%** | 92 / 0 / 56 / 0 | **Best four-class climbing result of any model** |
| Gemini | structured | native | entire | exo | 33% | 40 / 0 / 92 / 0 | |
| Gemini | fourclass | native | entire | ego | 32% | 48 / 0 / 80 / 0 | |
| Gemini | structured | native | entire | ego | 31% | 44 / 0 / 80 / 0 | |
| VideoLLaVA | structured | 8 | trimmed | ego | 30% | 60 / 4 / 12 / 44 | Best VideoLLaVA structured; only run with substantial Late-Expert recall |
| Qwen2.5-VL | fourclass | 16 | entire | exo | 27% | 92 / 0 / 16 / 0 | |
| Qwen3-VL | structured | ~1fps | entire | exo | 27% | 24 / 0 / 64 / 20 | First climbing run to meaningfully populate Late Expert |
| Qwen2.5-VL | reasoning (re-extracted) | 16 | entire | exo | 20% | 28 / 16 / 28 / 8 | **Only climbing four-class run resolving all four classes with non-zero accuracy** (`…n16_reasoning_patched.csv`) |

Full per-condition detail (all ~40 climbing four-class runs incl. all collapsed ego columns, trimmed variants, and both duplicate structured runs) is in `DISSERTATION_RESULTS.md` §1 — pull individual rows as needed; do not reproduce all of it in the thesis body (appendix material).

**Table F2 — Dance, best/representative four-class results.**

| Model | Prompt | Frames | Trim | View | Acc | Per-class (N/E/I/L) | Note |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL | structured | 16 | entire | exo | **41%** | 88 / 0 / 76 / 0 | **Best Qwen2.5-VL result in the project** |
| Qwen2.5-VL | structured | 16 | trimmed | exo | 39% | 68 / 0 / 88 / 0 | Close second |
| Qwen2.5-VL | structured | 8 | trimmed | exo | 35% | 68 / 0 / 68 / 4 | |
| Qwen2.5-VL | fourclass | 16 | trimmed | exo | 34% | 64 / 0 / 72 / 0 | |
| Gemini | reasoning | native | entire | exo | 32% | 28 / 4 / 96 / 0 | Gemini's best dance result |
| Qwen2.5-VL | reasoning | 8 | entire | exo | 32% | 52 / 0 / 76 / 0 | |
| VideoLLaVA | reasoning | 8 | trimmed | ego | 31% | 0 / 16 / 12 / 96 | Late-Expert-dominated (see collapse discussion) |
| Qwen3-VL | structured | ~1fps | entire | exo | 30% | 12 / 4 / 80 / 24 | Best Qwen3-VL dance result |
| Qwen3-VL | fourclass | ~1fps | entire | ego | 28% | 100 / 0 / 12 / 0 | |

**Key contrasts to write up:**
- **No model meaningfully exceeds the 25% four-class baseline once collapse is accounted for.** Re-scored pooled numbers: Gemini 29% climbing / 25% dance; Qwen3-VL 29% / 27%; Qwen2.5-VL and VideoLLaVA at or below. Per-class recall is always grossly uneven (e.g. Gemini climbing: Intermediate 86%, Early 0%, Late ~1%).
- **Early Expert is never resolved by any model** (recall ~0 everywhere; the sole exception is the patched Qwen reasoning re-extraction at 16%). **Late Expert** is populated only by VideoLLaVA-structured and Qwen3-VL-structured.
- **Qwen2.5-VL's best results are all dance + exo + structured** — the strongest evidence that viewpoint and prompt format, not model capability, drive observed accuracy (H3×H6 interaction). The mirror image: *every* Qwen2.5-VL dance ego fourclass/reasoning run is a 100%-Novice collapse at exactly 25%.
- The rescoring pipeline's blunt one-liner (use in discussion): "Qwen dance structured exo is the only family consistently above chance (32–41% vs 25%), driven by its Novice bias aligning with the 'Skill Level: Novice' template rather than genuine discrimination."

### 3.3 Entire vs trimmed (H5)

Motivated by the H4 diagnostics (setup/idle frames read as "Novice"). Finding: **trimming produces no systematic improvement** in any model × activity × prompt combination (fig03). Specifics:
- Qwen2.5-VL / VideoLLaVA binary: still 100%-Novice collapsed after trimming (all conditions).
- Qwen2.5-VL dance structured: 41% entire → 39% trimmed (n16 exo); 32% → 35% (n8 exo) — noise-level shifts in both directions.
- VideoLLaVA climbing structured ego: 23% entire → 30% trimmed (largest positive shift, still ≈ chance).
- Climbing reasoning (Qwen): trimming *hurts* (14–15% entire → 10–12% trimmed exo).
- Conclusion: the Novice default is not caused by idle footage; it survives removal of the idle footage.

### 3.4 Frame count (H2) — fixed-frame models only

- Qwen2.5-VL: 8 → 16 frames leaves four-class accuracy essentially unchanged; binary identical (collapsed) at 8, 16 and 32.
- VideoLLaVA: **degrades catastrophically with more frames** — valid-output rate on climbing falls from 92% (8 fr) to 27% (16 fr), 73% unusable at n16, 100% at n32 (root cause §7.2). fig02 shows both panels.
- Qwen3-VL (adaptive, 4–263 frames chosen per clip): accuracy **flat across frame-count quartiles** (~10 to ~165 frames); the only monotone effect is unusable (truncated) responses rising 0.5% → 6.6% on climbing (vfig03).
- Combined conclusion: **frame budget is not the bottleneck** — no model extracts more skill signal from more frames.

### 3.5 Viewpoint (H3)

- **Gemini**: viewpoint matters and is activity-dependent — ego helps on climbing (60% → 76% binary) and hurts on dance (60% → 52%) (vfig06).
- **Qwen2.5-VL**: ego strengthens the Novice default — Novice-share of predictions rises 75%→85% (climbing) and 36%→90% (dance) going exo→ego (fig04, right panel). Hence all its surviving results are exo.
- **Qwen3-VL**: insensitive to viewpoint.
- **cam01 vs cam03 (H8)**: no difference — Qwen binary cam03 is the same 100%-Novice collapse; VideoLLaVA cam03 reasoning (14–19%) statistically indistinguishable from cam01. Failure mode is not about camera placement.

---

## 4. The label-collapse phenomenon (dedicate a full subsection — it is a finding, not a caveat)

The single most important pattern in the project. Quantify it:

- Of the ~130 experimental conditions run, **the large majority are collapse or failure artifacts**: all 16 Qwen2.5-VL binary runs, all 16 VideoLLaVA dance binary/fourclass runs, most VideoLLaVA climbing binary runs, every Qwen2.5-VL dance-ego fourclass/reasoning run, all cross-domain binary runs.
- Re-scored prediction shares: binary — Qwen2.5-VL answers "Novice" on **100%** of clips (both activities); VideoLLaVA 89.5% (climbing) / 100% (dance). Four-class pooled — Qwen2.5-VL climbing: 80% Novice / 18% Intermediate / ~0% Early+Late.
- **The collapse target moves between model generations** (vfig04): frame-sampled models default to "Novice" (Qwen2.5-VL 71% of all four-class predictions, VideoLLaVA 51%); video-native/adaptive models default to "Intermediate Expert" (Gemini 79% climbing / 62% dance; Qwen3-VL ~47%). Newer models are not less biased — differently biased.
- **VideoLLaVA's prior even flips within one model by prompt format** (dance): binary/fourclass → "Novice"; free-text reasoning → "Late Expert" (94–95/100). Plausible reading: it associates flowing continuous dance motion with "expert" when allowed to free-associate — a bias, not a judgment.
- **Gemini never once predicts Early or Late Expert on climbing** in any fourclass/reasoning/structured run — it silently collapses the 4-point scale to Novice-vs-Intermediate.
- Confusion matrices (fig06, vfig05): rows are near-identical — the predicted distribution barely changes with true skill level, i.e. predictions are close to independent of the ground truth. Gemini labels 86–89% of Intermediate and Late Expert climbers "Intermediate Expert" — but also 67% of true novices.
- Methodological moral (state as a contribution): **naive accuracy is uninterpretable for zero-shot VLM classification; per-class recall + predicted-label distribution must always be reported**. JIGSAWS (§6.2) is the cautionary example: 65.5% "accuracy" from a model that gave the same answer 29/29 times.

---

## 5. Generated answers: phrases, hedging, stereotypy, refusals (the qualitative/text results)

This satisfies the "phrases / generated answers" requirement — four distinct findings:

### 5.1 Qwen2.5-VL hedges with a literal disjunction
In free-text reasoning answers, Qwen2.5-VL very frequently outputs the exact phrase **"novice or early expert"** instead of committing: 24× in `qwen_climbing_trimmed_n16_reasoning.csv`, 12× in `…entire_n16…`, 11× in `…trimmed_n8…`, 7× in `…entire_n8…` — versus ~0× under structured/fourclass prompts. Example (ground truth Novice): *"The climber appears to be a novice or early expert based on the following observations: … Overall, the climber's technique and body alignment suggest that they are at a novice or early expert level."* Implications: (a) free-text prompts elicit far more hedging than templates; (b) any Early-Expert "hit" scored from such a disjunction is a coin-flip artifact. The corrected parser scores unresolved disjunctions as `Unknown`.

### 5.2 Response stereotypy — frame-sampled models write boilerplate
fig09: **Qwen2.5-VL opens 99–100% of egocentric reasoning answers with an identical sentence** — the answer is generated from a prior, not the video. VideoLLaVA reasoning/structured text is heavily copy-pasted boilerplate across unrelated clips (documented in the audit).

### 5.3 Video-native models write genuinely per-clip text — that still isn't diagnostic
vfig09: Gemini and Qwen3-VL produce 1,000–2,500-character clip-specific analyses; at most 6–13% share an opening line. But per the confusion matrices the verbose, individualized descriptions still do not track true skill level. **Verbosity ≠ grounding** — a good discussion paragraph.

### 5.4 Gemini's principled refusals
Audit finding: on 3 dance-ego clips the egocentric video shows the camera operator filming (no dancer visible) — Gemini evaluated *the videographer's filming technique* or refused; 1 climbing-ego clip shows a briefing meeting — Gemini refused ("Unknown"). These are scored incorrect but are actually the most grounded behaviour observed, and they expose benchmark noise: some ego clips do not contain the activity at all. Qwen3-VL's `Unknown` rate (16–27% on reasoning) is partly truncation, partly similar refusal.

### 5.5 Answer-length / truncation cost
The reasoning prompt costs Qwen3-VL 18% of climbing responses to truncation (answer cut off before a verdict) — long-form prompting has a measurable reliability price on local models (vfig08 left). Gemini stays ≤1% invalid everywhere.

---

## 6. Generalization and control experiments

### 6.1 Basketball (Qwen2.5-VL, binary n=50, 8 fr, ego+exo) — report in the binary block
24% overall (12/50), **below chance**; Novice 48%, Expert 0%. 15 of the 16 ERROR rows fall on Expert-labelled clips (missing videos — the benchmark had to be rebuilt with `rebuild_basketball_benchmark.py` after 16/50 clips were found absent on disk). Evidence of pipeline fragility, not model judgment.

### 6.2 JIGSAWS surgical suturing (Qwen2.5-VL, binary, exo, **n=29, NOT balanced**: 19 Novice / 10 Expert)
Cross-dataset test (JIGSAWS, not EgoExo4D). **65.5% accuracy — entirely a majority-class artifact**: the model predicted "Novice" for all 29 clips and 19/29 labels happen to be Novice. Never cite the 65.5% without this sentence. Also flag the class imbalance as a limitation vs the balanced main benchmarks.

### 6.3 Mixed-activity benchmark (Qwen2.5-VL, 8 fr, trimmed; Basketball + Music + Cooking + Soccer; climbing/dance excluded by design; n=100, 25/skill-class, activities mixed within class)
- Binary framing: 25% both views — 100% Novice collapse (note: 25% on a *binary* prompt because the ground truth here spans 4 classes mapped to the binary question; well below the 50% binary chance).
- Structured: exo 19%, ego 23% — near/below chance.
- **Per-activity split (the interesting bit)**: Basketball 9–12/48 and Cooking 5–8/21 retain some signal; **Music 2–5/20 and Soccer 0–1/11 are near-total failures**. Hypothesis for future work (flag as speculative, n=11–48/activity): the residual signal tracks activities with visible object-interaction cues (ball, knife) rather than whole-body motion quality.

### 6.4 3-class label-granularity ablation (H10; Qwen2.5-VL structured, climbing, 10/class, n=30)
Mapping Novice / Intermediate (=Intermediate Expert) / Expert (=Late Expert), Early Expert excluded by construction. Ego: 33.3% (10/30; Novice 0%, Intermediate 40%, Expert 60%; predictions Expert-heavy). Exo: 36.7% but **29/30 predictions were "Expert"** — collapse. Coarsening the scale does not rescue discrimination; exploratory result, small n.

### 6.5 Text-only commentary control (no video at all)
Given ONLY the expert commentary text of a climbing clip, can the LLM state the 4-class skill level? (`commentary/`, n=100, 25/class):
- **Qwen2.5-VL: 26/100 = 26%** — Novice 1/25, Early 7/25, Intermediate 18/25, Late 0/25; predictions: Intermediate 67, Early 30, Novice 3.
- **VideoLLaVA: 29/100 = 29%** — Novice 15/25, Early 0/25, Intermediate 13/25, Late 1/25; predictions: Intermediate 39, Novice 35, Unknown 25, Late 1.
- Both ≈ chance (25%) with the familiar collapse shape. **Interpretation: skill level is not reliably recoverable from expert language alone either** — the failure is not purely one of visual grounding; per-clip skill signal is weak in every modality tested (or the models cannot exploit it). This is the cleanest closing argument of the chapter.

---

## 7. Problems, issues and challenges (report transparently — several are findings)

1. **Missing source videos (climbing).** 9 `uniandes_bouldering_*` takes referenced in the benchmark JSONs have no video files on disk (both exo cam01 and ego aria01) → a systemic ~9% ERROR floor in every Qwen/VideoLLaVA climbing file (fig07). Gemini was unaffected (reads uploaded/cloud video), which **confounds direct model comparison on those clips**. One clip (`uniandes_bouldering_027_87`) has valid video yet still fails — an unexplained residual pipeline bug. Basketball was worse: 16/50 clips missing, benchmark rebuilt.
2. **VideoLLaVA's 4096-token context limit** (root cause of its 16/32-frame failures). Vision tokens for 16 frames push the prompt over the model's 4096-token limit. Scripts with a hard `if n_tokens > 4096` guard (`reasoning_n16.py:80`, `structured_n16.py:83`) abort → files of literal `"ERROR"` / `"..."`; `fourclass_n16.py` has no guard → silently generates garbage parsed as `Unknown` (95/100 rows). The 32-frame binary run (48/50 empty) is the severe end of the same problem. **Frame budget interacts with context budget** — an architectural constraint worth a discussion paragraph.
3. **Qwen3-VL native-video fps=24 fallback bug.** The native pipeline ignored the requested `nframes` (~1fps to match Gemini), silently sampled at 24fps, and OOM-crashed; abandoned after 12/50 clips (all-Novice; 3 of 4 result files empty — vfig08 right). All reported Qwen3-VL numbers are from the manual OpenCV ~1fps substitute. No native dance attempt exists.
4. **Label-extraction (parsing) bugs — and the rescoring methodology they forced.** The original keyword parser misread negations ("not yet a Late Expert" → Late Expert) and coin-flipped hedged disjunctions. Fix: negation/aspiration-aware re-parser applied to *every* raw answer. Impact: 175/7,812 stored labels changed in the frame-sampled set (per-run accuracy shifts up to 8 points; largest in VideoLLaVA climbing reasoning) and 171/2,765 in the video set (≤3-point shifts) — fig08 / vfig10. Consequence for reproducibility: **stored `*_predicted`/`*_correct` columns in the result CSVs are unreliable; all thesis numbers should come from the rescore pipelines.**
5. **Run-to-run non-determinism.** Two independent Qwen2.5-VL 8-frame structured climbing runs over the same 100 clips disagree on 86/100 predictions (19%/24% vs 5%/14%) — a reliability finding in its own right.
6. **Data hygiene issues found by audit**: a duplicate stale-retry row in `gemini_climbing_entire_binary.csv` (deduplicated before scoring); `fourclass_trimmed.csv` written by both the n8 and n16 scripts to the same path (frame count ambiguous → excluded from frame plots); schema drift across nominally-parallel CSVs; `dissertation_v2/` → `diss_climb/` rename leaving stale paths in `statistics/master_results_summary.csv` (do not re-run `build_master_summary.py` without fixing `RESULT_DIRS`).
7. **Benchmark content noise**: some egocentric clips show setup/briefing/filming rather than the activity (Gemini's refusals, §5.4) — ego-view ground truth is noisier than exo.
8. **JIGSAWS class imbalance** (19/10) — makes raw accuracy misleading there by construction.

---

## 8. Figures to insert (with what each shows)

All exist as 300-dpi PNG + vector PDF. Suggested captions are in `figures-frame_count/README.md` and `figures-video/README.md` — adapt them, don't rewrite from scratch.

**Headline (main body):**
| Figure | File | Shows / why it's in the thesis |
|---|---|---|
| Fig. A (headline) | `figures-frame_count/fig01_accuracy_above_chance` | Every fixed-frame run plotted as accuracy *minus its own chance* (50% binary, 25% four-class) on one zero line — nothing departs meaningfully from chance. Solves the n=50-vs-n=100 comparability problem visually. |
| Fig. B (headline) | `figures-video/vfig01_accuracy_above_chance` | Same construction for Gemini/Qwen3-VL — only Gemini binary climbing clears zero. |
| Fig. C ("money figure") | `figures-video/vfig07_generation_comparison` | Balanced accuracy of all four models across both paradigms: Gemini 68% on 2-way climbing vs 50–53% for the rest; all four within 4 pts of 25% on 4-way. Video-native input helps only the coarsest distinction. |
| Fig. D | `figures-frame_count/fig05_label_collapse` + `figures-video/vfig04_label_collapse` | The core failure mode: predicted-label distributions vs balanced ground truth; and the collapse target shifting Novice → Intermediate Expert across generations. Could be combined into one two-panel thesis figure. |
| Fig. E | `figures-video/vfig02_binary_discrimination` | Novice-recall vs Expert-recall plane for the binary task (n=50) — the clean way to display binary results separately: all degenerate strategies lie on the 50% diagonal; Gemini climbing-ego at (84,68) is visibly apart. |
| Fig. F | `fig06_confusion_matrices` and/or `vfig05_confusion_matrices` | Row-normalised confusions, four-way pooled: rows near-identical ⇒ predictions ~independent of true skill. |

**Ablations (main body or appendix):**
| Figure | File | Shows |
|---|---|---|
| Frame count | `fig02_frame_count_effect` | 8→16 frames: Qwen flat, VideoLLaVA valid-output crashes 92%→27%. |
| Adaptive frames | `vfig03_adaptive_frame_count` | Qwen3-VL accuracy flat across 4–263 frames; truncation rises instead. |
| Entire vs trimmed | `fig03_entire_vs_trimmed` | No systematic trimming benefit (H5). |
| Viewpoint | `fig04_viewpoint_asymmetry`, `vfig06_viewpoint_asymmetry` | Exo-vs-ego parity plots + Novice-share by view; Gemini's activity-dependent viewpoint effect. |

**Methodology / reliability (appendix or a "limitations" subsection):**
| Figure | File | Shows |
|---|---|---|
| Output validity | `fig07_output_validity`, `vfig08_output_validity_coverage` | Unusable-response rates (missing-video floor, VideoLLaVA n16 collapse, Qwen3-VL truncation, native-run coverage gap). |
| Rescoring impact | `fig08_rescoring_impact`, `vfig10_rescoring_impact` | Stored vs re-parsed accuracy — justifies the re-scoring methodology. |
| Stereotypy | `fig09_response_stereotypy`, `vfig09_answer_style` | 99–100% identical Qwen openers vs genuinely per-clip Gemini/Qwen3-VL prose. Supports §5. |

Older one-off charts in `visualizations/` (`chart1_label_collapse_heatmap.png` … `chart6_exo_ego_asymmetry.png`) are superseded by the two figure sets but chart1 (heatmap) is an alternative collapse visual if preferred.

**Regenerating**: `.venv-stats/bin/python figures-frame_count/rescore.py && .venv-stats/bin/python figures-frame_count/make_figures.py` (and the `figures-video/` pair). Note `figures-video/` imports the parser from `figures-frame_count/` — don't rename that folder.

## 9. Tables to insert

1. **Table B1/B2 (binary, n=50)** — from §3.1 above. Keep physically separate from four-class tables; add a table note: "Binary chance = 50%; these results are not comparable with the four-class tables (chance = 25%)."
2. **Table F1/F2 (four-class, n=100)** — from §3.2; columns: model, prompt, frames, trim, view, accuracy, per-class recall, predicted-label distribution, significance.
3. **Collapse census table** — condition counts per model: total runs / collapsed / failed / kept (derive from `DISSERTATION_RESULTS.md` §5 + `figures-*/master_summary.csv`). Anchors §4.
4. **Cross-domain table** — Basketball / JIGSAWS / Mixed (binary and structured) / cam03 / 3-class, each with the artifact caveat column; plus the mixed per-activity breakdown (Basketball 12/48, Cooking 5/21, Music 2/20, Soccer 0/11 exo-structured, etc.).
5. **Commentary (text-only) table** — §6.5 numbers, next to the corresponding video-based four-class rows for the same clips.
6. **Experimental-grid coverage table** (methodology chapter, referenced from results): model × prompt × frames × trim × view, marking cells run / failed / not-run and why (VideoLLaVA n32, Qwen3-VL native, no Gemini trimmed runs, etc.).
7. **Hedging-frequency mini-table** — counts of the literal "novice or early expert" disjunction per file (§5.1), reasoning vs structured.
8. Appendix: the full per-condition dump (all ~130 rows) from `DISSERTATION_RESULTS.md` / `figures-*/master_summary.csv`.

## 10. Suggested section skeleton for the chapter

1. **6.1 Evaluation protocol and reporting conventions** — two chance baselines, per-class + distribution reporting, collapse definition, rescoring pipeline (fig08/vfig10), significance tests + chi-squared caveat.
2. **6.2 Binary skill discrimination (n=50)** — Tables B1/B2, vfig02; the Gemini result; the universal Novice collapse.
3. **6.3 Four-class skill classification (n=100)** — Tables F1/F2, fig01/vfig01, confusion matrices; nothing beats 25% meaningfully.
4. **6.4 The label-collapse phenomenon** — §4 material, fig05/vfig04; collapse census table.
5. **6.5 Ablations** — frame count (fig02, vfig03), entire vs trimmed (fig03), viewpoint incl. cam03 (fig04, vfig06), prompt format, video-input paradigm (vfig07 as capstone).
6. **6.6 What the models actually say** — hedging, stereotypy, verbosity-without-grounding, refusals (§5; fig09, vfig09).
7. **6.7 Generalization** — basketball, JIGSAWS, mixed, 3-class (§6.1–6.4).
8. **6.8 The text-only control** — §6.5; closes the loop on "is this a vision problem?"
9. **6.9 Reliability, engineering failures and limitations** — §7 in full (missing videos, context limits, native-pipeline bug, parser bugs, non-determinism, benchmark noise).
10. **6.10 Synthesis** — the three islands of (weak) competence are all different model/condition combinations (Gemini binary-climbing-ego 76%; Qwen3-VL fourclass-climbing-ego 37%; Qwen2.5-VL structured-dance-exo 41%): no single best configuration exists, and current zero-shot VLMs do not possess a robust visual notion of human skill.

---

## Source pointers (for checking any number)

- Full per-condition numbers: `DISSERTATION_RESULTS.md` (recomputed 2026-08-06; collapsed vs kept flagged per run).
- Project overview + hypotheses + writing guidance: `PROJECT_REPORT.md`.
- Re-scored condition summaries: `figures-frame_count/master_summary.csv`, `figures-video/master_summary.csv` (authoritative over stored CSV columns).
- Significance tests: `statistics/statistical_significance_summary.csv` (paths saying `dissertation_v2/` mean `diss_climb/`).
- Data-quality audit: `results_review_notes.md`.
- Commentary control: `commentary/commentary_4class.csv`, `commentary/videollava_commentary.csv` (accuracies computed 2026-08-13, this file §6.5).
