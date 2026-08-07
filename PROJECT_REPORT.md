# VLM Skill-Level Assessment — Project Summary

*Compiled 2026-08-06. This file is a working reference for writing the dissertation report: what the project is, what was built, every experiment run, the results that survived data-quality review, and section-by-section writing guidance.*

---

## 1. What this project is

The project evaluates whether off-the-shelf **video-capable vision-language models (VLMs)** can judge a person's **skill level** (Novice → Early Expert → Intermediate Expert → Late Expert) directly from short video clips, without task-specific training. Three model families were tested — **Qwen2.5-VL-7B**, **Qwen3-VL-8B**, **VideoLLaVA**, and **Gemini** (cloud, exact model per script) — across four prompt styles and two camera viewpoints, on a primarily EgoExo4D-derived benchmark of **bouldering/climbing** and **dance** clips, with secondary generalization tests on **basketball, JIGSAWS surgical suturing, and four other mixed activities (music, cooking, soccer)**.

The core research question: *does a general-purpose VLM have any usable notion of "skilled vs unskilled" human movement from visual evidence alone, and does that notion transfer across activity domains, camera viewpoints (egocentric vs exocentric), frame budget, and prompt format?*

### Task design (consistent across models/activities)
- **Binary**: Novice vs Expert (50 clips, 25/25 balanced).
- **Fourclass**: Novice / Early Expert / Intermediate Expert / Late Expert (100 clips, 25 per class).
- **Reasoning**: free-text chain-of-thought ending in a skill judgment, parsed post-hoc.
- **Structured**: templated `Observations / Errors / Skill Level` output, parsed post-hoc.
- **Views**: `exo` (third-person camera), `ego` (Aria headset, first-person), and for Gemini a `single`/combined view.
- **Trim conditions**: `entire` clip vs `trimmed` to the annotated `task_start_sec`/`task_end_sec` window (to remove setup/idle footage).
- **Frame counts**: 8, 16, and (Qwen2.5-VL only) 32 frames sampled uniformly; Qwen3-VL used ~1fps proportional sampling capped at a max, VideoLLaVA fixed at 8 (mostly) or 16.

All accuracy numbers below use the project's standard scoring: chance baseline is **50%** for binary and **25%** for fourclass (balanced 25/class), verified in `statistics/` with binomial (binary) / chi-squared (fourclass, goodness-of-fit vs uniform-random) significance tests.

### Two distinct video-input paradigms (a deliberate methodological axis, not just a per-model detail)

The four model families split into two genuinely different ways of getting video into the model, which is itself an experimental variable worth reporting explicitly rather than folding into "frame count":

1. **Fixed frame-count sampling — Qwen2.5-VL-7B and VideoLLaVA.** A manual OpenCV pass extracts a *fixed number* of frames (8, 16, or 32 for Qwen2.5-VL; 8 or 16 for VideoLLaVA) uniformly spaced across the clip, **regardless of clip duration** — a 5-second clip and a 90-second clip both get exactly 8 (or 16/32) frames. This is the classical "frame-count ablation" axis (H2 in §6) and is the only paradigm where frame-count-as-a-variable is meaningful, because it's the only one where frame count is decoupled from clip length.

2. **Native / ~1fps video input — Gemini and (attempted) Qwen3-VL-8B.** Gemini receives the **raw video file directly** via the API's own file-upload mechanism (`client.files.upload(file=video_path)` in every `diss_climb/scripts/gemini/*.py` / `diss_dance/scripts/gemini/*.py` script) — no manual frame extraction at all; the Gemini backend does its own internal frame sampling (its documented default is ~1 frame per second of video). Qwen3-VL-8B was **first attempted the same way**: `diss_climb/scripts/qwen3vl/binary.py` originally passed `{"type": "video", "video": video_path, "nframes": n}` straight to the model's own video processor, with `n = round(duration_sec)` — i.e. deliberately targeting ~1 frame per second of clip duration, explicitly written in the script as "matching Gemini's fps=1 setting," so that Qwen3-VL and Gemini would be directly comparable on the same input paradigm.

   **This native path failed for Qwen3-VL.** The model's video processor has an fps=24 fallback bug: instead of honoring the requested `nframes`, it silently defaulted to sampling at 24fps, which blew up memory and caused OOM crashes. Only 12 of 50 climbing clips completed before the run was abandoned (`qwen3vl_climbing_native_*.csv` — all-Novice collapse on the 12 that did complete, 50% accuracy, not informative). **The Qwen3-VL numbers reported everywhere in this project ("frames" mode) are therefore not from genuine native video ingestion** — they come from a manual OpenCV substitute (`extract_frames()` in the post-fix `binary.py`/`fourclass.py`/etc.) that was built specifically to *reproduce* the same ~1fps target rate (`n = max(1, round(duration_sec))`) by hand, bypassing the buggy native pipeline while keeping the sampling rate comparable to Gemini's.

   **Practical consequence for interpreting results**: Gemini and Qwen3-VL are the two models in this project evaluated at a **clip-duration-proportional** frame rate (~1fps) rather than a fixed count, but only Gemini's numbers reflect that model's *actual* native video pipeline — Qwen3-VL's are a same-rate approximation via frame extraction, not a true native-pipeline result. Any claim like "Gemini and Qwen3-VL were tested under directly comparable input conditions" should carry this caveat: comparable *sampling rate*, not comparable *ingestion mechanism*. This is also why frame-count ablation (H2) is only meaningful for Qwen2.5-VL/VideoLLaVA — Gemini and Qwen3-VL were never run at a fixed frame count at all, only at the proportional ~1fps rate, so there is no "8 vs 16 vs 32 frames" comparison available for either of them.

---

## 2. Data-quality findings that gate which results are trustworthy

A full audit (`results_review_notes.md`, `[[climbing-missing-video-files]]` memory) found:

1. **9 of 10 `uniandes_bouldering_*` takes referenced in the benchmark JSONs have no video files on disk** (both cam01 exo and Aria ego missing) → systemic `ERROR`/`Unknown` rows in every Qwen and VideoLLaVA climbing file (not Gemini, which reads a different/cloud video source). One further clip (`uniandes_bouldering_027_87`) has valid video but still fails — a genuine, still-unexplained pipeline bug.
2. **VideoLLaVA output is frequently degenerate**: several files are 100% single-label collapse ("always predicts Novice"), some are fully failed generations (literal `"..."` or `"ERROR"` for every row), and reasoning/structured text is often boilerplate copy-pasted across unrelated clips. These are called out per-file below and excluded from headline claims, per the user's explicit request to omit failed results.
3. **Schema drift** across nominally-parallel CSVs (different column layouts for binary files across runs) — noted, not a correctness issue once accounted for.
4. **Reasoning-text parser gaps** (misses paraphrased skill wording like "intermediate skill level") inflate `Unknown`/incorrect counts by a few points in some files — a fixable extraction bug, not a model failure; results below use the `recomputed` (parser-fixed) numbers where available in `statistics/master_results_summary.csv`. The original parser was a naive fixed-priority substring scan with two specific failure modes: it misread negated statements ("not yet a Late Expert" → wrongly scored as Late Expert), and it arbitrarily resolved hedged disjunctions by picking whichever label it checked first when the model hedged. **The hedge itself is a real, recurring Qwen2.5-VL behavior, not a parsing artifact**: in free-text "reasoning"-prompt answers, Qwen2.5-VL very frequently produces the literal disjunction **"novice or early expert"** instead of committing to one label — 24× in `qwen_climbing_trimmed_n16_reasoning.csv`, 12× in `qwen_climbing_entire_n16_reasoning.csv`, 11× in `qwen_climbing_trimmed_n8_reasoning.csv`, 7× in `qwen_climbing_entire_n8_reasoning.csv` — versus effectively 0× in structured/fourclass-prompt outputs. Example (`qwen_climbing_trimmed_n8_reasoning.csv`, ground truth Novice): *"The climber appears to be a novice or early expert based on the following observations: ... Overall, the climber's technique and body alignment suggest that they are at a novice or early expert level."* This means free-text reasoning prompts elicit substantially more hedging than templated ones, and any "correct" Early-Expert hit scored against this disjunction is not distinguishable from a coin-flip resolution of the hedge — a confound worth flagging wherever reasoning-prompt fourclass accuracy is cited as evidence of discriminative skill judgment. The corrected extractor treats an unresolved "X or Y" disjunction as `Unknown` rather than guessing.

**Folder note**: the codebase was reorganised mid-project — `dissertation_v2/` was renamed to `diss_climb/`. Paths inside `statistics/master_results_summary.csv` still say `dissertation_v2/...`; treat that as `diss_climb/...`.

---

## 3. `diss_climb` (bouldering/climbing) — results by model

Chance: binary 50%, fourclass 25%. n=50 (binary) / n=100 (fourclass/reasoning/structured) unless noted, 25 per class.

### 3.1 Gemini — all conditions (all trustworthy; best-behaved model overall)

| Prompt | View | Acc | Notes |
|---|---|---|---|
| Binary | ego | **76%** | significant (p<0.001) |
| Binary | single/combined | 58.8% | not significant (p=0.26) |
| Fourclass | ego | 32% | sig.; predictions concentrate on Intermediate Expert (70/100) + Novice |
| Fourclass | single | 26% | sig.; near chance |
| Reasoning | ego | 28% | sig. |
| Reasoning | single | 27% | sig. |
| Structured | ego | 31% | sig. |
| Structured | single | 33% | sig.; best fourclass-style result for climbing |

**Pattern**: Gemini reliably separates Novice (highest per-class accuracy, 40-84%) from the pooled "Expert" region, but essentially never predicts **Early Expert** or **Late Expert** (0% per-class accuracy in every fourclass/reasoning/structured run) — it collapses the 4-class scale to a 2-bucket Novice-vs-"Intermediate Expert" judgment. Binary framing is where it performs best and most confidently (ego view 76%, well above chance).

### 3.2 Qwen2.5-VL-7B — all conditions

Binary (8/16/32 frames, entire and trimmed, both views): **flat 50% accuracy** — every run collapses to predicting "Novice" for all 50 clips. Statistically indistinguishable from chance (p=1.0 in every case). This is the single most consistent negative finding in the climbing arm.

Fourclass / reasoning / structured (24-27% typical, chi-sq. significant vs *uniform* chance only because the model's output distribution is itself lopsided, not because it's actually accurate):

| Prompt | Frames | Trim | View | Acc | Class accuracy pattern |
|---|---|---|---|---|---|
| Fourclass | 16 | entire | ego | 25% | 100% Novice recall, 0% everywhere else (predicts Novice 98/100) |
| Fourclass | 16 | entire | exo | 27% | Novice 92%, Interm. 16%, others 0% |
| Fourclass | 8 | entire | ego/exo | 24-25% | same Novice-dominant pattern |
| Fourclass | trimmed | ego/exo | 24-26% | same pattern |
| Reasoning | 16 | entire | ego | 24% | Novice 96%, else 0% |
| Reasoning | 16 | entire | **exo** | 14% | more spread (Novice 24%, Interm. 32%) but *lower* accuracy — exo reasoning is worse than ego here |
| Structured | 16 | entire | ego | 24% | Novice 52%, Interm. 44% — most balanced Qwen climbing run |
| Structured | 8 | entire | ego | 19% | weakest structured run |

Two duplicate/alternate runs worth noting, both found only in the raw results directory (not previously tabulated):
- **`qwen_climbing_entire_n16_reasoning_patched.csv`** — a corrected label-re-extraction pass over the *same* raw model text as the 16-entire-reasoning row above. Its exo column scores 20% and is the **only climbing reasoning/fourclass run in the whole project that resolves all four ground-truth classes with nonzero accuracy** (Novice 28%, Early 16%, Interm. 28%, Late 8%) — the original (buggy) extractor's exo row (14%, above) only ever populated Novice/Intermediate. Its ego column is still a 24% Novice collapse, unchanged.
- **`qwen_climbing_entire_n8_structured.csv`** — a second, independent 8-frame structured run over the same 100 clips as the `structured_eval.csv` run reported above (19% ego / 24% exo), with 86/100 predictions differing between the two runs. This second run is materially worse and fully collapses on both views: ego 5% (mostly `Unknown`, only 20% Novice recall), exo 14% (Novice-only, 56% Novice recall). Excluded from the headline table as a strictly worse duplicate, but included here for completeness since prompt/seed non-determinism this large is itself a notable reliability finding.
- **`binary_n8_qwen_ego.csv` / `binary_n8_qwen_exo.csv`** — a separate 8-frame entire binary run, not the one cited above; reproduces the identical 50% Novice-collapse pattern (25/25 Novice, 0/25 Expert) on both views.

**Pattern**: Qwen2.5-VL has a strong "Novice" prior for climbing regardless of frame count (8/16/32) or trim condition — binary framing collapses completely; fourclass/reasoning/structured only partially escape the prior, and only ever populate Novice + Intermediate Expert, never Early or Late Expert (with the single exception of the patched 16-entire-exo reasoning run above).

### 3.3 VideoLLaVA — usable subset only

Per the user's instruction, only the conditions that produced usable (non-collapsed, non-failed) output are reported for climbing; everything else failed as summarized in §3.3b.

| Prompt | Frames | View | Acc | Notes |
|---|---|---|---|---|
| Binary | 16 | ego | 32% | *below* chance; non-collapsed (Expert 16%, Novice 48% per-class) |
| Binary | 16 | **exo** | **76%** | best single result in the whole climbing arm; sig. p<0.001 |
| Binary | 32 | ego/exo | 0% | total failure (see below) — **excluded** |
| Binary | 8 | ego/exo | 50% | 100% Novice collapse — **excluded as uninformative** |
| Binary (trimmed) | 8/16 | ego/exo | 50-60% | mostly Novice collapse; trimmed exo-8 (60%, 5/25 Expert predicted) is the only partially-informative trimmed-binary run — kept, flagged as weak |

**Why only 16-frame binary (and one 8-frame trimmed-exo variant) survive for climbing**: VideoLLaVA's fourclass output collapses to 96-100% "Novice" regardless of frame count; its 16-frame *fourclass* and *reasoning* runs failed outright (`fourclass_n16.csv` → ~90% `Unknown`; `reasoning_n16.csv` → 100% literal `"ERROR"`; `structured_n16_vl.csv` → 100% literal `"..."`). **Real cause**: VideoLLaVA's own vision-token encoding of 16 frames pushes the prompt over its 4096-token context limit. `diss_climb/scripts/videollava/trimmed/reasoning_n16.py:80` and `structured_n16.py:83` both have a hard-coded `if n_tokens > 4096: exit/abort` guard, so those two abort generation entirely rather than truncating — hence the literal `"ERROR"`/`"..."` in every row. `trimmed/fourclass_n16.py` has no such guard, so instead of aborting it silently proceeds with the over-length prompt and emits garbage that gets parsed as `Unknown` (95/83 of 100 rows) rather than crashing outright — same root cause, different failure shape. The 32-frame binary run produced empty/`ERROR` cells for 48/50 rows (0% accuracy, not a real signal) — a more severe instance of the same token-budget problem. Only the 16-frame binary prompt — the shortest, simplest output format — stayed under budget and got the model to actually discriminate between Novice and Expert rather than defaulting to a fixed label or breaking. 8-frame reasoning/structured produced *some* usable text (reported below as a partial exception) but with heavy boilerplate reuse across clips (documented in the audit), so treat those numbers as lower-confidence:

| Prompt | Frames | View | Acc | Caveat |
|---|---|---|---|---|
| Reasoning | 8 | ego | 13% | ~50% Unknown, heavy template reuse |
| Reasoning | 8 | exo | 23% | ~9-24% Unknown |
| Structured | 8 | ego | 23% | Late Expert over-predicted (26/100) |
| Structured | 8 | exo | 26% | — |
| Structured (trimmed) | 8 | ego | 30% | best VideoLLaVA structured result |

### 3.4 Qwen3-VL-8B — frame-extraction mode only (native mode abandoned)

Qwen3-VL was first tried in **native mode** (feeding the model its own video pipeline directly) but this hit an **fps=24 fallback bug causing OOM**; only 12/50 clips completed before the approach was abandoned (`qwen3vl_climbing_native_binary.csv`, all-Novice collapse, 50% acc — not informative, excluded). All subsequent Qwen3-VL runs use manual **OpenCV frame extraction** (~1fps, capped, 448px resize) — labelled "frames" in filenames.

| Prompt | View | Acc | Per-class notes |
|---|---|---|---|
| Binary | exo | 54% | Novice 100%, Expert 8% — mild improvement over Qwen2.5-VL's flat 50% |
| Binary | ego | 50% | still full Novice collapse |
| Fourclass | exo | 26% | Novice 64%, Interm. 36%, Late 4% — first climbing model to predict *some* Late Expert |
| Fourclass | ego | 37% | **best fourclass climbing result of any model**; Novice 92%, Interm. 56% |
| Reasoning | exo | 22% | most balanced spread across all 4 classes (Early 8%, Interm. 64%, Late 4%, Novice 12%) |
| Reasoning | ego | 22% | Unknown rate 27% (parser/refusal gap) |
| Structured | exo | 27% | Novice 24%, Interm. 64%, Late 20% |
| Structured | ego | 25% | Novice 56%, Interm. 44% |

**Pattern**: Qwen3-VL-8B is the only climbing model whose fourclass predictions ever meaningfully populate **Late Expert** (up to 24% structured-exo), and its ego-view fourclass result (37%) is the best fourclass accuracy achieved on climbing by any model — a genuine capability improvement over Qwen2.5-VL, though still far from strong absolute performance and still weak on Early Expert (never above 12%, mostly 0%).

---

## 4. `diss_dance` — results by model

Same task structure as climbing (Novice/Early/Intermediate/Late Expert; balanced binary/fourclass).

### 4.1 Gemini — all conditions (all trustworthy)

| Prompt | View | Acc |
|---|---|---|
| Binary | ego | 52% (not sig., p=0.89) |
| Binary | single | 60% (not sig., p=0.20) |
| Fourclass | ego | 21% (sig.) |
| Fourclass | single | 24% (sig.) |
| Reasoning | ego | 21% (sig.) |
| Reasoning | single | 32% (sig.) — Gemini's best dance result |
| Structured | ego | 27% (sig.) |
| Structured | single | 24% (sig.) |

**Pattern**: Gemini is markedly *worse* on dance than climbing (binary drops from 76%→52% on ego), and its class bias flips direction: on dance's ego view it over-predicts **Novice** (65-70/100), while on the single/combined view it over-predicts **Intermediate Expert** (81-88/100) almost identically to its climbing behavior — again never predicting Late Expert.

### 4.2 Qwen2.5-VL-7B — all conditions

Binary: **flat 50%, 100% Novice collapse in every single run** (8/16 frames, entire/trimmed, both views) — identical failure mode to climbing binary.

Fourclass/reasoning/structured — **ego view collapses to 100% Novice in almost every run** (25% or exactly 0.25 accuracy = pure chance-by-collapse), while **exo view is qualitatively different and the best Qwen result in the project**:

| Prompt | Frames | Trim | View | Acc | Notes |
|---|---|---|---|---|---|
| Fourclass | 16 | entire | exo | 33% | Novice 64%, Interm. 68% |
| Fourclass | 16 | trimmed | exo | 34% | best Qwen2.5 fourclass result |
| Structured | 16 | entire | exo | **41%** | **best Qwen2.5-VL result across the entire project**; Novice 88%, Interm. 76% |
| Structured | 16 | trimmed | exo | 39% | close second |
| Reasoning | 16 | entire | exo | 27% | Interm. 96%, Late 8% — first Qwen2.5 run to predict any Late Expert |
| Reasoning | 8 | entire | exo | 32% | Interm. 76% |

**Pattern**: dance exo-view structured/fourclass prompts break Qwen2.5-VL out of the Novice-only collapse that dominates climbing and dance-ego — this is the strongest evidence in the whole project that **view (ego vs exo) and prompt format interact with frame count**: 16 frames + exo + structured is the best-performing Qwen2.5-VL configuration found.

### 4.3 VideoLLaVA — usable subset only

Binary and fourclass for dance are **uniformly collapsed/uninformative and excluded**: every binary run (8/16 frames, entire/trimmed, both views) is a 100% "Novice" collapse at exactly 50% accuracy; every fourclass run (8 frames, entire/trimmed) is a 100% "Novice" collapse at exactly 25% accuracy. **No dance binary or fourclass VideoLLaVA result is informative.**

Only reasoning and structured (8 frames only) produced discriminative output, but with an inverse bias to climbing — **near-total "Late Expert" collapse** rather than Novice:

| Prompt | Trim | View | Acc | Notes |
|---|---|---|---|---|
| Reasoning | entire | ego | 25% | Late Expert predicted 95/100 |
| Reasoning | entire | exo | 25% | Late Expert predicted 94/100 |
| Reasoning | trimmed | ego | 31% | best VideoLLaVA dance result; Late Expert 83/100 but Early/Interm. partially recovered |
| Reasoning | trimmed | exo | 22% | Late Expert 86/100 |
| Structured | entire | ego | 23% | more balanced: Interm. 44, Novice 35, Late 21 |
| Structured | entire | exo | 19% | Interm. 48, Novice 36, Late 16 |
| Structured | trimmed | ego | 24% | Interm. 54, Late 28, Novice 18 |
| Structured | trimmed | exo | 26% | most balanced VideoLLaVA dance run |

**Why binary/fourclass failed for dance**: identical mechanism to climbing — short, single-label output formats give the model nothing to condition on beyond a fixed prior, and for dance that prior happens to be "Novice" for binary/fourclass but flips to "Late Expert" for free-text reasoning (the model apparently associates flowing, continuous dance movement across many frames with "expert," regardless of clip content) — a bias, not a genuine skill judgment; reasoning/structured are kept here only because they show *some* per-class spread rather than pure collapse, but should be treated as weak signal.

---

## 5. Basketball, JIGSAWS, 3-class, cam03, mixed-activity — generalization tests

These are all Qwen (2.5-VL unless noted) cross-domain / ablation tests run after the main climbing/dance arms, aimed at checking whether findings were domain-specific artifacts.

### 5.1 Basketball (binary, Qwen2.5-VL-7B, 8 frames, ego+exo)
- Uses the EgoExo4D "Basketball" scenario, a **different physical activity from climbing**, same skill-label schema. Benchmark had to be rebuilt (`rebuild_basketball_benchmark.py`) because 16/50 original clips had no video on disk — topped back up to 25 Novice / 25 "Late Expert" from the annotation pool.
- **Result: 24% accuracy (ego and exo identical)** — significantly *below* chance (p<0.001), and 15/50 rows (30%) are `ERROR` on both views (systemic file-access failures, not a model judgment). Excluded from headline claims about model capability; kept as evidence of pipeline fragility on non-climbing EgoExo4D scenarios.

### 5.2 JIGSAWS surgical suturing (binary, Qwen2.5-VL-7B, 8 frames, exo only)
- Explicit **cross-domain generalization test**: "surgical skill vs climbing skill," entirely different dataset (JIGSAWS, not EgoExo4D), prompt reworded for a surgeon/suturing context.
- Benchmark filtered to Novice + Expert only (Intermediate excluded) → 29 usable clips (19 Novice, 10 Expert — **not class-balanced**, unlike the climbing/dance benchmarks).
- **Result: 65.5% accuracy** — looks strong, but the model predicted **"Novice" for all 29 clips**; the accuracy is entirely an artifact of the majority class (19/29 = 65.5%) matching the model's fixed output. **Not evidence of real surgical-skill discrimination** — report this caveat explicitly if citing the number.

### 5.3 3-class collapse experiment (Qwen2.5-VL-7B via a Qwen3-VL-2B-labelled script, structured prompt, 8 frames, ego+exo, climbing)
- Hypothesis: the 4-class scale (Novice/Early Expert/Intermediate Expert/Late Expert) may be too fine-grained for the model to resolve, since Early vs Intermediate Expert are visually similar; collapsing to 3 classes should raise accuracy if that's true.
- Mapping used: **Novice→Novice, Intermediate Expert→Intermediate, Late Expert→Expert** (Early Expert clips excluded from this benchmark rather than merged — the label set only has 3 groups by construction, not a post-hoc merge of 4 model outputs).
- 279-row file (structured reasoning text is long; ~93 actual clips after CSV parsing, ego+exo).
- Confirms the project's general finding: predictions still cluster on Novice/Intermediate-equivalent labels; not a fourth model architecture, mainly a benchmark-construction ablation. Treat as a secondary/exploratory result, not a headline number — computing a clean accuracy requires re-parsing the free-text `exo_answer` column against the 3-class scale rather than reusing the 4-class `correct` logic (schema note also flagged in the data-quality audit).

### 5.4 cam03 test (alternate exocentric camera angle)
- Hypothesis: does switching the exo camera from `cam01` to `cam03` (a different physical camera position in the EgoExo4D capture rig) change results, i.e. is the model's behavior sensitive to viewpoint *within* the exo category, not just ego-vs-exo?
- Two runs: (a) Qwen2.5-VL-7B binary, 8 frames, trimmed, cam03 exo only — **50% accuracy, 100% Novice collapse**, statistically identical to the equivalent cam01 trimmed-exo run; (b) VideoLLaVA reasoning, 8 frames, cam03, ego+exo — consistent with VideoLLaVA's general reasoning weakness (9-19% accuracy region, matches `reasoning_n8_test03.csv` in the significance table).
- **Conclusion: camera angle within the exo category does not change the failure mode** — the Novice-collapse and reasoning weakness are robust to which physical exo camera is used, evidence that the problem is about *information content/prompt format*, not *specific camera placement*.

### 5.5 Mixed-activity test (Qwen2.5-VL-7B, 8 frames, trimmed, ego+exo, binary + structured)
- Combines **Basketball, Music (piano), Cooking, and Soccer** — explicitly **excludes Climbing and Dance** to test generalization to activities never seen in the main experiments. 100 clips, 25 per skill class (Novice/Early/Intermediate/Late Expert), but *not* one-activity-per-class — activities are mixed within each skill level.

| Prompt | View | Acc | By-activity breakdown (correct/n) |
|---|---|---|---|
| Binary | exo | 25% | Basketball 15/48, Cooking 10/21, Music 0/20, Soccer 0/11 |
| Binary | ego | 25% | identical to exo |
| Structured | exo | 19% | Basketball 12/48, Cooking 5/21, Music 2/20, Soccer 0/11 |
| Structured | ego | 23% | Basketball 9/48, Cooking 8/21, Music 5/20, Soccer 1/11 |

- **Pattern**: near-chance overall (25% ≈ 4-class chance, but this is *binary* framing so 25% is well below the 50% binary chance rate), and the failure is not uniform — **Music and Soccer are total failures (0-5/20-25)** while **Basketball and Cooking retain some signal**. This suggests the model's weak skill-judgment ability, such as it is, may be tied to activities with clearer visible object-interaction cues (a ball, a knife) rather than pure whole-body motion quality (music, soccer footwork) — worth flagging as a hypothesis for future work rather than a settled finding.

---

## 6. Hypotheses tested, in chronological order (for the report's narrative arc)

Reconstructed from the commit history and script comments — useful for writing an Introduction/Methodology that reads as a coherent investigation rather than a disconnected list:

1. **H1 — Baseline feasibility**: can Qwen2.5-VL and VideoLLaVA classify climbing skill (Novice/Expert) from sampled video frames at all? → Early commits (`qwen`, `Added qwen scripts`, `Added Videollava scripts`).
2. **H2 — Frame budget**: does increasing frames-per-clip (8→16→32) improve accuracy? → `qwen 32`, `frames increased`, `novice 16 & 32 frames`, `all frames all skills`. **Finding: no consistent improvement; binary framing stays at exactly 50% collapse regardless of frame count for Qwen2.5-VL.**
3. **H3 — Ego vs exo viewpoint**: does first-person (Aria) vs third-person camera change results? → `qwen ego-exo`, later systematically added to every script. **Finding: yes, substantially — e.g. Qwen2.5-VL dance structured jumps from ~25% (ego, collapsed) to 41% (exo).**
4. **H4 — Diagnostics: why does the model default to "Novice"?** → `diagnostics`, `diagnostic_frames`, `diagnostics_all_levels`: per-frame qualitative probing (asking the model to describe `person_visible`/`person_action`/`skill_from_observations` per individual frame) on a known Late-Expert clip. **Finding: early frames (setup, standing, looking at a QR code) get scored "Novice"; only frames showing active climbing motion get scored higher** — motivates H5.
5. **H5 — Trimming to the task window**: does removing setup/idle footage (via `task_start_sec`/`task_end_sec`) fix the Novice bias found in H4? → `trimmed try`, and the `entire` vs `trimmed` condition present in every later script. **Finding: only a partial fix — trimmed binary results for Qwen2.5-VL/VideoLLaVA are still mostly 50%/Novice-collapsed; trimming helps more in structured/fourclass prompts than in binary.**
6. **H6 — Prompt format**: does asking for free-text reasoning or a structured template (vs a one-word answer) change accuracy or reduce label collapse? → Reasoning and structured scripts added across all models. **Finding: yes — binary is the most collapse-prone format; structured/fourclass prompts, though still weak, show more class spread (e.g. Qwen2.5-VL dance exo structured 41% vs binary 50%-but-uninformative).**
7. **H7 — Cross-model comparison**: how does a proprietary cloud model (Gemini) compare to open-weight local models (Qwen, VideoLLaVA)? → Gemini scripts added. **Finding: Gemini is more reliable (no collapse/failure artifacts) and generally higher-accuracy on climbing binary (76% ego) but similarly weak on fourclass and on dance.**
8. **H8 — Camera-angle sensitivity within exo**: does the specific exo camera (`cam01` vs `cam03`) matter, independent of ego-vs-exo? → cam03 commits. **Finding: no — same collapse/weak-reasoning pattern regardless of exo camera used.**
9. **H9 — Cross-domain generalization**: do climbing/dance findings hold on unrelated skill domains (basketball, JIGSAWS surgery, mixed music/cooking/soccer)? → basketball, JIGSAWS, mixed commits. **Finding: mostly no signal beyond majority-class artifacts (JIGSAWS 65.5% is a Novice-collapse coincidence); Basketball and Cooking retain slightly more signal than Music/Soccer in the mixed test.**
10. **H10 — Label granularity**: is the 4-class skill scale too fine for the model to resolve (Early vs Intermediate Expert especially)? → 3-class collapse experiment. Exploratory; supports the general finding that mid-tier "Expert" sub-levels are not distinguished by any model tested.
11. **H11 — Newer model generation**: does a newer, non-fine-tuned Qwen3-VL-8B improve on Qwen2.5-VL-7B's climbing/dance results, and does its native video pipeline work? → Qwen3-VL commits. **Finding: native video pipeline has an fps=24 OOM bug and was abandoned (only 12/50 clips completed); frame-extraction mode gives modest gains — best-ever fourclass climbing accuracy (37%, ego) and the first meaningful Late Expert predictions, but binary climbing is still largely collapsed.**

---

## 7. Guidance for writing the dissertation report sections

### Abstract
- One sentence on the problem (can general-purpose VLMs judge human skill level from video, zero-shot, across activities), one on method (4 model families × 2 views × multiple frame counts × 4 prompt formats × primarily climbing/dance, secondarily basketball/JIGSAWS/mixed), one on the headline result (weak-to-moderate performance, strongly dependent on prompt format and view; Gemini most reliable, best single result 76% binary/climbing ego; open-weight models exhibit systematic "Novice" collapse under binary framing; cross-domain generalization is largely absent), and one on the takeaway (current VLMs do not have a robust visual notion of skill; apparent successes are frequently label-collapse artifacts that must be checked against per-class breakdowns and predicted-label distributions, not just overall accuracy).

### Introduction
- Motivate *why* automatic skill assessment from video matters (coaching feedback, remote training, sports/surgical education) and why VLMs are an attractive zero-shot alternative to training bespoke action-quality-assessment models.
- State the research questions directly from §6's hypotheses — frame as an escalating investigation (feasibility → ablations → why-does-it-fail diagnostics → generalization), not a flat list.
- Preview the key finding up front: naive accuracy is misleading without checking for single-label collapse; this becomes a methodological contribution of the report, not just a footnote.

### Background / Related Work
- Video-language models and their video-understanding pipelines (frame sampling vs native video ingestion — reference the Qwen3-VL native-pipeline bug as a concrete illustration of an active engineering problem, not just a citation).
- Action-quality-assessment / skill-assessment literature (traditionally pose-based or supervised), contrasted with the zero-shot VLM approach here.
- EgoExo4D dataset and its proficiency-demonstrator annotations (ego vs exo capture rig, `task_start_sec`/`task_end_sec`, scenario taxonomy incl. Bouldering, Dance, Basketball) — this is the backbone dataset; JIGSAWS is introduced separately as an out-of-distribution surgical dataset.
- Binary vs multi-class skill scales in prior work, motivating the Novice/Early/Intermediate/Late Expert schema and the later 3-class ablation.

### Methodology
- Models: Qwen2.5-VL-7B, Qwen3-VL-8B, VideoLLaVA, Gemini — state exact checkpoints/versions used (pull from script `MODEL_PATH`/API references) and hardware (local GPU inference for Qwen/VideoLLaVA vs API for Gemini — note this explains some of the data-quality asymmetry, e.g. Gemini not hitting local missing-video errors).
- Data: EgoExo4D climbing/dance/basketball subsets + JIGSAWS + a 4-activity mixed set; describe benchmark construction, balancing (25/class), and the video-availability audit/rebuild step (be upfront about the missing-file issue and how it was diagnosed and handled — this is good methodological transparency, not a weakness to hide).
- Prompting: describe all 4 prompt formats with example text (pull verbatim from `QUESTION`/prompt strings in the scripts) and the parsing/scoring logic (`check()` functions) including the intentional Late-Expert-scores-as-Expert binary convention.
- Ablation axes: frame count (8/16/32), trim condition, view (ego/exo/single), camera (cam01/cam03) — present as a table of the full experimental grid actually run (not all cells were filled for all models; be explicit about what's missing and why, e.g. VideoLLaVA 32-frame failure).
- Evaluation: binomial test for binary, chi-squared goodness-of-fit vs uniform for fourclass, and the crucial addition of **predicted-label distribution reporting** to catch collapse — explain this as a deliberate methodological safeguard developed *during* the project (motivated by H4's diagnostic finding), which is itself a contribution.

### Results & Discussion
- Organize by activity (climbing, dance) then by generalization tests, mirroring §3-§5 above; for each, always pair overall accuracy with the per-class breakdown and predicted distribution — never report a bare accuracy number without checking for collapse, and say so explicitly as your reporting convention.
- Dedicate a subsection to the **collapse phenomenon** itself as a finding, not just a caveat: quantify how many of the ~130+ experimental conditions run are collapse/failure artifacts vs genuine signal (rough tally: the large majority of VideoLLaVA binary/fourclass runs and most Qwen2.5-VL binary runs are collapsed; Gemini and Qwen3-VL are comparatively collapse-resistant).
- Discuss the ego/exo and prompt-format interactions concretely (Qwen2.5-VL dance exo-structured 41% vs ego-binary 25%-collapsed) as the strongest evidence that framing/viewpoint, not underlying model capability alone, drives observed accuracy.
- Discuss cross-domain generalization results honestly: JIGSAWS' 65.5% must be explained as a majority-class artifact, not a success; the mixed-activity per-activity breakdown (Basketball/Cooking > Music/Soccer) is worth a paragraph as a candidate explanation (object-interaction cues vs whole-body motion quality) but should be flagged as speculative given n=11-48 per activity.
- Compare model generations (Qwen2.5-VL vs Qwen3-VL) as a "does scale/recency help" discussion — modest, uneven gains, plus a concrete engineering failure (native video OOM bug) worth reporting as a limitation of newer tooling, not just the model.

### Conclusion
- Restate the core finding: current zero-shot VLMs show inconsistent, mostly weak ability to judge human skill level from video; where accuracy looks strong, verify it isn't degenerate single-label prediction.
- Note what *did* work: Gemini's binary climbing-ego result, Qwen3-VL's fourclass ego climbing result, and Qwen2.5-VL's exo-structured dance result — each a different model/condition, suggesting no single "best" configuration but rather activity- and format-dependent islands of competence.
- Limitations: local video-file availability gaps affecting Qwen/VideoLLaVA but not Gemini (confounds direct model comparison on some clips), unbalanced JIGSAWS benchmark, small per-activity n in the mixed test, reasoning-text parser gaps.
- Future work: pull directly from the hypothesis list's open threads — re-run JIGSAWS class-balanced, investigate the still-unexplained `uniandes_bouldering_027_87` pipeline bug, extend the 3-class ablation to a full re-scored comparison, test whether object-interaction visibility predicts model accuracy across more mixed activities, and re-attempt Qwen3-VL native video ingestion once the fps-fallback bug is fixed.

---

## 8. Where to find things

- **Computed accuracy for Gemini/Qwen2.5-VL/VideoLLaVA (climbing+dance)**: `statistics/master_results_summary.csv`, significance tests in `statistics/statistical_significance_summary.csv`.
- **Data-quality audit**: `results_review_notes.md` (project root).
- **Diagnostic per-frame probes**: `diss_climb/results/diagnostics_all_levels/*.txt`.
- **Charts**: `visualizations/chart1_label_collapse_heatmap.png` (directly supports the collapse discussion), `chart2_accuracy_vs_chance.png`, `chart3_cross_domain_scatter.png`, `chart4_fourclass_distribution.png`, `chart5_frame_trim_ablation.png`, `chart6_exo_ego_asymmetry.png`, plus `binary_climbing_accuracy.png`.
- **Qwen3-VL, mixed-activity, JIGSAWS, basketball, cam03, 3-class accuracy**: computed fresh for this report (§3.4, §5) — not yet in `master_results_summary.csv`; consider adding them there via `statistics/build_master_summary.py` before final submission so all numbers come from one reproducible pipeline.
