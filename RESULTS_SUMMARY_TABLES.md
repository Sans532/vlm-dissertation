# Results Summary — Simplified Tables

*Derived from `DISSERTATION_RESULTS.md` (2026-08-13 pass, updated 2026-08-18 with dance n64). Every number here is a **Kept** result unless a table explicitly shows Collapsed rows for contrast — see that document's exclusion rule for what "Kept" vs "Collapsed" means. Chance baselines: 50% binary, 25% four-class/reasoning/structured, 33.3% three-class.*

---

## 1. Headline: best result per model

The single best **Kept** accuracy each model reached, anywhere in the project.

| Model | Task | Condition | Accuracy | Chance | Above chance |
|---|---|---|---|---|---|
| Gemini | Binary | Climbing, ego | **76.0%** | 50% | +26pp — only unambiguous above-chance result in the whole project |
| VideoLLaVA | Binary | Climbing, exo, 16 frames | **76.0%** | 50% | +26pp — but a single high outlier; other VideoLLaVA binary runs sit at 60%, 32% (below chance), or fully collapsed |
| Qwen2.5-VL-7B | Structured | Dance, exo, 16 frames | **41.0%** | 25% | +16pp — best result of any kind from this model |
| Qwen3-VL-8B | Fourclass | Climbing, ego | **37.0%** | 25% | +12pp — best fourclass result of any model |

**Reading:** only Gemini's climbing binary-ego result is both large in margin and not a one-off outlier surrounded by chance-level neighbors. Every other "best" number above is the peak of a noisy band, not a stable effect.

---

## 2. Best Kept accuracy by model × activity

| Model | Best — Climbing | Best — Dance |
|---|---|---|
| Gemini | 76.0% (binary, ego) | 60.0% (binary, exo) |
| Qwen2.5-VL-7B | 32.0% (reasoning, exo, 64 frames) | 41.0% (structured, exo, 16 frames) |
| VideoLLaVA | 76.0%* (binary, exo, 16 frames) | 31.0% (reasoning, trimmed, ego) |
| Qwen3-VL-8B | 37.0% (fourclass, ego) | 30.0% (structured, exo) |

*\*Caution: VideoLLaVA's 76% is one strong run inside an otherwise collapsed/chance-level binary family — see §1.*

---

## 3. Binary prompt: the universal collapse

Binary (Novice vs. Expert) is the most failure-prone prompt format in the whole project.

| Model | Pattern across all documented binary conditions |
|---|---|
| Gemini | **Never collapses.** All 4 conditions (climbing/dance × ego/exo) land 52–76%, i.e. at or above chance. |
| Qwen2.5-VL-7B | **Always collapses.** Every documented condition — 8/16/64 frames, entire/trimmed, ego/exo, both activities, plus the best-exo-camera probe — predicts "Novice" for ~100% of clips. Accuracy is exactly 50% (chance) by construction, every time. |
| VideoLLaVA | **Mostly collapses.** All 8 documented dance conditions collapse to 100% Novice. Climbing is mixed: 5 of 8 conditions collapse, but 3 escape it (16fr entire exo 76%, 16fr entire ego 32%, 8fr trimmed exo 60%). |
| Qwen3-VL-8B | **View-dependent.** Ego view always collapses (both activities). Exo view narrowly escapes collapse — climbing 54%, dance 50% (exactly chance). |

**Reading:** binary framing is the format models are worst at handling honestly — three of four models default to a single label almost regardless of what's in the video.

---

## 4. Ego vs. exo view: the four-class / reasoning / structured pattern

For the harder prompt formats (fourclass, reasoning, structured), a consistent view effect shows up for the frame-sampling models.

| Model | Ego view | Exo view |
|---|---|---|
| Qwen2.5-VL-7B | Collapses to ~100% Novice on almost every fourclass/reasoning condition, both activities | Resolves partial discrimination — this is where nearly all of this model's Kept results live |
| VideoLLaVA (dance) | **Inverted:** reasoning-ego is Kept (25–31%) | reasoning-exo collapses to Late Expert (88–100%) |
| VideoLLaVA (climbing) | Kept but weak (mostly 15–23%) | Kept, similar range (16–26%) |
| Qwen3-VL-8B | Mixed — fourclass-ego is this model's single best result (37%) | Generally Kept, comparable accuracy to ego |
| Gemini | Both views produce usable, non-collapsed results throughout | Both views produce usable, non-collapsed results throughout |

**Reading:** for the two fixed-frame-sampling models, "exo" is generally the safer default view — except VideoLLaVA on dance, which flips the pattern entirely. Gemini is the only model indifferent to view.

---

## 5. Does more frames help? (Qwen2.5-VL frame-count ablation, 8/16/64)

The only place frame count was tested beyond 8 vs. 16 (adds a 64-frame condition, 8× the standard). Originally climbing-only; dance n64 results were added 2026-08-18.

**Climbing**

| Prompt | View | 8 frames | 16 frames | 64 frames |
|---|---|---|---|---|
| Reasoning | exo | 18.0% | 18.0% | **32.0%** — best Qwen2.5-VL climbing result in the project |
| Reasoning | ego | — | — | 29.0% |
| Fourclass | ego/exo | 24–27% | 25–27% | 25–26% (flat) |
| Structured | ego/exo | 19–24% | 24–26% | 20–21% (flat) |

**Reading:** more frames only helped once — reasoning-exo — and it changed *what* the model defaults to (from Novice-leaning to Early/Intermediate-leaning) rather than fixing the underlying collapse tendency. Fourclass and structured prompts are flat regardless of frame count; binary stays collapsed at every frame count tested (8/16/64).

**Dance**

| Prompt | View | 8 frames | 16 frames | 64 frames |
|---|---|---|---|---|
| Reasoning | exo | 32.0% | 26.0% | 20.0% (below both) |
| Fourclass | exo | 31.0% | 33.0% | 31.0% (flat) |
| Structured | ego | 25.0% | 23.0% | 26.5%\* (flat) |
| Structured | exo | 32.0% | **41.0%** — best Qwen2.5-VL result of any kind | 27.0% (below both) |

\*Structured-ego n64 accuracy is computed over 83/100 valid answers (17 unparseable), unlike every other cell here.

**Reading:** unlike climbing, dance gets no benefit from 64 frames at all — every prompt/view either stays flat or drops below its 8/16-frame values. The one genuinely interesting frame-count effect in the whole project (reasoning-exo climbing, 18%→32%) does not generalize to the other activity.

---

## 6. Cross-domain generalization

Tests on activities/scales outside the core climbing/dance benchmark.

| Test | Model | Accuracy | Verdict |
|---|---|---|---|
| Basketball, binary | Qwen2.5-VL-7B | 24.0% | Below chance; 32% of rows unparseable from missing video files |
| JIGSAWS surgical suturing, binary | Qwen2.5-VL-7B | 65.5% | **Not real signal** — 100% Novice collapse; number is a majority-class artifact of an imbalanced 19N/10E test set |
| Mixed-activity (basketball/music/cooking/soccer), binary | Qwen2.5-VL-7B | 25.0% | 100% Novice collapse, both views |
| Mixed-activity, structured, exo | Qwen2.5-VL-7B | 19.0% | Kept — signal concentrates in Basketball (12/48) and Cooking (5/21); Music (2/20) and Soccer (0/11) are near-total failures |
| Mixed-activity, structured, ego | Qwen2.5-VL-7B | 23.0% | Kept — same activity-skew pattern |
| Camera placement (cam03 vs. cam01) | Qwen2.5-VL / VideoLLaVA | No change | Camera angle *within* "exo" does not affect the collapse pattern, at any prompt format tested |
| Per-clip best-exo-camera selection | Qwen2.5-VL-7B | No change | Choosing each clip's annotated best exo camera (vs. a fixed cam01) doesn't rescue the binary collapse either |

**Reading:** the one place cross-domain testing surfaces a real (if weak) capability difference is by *activity type* — object-interaction-heavy activities (basketball, cooking) carry more signal than motion-only ones (music, soccer).

---

## 7. Text-only vs. video: the commentary control

Given only an expert's written coaching commentary (no video at all), can the model recover skill level from language alone?

| Model | Text-only, 4-class | Best video 4-class family (climbing) | Verdict |
|---|---|---|---|
| Qwen2.5-VL-7B | 26.0% | ~27% (16fr fourclass-exo) / 32% (64fr reasoning-exo) | Same band as video — text does not clearly help or hurt |
| VideoLLaVA | 33.0% | ~30% (structured, trimmed, ego) | Text-only is this model's **second-best** fourclass-family result of any modality |

**Reading:** text-only commentary performs about as well as video, in the same ~20–33% band. This suggests the bottleneck isn't specifically *visual* grounding — describing exactly what an assessor would need to see doesn't help either — pointing instead to a more fundamental difficulty mapping either modality onto this four-class proficiency scale.

---

## 8. Label granularity: does collapsing 4 classes to 3 help? (Qwen2.5-VL, structured, climbing)

| View | Overall | Novice recall | Intermediate recall | Expert recall | Status |
|---|---|---|---|---|---|
| Ego | 26.7% (8/30) | 0.0% | 40.0% | 40.0% | Kept, but barely — still resolves only 2 of 3 classes |
| Exo | 30.0% (9/30) | 0.0% | 10.0% | 80.0% | Collapsed — 22 of 30 raw predictions are "Expert" |

**Reading:** reducing the label set from 4 classes to 3 (chance rises from 25% to 33.3%) doesn't produce a cleaner result — both views still fail to resolve Novice at all, and exo just trades one collapse target (Novice) for another (Expert).

---

## 9. Chance baselines (reference)

| Task format | Chance accuracy | Split |
|---|---|---|
| Binary (Novice/Expert) | 50.0% | 25/25 |
| Four-class (Novice/Early/Intermediate/Late Expert) | 25.0% | 25 per class |
| Four-class, large-scale benchmark (n=389) | ~25.7% | 100/100/100/89 |
| Three-class ablation (Novice/Intermediate/Expert) | 33.3% | 10/10/10 |

---

## Bottom line

Across every model, activity, and prompt format tested, **exactly one result** — Gemini, binary, climbing, ego view, 76.0% — is both clearly above chance and not surrounded by chance-level neighbors from the same model on the same task. Every other elevated number in this project sits inside a noisy band a few points above chance, is a single outlier next to otherwise-collapsed sibling conditions, or is a majority-class artifact rather than genuine discrimination. The most reliable finding isn't a capability — it's a failure mode: binary and ego-view prompts collapse models to a single default label far more often than they produce real skill judgments.
