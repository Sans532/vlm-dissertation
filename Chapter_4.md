# Chapter 4: Results and Discussion

This chapter presents the empirical findings of the zero-shot evaluation of Vision-Language Models (VLMs) on the task of human skill assessment. Guided by the overarching research question of whether VLMs can judge skill levels from video—and, if they fail, whether that failure is perceptual or linguistic—this section details the performance of four VLMs across two distinct video-input paradigms and multiple prompt formats. 

## 4.1 Evaluation Protocol and Reporting Conventions

To ensure rigorous evaluation and avoid misleading accuracy metrics, the following reporting conventions are employed throughout this chapter:
1. **Task Baselines**: Binary tasks (Novice vs. Expert) are evaluated on a balanced set of $n=50$ (chance = 50%). Four-class tasks (Novice, Early Expert, Intermediate Expert, Late Expert) use a balanced set of $n=100$ (chance = 25%). Binary and four-class results are reported separately, as their raw accuracies are not directly comparable.
2. **Distribution Reporting**: Because raw accuracy can obscure degenerate model behaviour, every accuracy score is paired with its per-class recall and the predicted-label distribution.
3. **Label Collapse**: A result is designated as "collapsed" if a model overwhelmingly defaults to a single label (at most one ground-truth class has non-zero accuracy). Collapsed results are reported as behavioural findings but are not treated as evidence of discriminative skill judgment.
4. **Video-Input Paradigms**: Models are grouped by their ingestion mechanism. Group 1 utilizes fixed-count frame extraction (Qwen2.5-VL-7B and Video-LLaVA-7B at 8, 16, and 32 frames). Group 2 uses duration-proportional ~1fps sampling (Gemini 3.1 Flash-Lite via native ingestion, and Qwen3-VL-8B via an OpenCV substitute after its native pipeline encountered an OOM bug).

## 4.2 Binary Skill Discrimination

The binary skill assessment task serves as the baseline feasibility test. Table 4.1 summarises the results for climbing, and Table 4.2 summarises the results for dance.

**Table 4.1: Binary results, climbing (Novice vs. Expert, balanced $n=50$).**

| Model | Input | Trim | View | Accuracy | Novice Recall | Expert Recall | Predicted Labels | Verdict |
|---|---|---|---|---|---|---|---|---|
| Gemini | native | entire | ego | **76.0%** | 84% | 68% | Nov 29, Exp 21 | Genuine signal ($p=0.0003$) |
| Gemini | native | entire | exo | 60.0% | 52% | 68% | Exp 29, Nov 21 | Not significant, non-collapsed |
| VideoLLaVA | 16 fr | entire | exo | 76.0% | 88% | 64% | mixed | Isolated artefact (ego is 32%) |
| VideoLLaVA | 16 fr | entire | ego | 32.0% | 48% | 16% | mixed | Below chance |
| Qwen3-VL | ~1fps | entire | exo | 54.0% | 100% | 8% | Nov 48, Exp 2 | Near-collapse |
| Qwen2.5-VL | 8/16/32 fr | all | all | 50.0% | 100% | 0% | Nov 50 | Total collapse ($p=1.0$) |

**Table 4.2: Binary results, dance (Novice vs. Expert, balanced $n=50$).**

| Model | Input | View | Accuracy | Novice Recall | Expert Recall | Predicted Labels | Verdict |
|---|---|---|---|---|---|---|---|
| Gemini | native | exo | 60.0% | 52% | 68% | Exp 29, Nov 21 | Not significant, non-collapsed |
| Gemini | native | ego | 52.0% | 100% | 4% | Nov 49, Exp 1 | Near-collapse |
| Qwen3-VL | ~1fps | exo | 50.0% | 92% | 8% | Nov 46, Exp 4 | Near-collapse |
| Qwen2.5-VL | all | both | 50.0% | 100% | 0% | Nov 50 | Total collapse |
| VideoLLaVA | all | both | 50.0% | 100% | 0% | Nov 50 | Total collapse |

The single most striking observation is the pervasive collapse in the open-weight models. Qwen2.5-VL-7B invariably predicts "Novice" for 100% of clips across all tested binary conditions—resulting in a fabricated 50% accuracy by virtue of the balanced dataset. Gemini provides the only unambiguous, genuine above-chance result in the entire project, achieving 76.0% on the climbing egocentric view. This demonstrates that while the binary format is highly susceptible to collapse, genuine discriminative judgment is possible, albeit only by the most capable proprietary model.

## 4.3 Four-Class Skill Classification

When the granularity is increased to four skill levels, no model produces a reliable above-chance result. Accuracies hover around the 25% chance baseline, and apparent improvements are largely artefactual.

**Table 4.3: Climbing, representative four-class results.**

| Model | Prompt | Input | View | Acc | Per-class Recall (N/E/I/L) | Note |
|---|---|---|---|---|---|---|
| Qwen3-VL | fourclass | ~1fps | ego | **37%** | 92 / 0 / 56 / 0 | Best four-class climbing result |
| Gemini | structured | native | exo | 33% | 40 / 0 / 92 / 0 | Gemini's best climbing result |
| VideoLLaVA | structured | 8 fr | ego | 30% | 60 / 4 / 12 / 44 | Substantial Late-Expert recall |
| Qwen2.5-VL | fourclass | 16 fr | exo | 27% | 92 / 0 / 16 / 0 | Exo view resolves partial signal |

**Table 4.4: Dance, representative four-class results.**

| Model | Prompt | Input | View | Acc | Per-class Recall (N/E/I/L) | Note |
|---|---|---|---|---|---|---|
| Qwen2.5-VL | structured | 16 fr | exo | **41%** | 88 / 0 / 76 / 0 | Best Qwen2.5-VL result |
| Gemini | reasoning | native | exo | 32% | 28 / 4 / 96 / 0 | Gemini's best dance result |
| Qwen3-VL | structured | ~1fps | exo | 30% | 12 / 4 / 80 / 24 | Best Qwen3-VL dance result |

A critical finding is that the "Early Expert" class is almost never resolved by any model. The highest recorded accuracy (Qwen2.5-VL on dance, 41%) is driven by a Novice bias aligning favourably with the structured prompt template, rather than genuine multi-class discrimination. Gemini, which succeeded in the binary task, silently collapses the four-point scale to a binary Novice-vs-Intermediate choice, failing to predict Early or Late Expert on climbing.

## 4.4 The Label-Collapse Phenomenon

The central finding of this dissertation is that zero-shot VLMs fundamentally fail at multi-level skill assessment due to a label-collapse failure mode. Across roughly 45% of all tested conditions, models predict a single label for 87–100% of clips, regardless of the ground-truth skill present in the video.

Interestingly, the target of this collapse moves between model generations. The frame-sampled models (Qwen2.5-VL, VideoLLaVA) heavily default to "Novice" (constituting 71% and 51% of all four-class predictions, respectively). Conversely, the video-native and adaptive models (Gemini, Qwen3-VL) shift their default prior to "Intermediate Expert" (constituting 79% of Gemini's climbing predictions). Newer models are not necessarily less biased; rather, their bias has relocated. The resulting confusion matrices show that row distributions are nearly identical, proving that the models' outputs are virtually independent of the actual skill demonstrated in the video.

## 4.5 Locating the Failure: Perception vs. Language

If models fail to classify skill, is the bottleneck in the visual perception of the scene or in the linguistic generation step? This dissertation leverages two specific controls to isolate the failure point, yielding its most precise claim.

First, diagnostic controls demonstrate that the models perceive scene content accurately. When prompted, they correctly identify the activity, subjects, and colors—indicating that visual features are successfully processed. 

Second, a linear classifier trained on frozen CLIP features extracted from the same videos achieves a 67% four-class accuracy (5-fold CV, 95% CI [62.5%, 71.5%]) on clips where the VLMs score a mere 24–25%. This proves that the raw visual embeddings contain a robust, linearly separable signal for human skill. **Vision alone succeeds.**

However, a text-only control, which provides the models with expert written commentary describing the performance in detail (without any video), results in a similar collapse at just 26% accuracy. **Language-to-language fails.**

Because the visual encoders successfully capture the necessary discriminative features, and because explicit linguistic descriptions of the skill are insufficient for the LLM to classify it, the bottleneck is firmly placed at the language-generation and reasoning step, not at visual grounding. The VLMs fail to translate the embedded perceptual reality of "skill" into the correct categorical language token.

## 4.6 Ablations and Generalization

Several methodological ablations were conducted to ensure the collapse was not a trivial artefact of prompt engineering or data formatting.

**Frame Count and Context Limits:** Increasing the frame budget in the fixed-count models (from 8 to 16 to 32) yielded no consistent improvement. In fact, Qwen2.5-VL's binary collapse remained identical across all frame counts. For VideoLLaVA, exceeding 8 frames precipitated catastrophic failure; processing 16 frames consumed its entire 4,096-token context window (ViT-L/14 produces 256 tokens per frame), leading to silent truncation and garbage outputs.
**Trimming:** Trimming videos to exactly the annotated task window did not repair the collapse, disproving the hypothesis that idle footage or setup frames confused the models.
**Cross-Domain Transfer:** The failure holds across diverse domains. In a cross-domain test on the JIGSAWS surgical suturing dataset, Qwen2.5-VL achieved 65.5% accuracy. However, this was entirely a majority-class artefact: the model blindly predicted "Novice" 100% of the time, and the dataset was imbalanced (19 Novice / 10 Expert). On a mixed-activity benchmark (basketball, music, cooking, soccer), the models exhibited the same 100% Novice collapse on the binary task (25% accuracy due to a 4-way mapping). Signal was weakly retained only on object-interaction activities (basketball, cooking), while motion-centric activities (music, soccer) were near-total failures.

## 4.7 Qualitative Analysis: What the Models Actually Say

Textual analysis of the generated responses reveals how different architectures mask their lack of discriminative ability:
- **Hedging and Disjunctions:** When given a reasoning prompt, Qwen2.5-VL frequently outputs the literal phrase "novice or early expert" rather than committing to a label, a behaviour entirely absent under structured templates.
- **Stereotypy vs. Verbosity:** Frame-sampled models suffer from severe response stereotypy; Qwen2.5-VL opens 99–100% of its egocentric reasoning answers with identical boilerplate text. In contrast, Gemini and Qwen3-VL generate 1,000–2,500 characters of highly specific, per-clip prose. However, this verbosity does not equate to grounding; despite the bespoke descriptions, their final skill judgments remain disconnected from the ground truth.
- **Principled Refusals:** On certain egocentric clips where the activity was out of frame, Gemini correctly refused to answer or evaluated the camera operator instead. While scored as incorrect, this was the most genuinely grounded visual behaviour observed, exposing inherent noise in ego-view datasets.

## 4.8 Reliability and Limitations

This evaluation uncovered several engineering frailties in modern VLM pipelines. Beyond VideoLLaVA's context limits, Qwen3-VL's native pipeline failed with an OOM bug by defaulting to 24fps instead of the requested ~1fps. Missing source files in the public benchmark induced a ~9% error floor for local models. Furthermore, traditional keyword parsers proved inadequate; early parsing attempts misread negated phrases ("not yet an Expert"), necessitating a custom aspiration-aware parser to accurately capture the model's true intent. Finally, significant run-to-run non-determinism was observed, with independent runs of Qwen2.5-VL disagreeing on up to 86% of predictions on identical data.

## 4.9 Synthesis

Trained, non-zero-shot methods report 47–53% top-1 accuracy on this exact four-class benchmark, demonstrating that the task is solvable. The results presented here position the current VLM failure strictly as a zero-shot transfer deficit. 

The evaluation reveals three distinct islands of weak competence: Gemini's binary climbing (76%), Qwen3-VL's four-class ego climbing (37%), and Qwen2.5-VL's structured exo dance (41%). There is no single superior configuration. Instead, the data cohesively demonstrates that current zero-shot VLMs do not possess a robust, linguistically accessible notion of human skill. The successful linear classification of frozen CLIP features confirms that the requisite perceptual information is extracted, but the language generation backbone is currently incapable of utilizing it.
