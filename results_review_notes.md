# Results Review Notes (auto-generated audit, 2026-07-26)

Scope: every CSV under `diss_dance/results/{gemini,qwen,videollava}` and
`dissertation_v2/results/{gemini,qwen,videollava}` (including `trimmed/`
subfolders). Read row-by-row with pandas; checked nulls, `correct`-column
logic, predicted-vs-text consistency, garbage/refusal/truncated text,
duplicate clip_ids, and label/schema consistency. This file is a checklist —
nothing has been modified.

---

## GEMINI

Overall very clean — 13 of 16 files had zero issues. Note: for binary files,
`ground_truth="Late Expert"` scored `correct=True` against `predicted="Expert"`
is **intentional** (see `binary.py` `check()`), not a bug.

- **`dissertation_v2/results/gemini/gemini_climbing_entire_binary.csv`** —
  clip_id `97b78075-0a0c-4198-bc78-b08a2c92b0aa` (take `uniandes_bouldering_027_56`)
  appears **twice**: row 0 is a stale failed attempt (`answer=ERROR`,
  `predicted=Unknown`, `correct=False`), row 1 is the successful retry
  (`answer=Novice`, `correct=True`). File has 51 rows instead of 50 — a retry
  was appended without removing the earlier failed row. Recommend
  de-duplicating, keeping the last row per clip_id.
- **`diss_dance/results/gemini/gemini_dance_entire_reasoning_ego.csv`** — 3
  rows where the model evaluated the camera operator/videographer instead of
  a dancer (egocentric clip shows filming/setup, not a performance), so it
  correctly refused and got scored `False`:
  - row idx 2, clip_id `7d810b75-87a9-4b95-86d0-c970ae6d03a5` (take `uniandes_dance_023_26`)
  - row idx 48, clip_id `6b3a51f3-973e-44de-b300-ed1c25bc82d3` (take `uniandes_dance_008_8`) — evaluates the *camera operator's filming technique*, not the dancer
  - row idx 65, clip_id `3abfad56-574e-41ad-abb1-6a002c69a5c7` (take `uniandes_dance_024_17`)
- **`dissertation_v2/results/gemini/gemini_climbing_entire_reasoning_ego.csv`** —
  row idx 40, clip_id `1d42bb0b-f2a3-4e24-9658-a27c3fceee4e` (take `uniandes_bouldering_022_21`):
  clip shows a briefing/meeting, not climbing; model correctly refuses,
  `predicted=Unknown`.

No issues: `gemini_dance_entire_binary.csv`, `gemini_dance_entire_binary_ego.csv`,
`gemini_dance_entire_fourclass.csv`, `gemini_dance_entire_fourclass_ego.csv`,
`gemini_dance_entire_reasoning.csv`, `gemini_dance_entire_structured.csv`,
`gemini_dance_entire_structured_ego.csv`, `gemini_climbing_entire_binary_ego.csv`,
`gemini_climbing_entire_fourclass.csv`, `gemini_climbing_entire_fourclass_ego.csv`,
`gemini_climbing_entire_reasoning.csv`, `gemini_climbing_entire_structured.csv`,
`gemini_climbing_entire_structured_ego.csv`.

---

## QWEN

No duplicate clip_ids, no `correct`-logic errors, and no missing cells found
anywhere. Main problems are (a) a large, systemic set of clips whose API/video
call failed outright, and (b) schema drift across nominally-parallel files.

**Systemic failed-clip set** — this recurring group of clip_ids shows up as
`ERROR`/`Unknown` on **both exo and ego** across almost every climbing
reasoning/structured/fourclass file (entire and trimmed, n8 and n16):
`0ec8c380…`, `e77f05f8…`, `d065cac0…`, `368bcbb7…`, `783a070b…`, `50b31b18…`,
`fe9cc639…`, `8d755767…`, `f994a8f3…`, plus `06846813…` and `495c922c…` in the
trimmed set. Worth checking whether these specific source videos are corrupt
or too short.

- **`dissertation_v2/results/qwen/qwen_climbing_entire_n8_structured.csv`** —
  stands out badly: **31/100 exo rows and 73/100 ego rows** are
  `Unknown`/`ERROR`, far above every sibling file (~9-10 failures/100). Looks
  like a broken/partial run specific to this file.
- **`dissertation_v2/results/qwen/binary_basketball.csv`** — 15/50 rows (30%)
  have `exo_answer`/`ego_answer == 'ERROR'` on both views: clip_ids
  `36eca438…`, `e1534767…`, `ff6785b7…`, `e15c972f…`, `819780c3…`,
  `9512a137…`, `4f8aee02…`, `0fcfb9c4…`, `5fa3622d…`, `a658af3b…`,
  `c0ffb507…`, `ced0e340…`, `beb755b3…`, `6390eb50…`, `52932faa…`, `4939903a…`.
- **`dissertation_v2/results/qwen/binary_n32.csv`**,
  **`binary_n8_qwen_ego.csv`**, **`binary_n8_qwen_exo.csv`**,
  **`qwen_climbing_entire_n16_binary.csv`**,
  **`qwen_climbing_entire_n8_binary.csv`** — same 2 clips fail on both views:
  `6bdb5463…`, `fea346bb…`.
- **`dissertation_v2/results/qwen/structured_eval.csv`** — same clip set/failures
  as `qwen_climbing_entire_n8_reasoning.csv`/`n8_structured.csv`; looks like a
  redundant/leftover duplicate output file — check if still needed.
- **`3class_structured.csv`** — 2 rows with `ego_predicted='Unknown'`:
  clip_ids `5c6adb2f…` (row 14), `8392e60f…` (row 17). Also note: this file
  uses a 3-class label set (Expert/Intermediate/Novice) collapsed from a
  4-class `ground_truth_original` — intentional, but a different "ground_truth"
  meaning than every other file, easy to mix up in downstream scripts.
- **Minor non-ascii artifact** (`è` character from a hallucinated backdrop
  detail, cosmetic only) in a handful of dance structured rows: e.g.
  `diss_dance/results/qwen/qwen_dance_entire_n8_structured.csv` rows
  `92ee422b…`, `6e4400c5…`, `cbc710de…`, `dd258fcf…`, and the n16 counterpart.
- **Occasional `Unknown` predictions** with no clear failure cause in
  `qwen_dance_entire_n8_structured.csv` (rows `50e6c59b…`, `5b5a6017…`,
  `f60ab291…`, `7e4cdc23…`) and `qwen_dance_entire_n16_structured.csv`
  (`d351ce4f…`, `657b877d…`).

**Schema inconsistency worth normalizing before cross-model comparison:**
binary files use at least 4 different column layouts across the corpus
(`exo_answer/exo_predicted/exo_correct` + ego triplet vs. `gt_binary` +
`exo_answer/exo_correct` with no `predicted` vs. prefixed
`binary_exo_answer`/`binary_ego_answer` vs. single-view
`ground_truth,answer,predicted,correct`); the 4 trimmed climbing binary files
also add extra `task_start`/`task_end` columns not present elsewhere.
Binary ground-truth wording also varies between "Novice/Expert" and
"Novice/Late Expert" across files meant to represent the same distinction.

No issues found: `qwen_dance_entire_n8_binary.csv`, `qwen_dance_entire_n16_binary.csv`,
`qwen_dance_entire_n8_fourclass.csv`, `qwen_dance_entire_n16_fourclass.csv`,
all `diss_dance/results/qwen/trimmed/` binary/fourclass files,
`qwen_climbing_entire_n16_fourclass.csv`, and the 4
`qwen_climbing_trimmed_{ego,exo}_n{8,16}_binary.csv` files.

(Note: an automated repetition-heuristic flagged many reasoning/structured
rows as "looping text" across Qwen files — manual spot checks showed these
are just template-driven, coherent multi-paragraph reasoning, not corruption.
Not listed as issues above.)

---

## VIDEOLLAVA

No duplicate clip_ids and no `correct`-logic errors anywhere. But this model's
output is far more degenerate than Gemini/Qwen: total single-label collapse
on several binary/fourclass files, whole fully-failed files, and heavy
copy-pasted boilerplate reasoning across unrelated clips. Flagging clearly
since it affects how much weight these results can bear.

**Total "always predicts Novice" collapse** (100% of predictions on the
non-error rows are `Novice`, regardless of ground truth):
- `diss_dance/results/videollava/vl_dance_entire_n16_binary.csv` (50/50 rows, both views)
- `diss_dance/results/videollava/vl_dance_entire_n8_binary.csv` (50/50, both views)
- `diss_dance/results/videollava/vl_dance_entire_n8_fourclass.csv` (100/100, both views)
- `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n16_binary.csv` (50/50)
- `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_binary.csv` (50/50)
- `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_fourclass.csv` (100/100)
- `dissertation_v2/results/videollava/trimmed/vl_climbing_trimmed_ego_n16_binary.csv` (50/50)
- `dissertation_v2/results/videollava/trimmed/vl_climbing_trimmed_ego_n8_binary.csv` (50/50)
- `dissertation_v2/results/videollava/trimmed/vl_climbing_trimmed_exo_n16_binary.csv` (50/50)
- `dissertation_v2/results/videollava/vl_climbing_entire_n8_binary.csv` (48/50 non-error rows all Novice)

**Fully-failed files (no usable model output at all):**
- `dissertation_v2/results/videollava/structured_n16_vl.csv` — `exo_full_answer`
  and `ego_full_answer` are literally `"..."` for all 100/100 rows;
  `predicted="Unknown"` throughout.
- `dissertation_v2/results/videollava/trimmed/reasoning_n16.csv` — both answer
  columns are literally `"ERROR"` for all 100/100 rows.
- `dissertation_v2/results/videollava/binary_n32.csv` — `exo_answer`/`ego_answer`
  empty/NaN for 48/50 rows; the other 2 rows (clip_ids `6bdb5463…`, `fea346bb…`)
  are `"ERROR"`. `correct=False` for all 50 rows. Also has no `predicted`
  column at all, unlike the dance-side binary files (schema drift).
- `dissertation_v2/results/videollava/bin_struct_n16.csv` — despite the
  "structured" name, `exo_full_answer` is literally the single word
  `"Novice"` for all 50/50 rows (not real reasoning text at all); `ego_full_answer`
  is also just `"Novice"` for 43/50 rows. Looks like a pipeline bug where the
  reasoning column got overwritten with a bare label.

**Parser gap** — text clearly states an intermediate/expert skill level but
`predicted="Unknown"` because the parser only matches literal label strings
like "Intermediate Expert" and misses VideoLlava's common phrasing
"intermediate skill level" / "intermediate level of expertise":
- `diss_dance/results/videollava/vl_dance_entire_n8_reasoning.csv` row 37,
  clip_id `04f0854f-7a61-4902-a5ea-cf9b9ff2785f`
- `diss_dance/results/videollava/trimmed/vl_dance_trimmed_n8_reasoning.csv`
  row 60, clip_id `f809a447-5414-4539-8bfd-2f6baa32dc1c`
- `dissertation_v2/results/videollava/vl_climbing_entire_n8_reasoning.csv` —
  systemic: exo rows 16, 26, 74 (clip_ids `97b78075…`, `a7ee65f5…`, `06846813…`)
  and 16 ego rows (rows 1, 2, 11, 14, 17, 19, 32, 36, 66, 74, 76, 82, 85, 88,
  94, 99)
- `dissertation_v2/results/videollava/trimmed/vl_climbing_trimmed_n8_reasoning.csv` —
  exo rows 4, 6, 16, 69, 97; ego rows 5, 8, 53

**Genuine mid-sentence truncation** in
`dissertation_v2/results/videollava/vl_climbing_entire_n8_reasoning.csv`:
- row 44, clip_id `4f3ad258-3d8e-430e-88be-d3f6d1db4e90` — ends "...body
  alignment and movement fluency are"
- row 54, clip_id `4d46b132-1c9e-404d-86fd-f7fc1dd4c320` — ends "...In the
  eighth frame"
- row 79, clip_id `823628bf-e32c-4b93-9ec2-0c615eb6a999` — ends "...is more
  fluid, indicating that they are"

**Duplicate `ego_full_answer` text attributed to different clips** — rows 1
and 2 of `dissertation_v2/results/videollava/vl_climbing_entire_n8_reasoning.csv`
(clip_ids `586597cc-59c5-45d5-9596-0b6506d9c0ba` and
`9ceee09a-ab19-4463-baac-0ed48d21827e`) contain byte-for-byte identical
reasoning text.

**Heavy generic/templated boilerplate reasoning** (same paragraph reused
across many unrelated clips — model isn't conditioning on the actual video):
present in most reasoning/structured files, worst in
`vl_climbing_entire_n8_reasoning.csv` (~half of exo answers are copies of a
few template paragraphs) and `vl_climbing_entire_n8_structured.csv` (50/100
exo, 41/100 ego rows duplicated).

**Recurring `ERROR` clip set** (18-20 rows out of 100 in several climbing
reasoning/structured files) — same clips fail consistently:
`0ec8c380…`, `e77f05f8…`, `d065cac0…`, `368bcbb7…`, `783a070b…`, `50b31b18…`,
`fe9cc639…`, `8d755767…`, `f994a8f3…` (matches the same systemic failure set
Qwen hit on the same clips — likely a source-video problem, not model-specific).

**Schema inconsistency:** four different binary-file layouts across the
VideoLlava corpus (with/without `predicted` column, single-view vs.
exo/ego-split, extra `task_start`/`task_end` timing columns in the trimmed
ego/exo climbing binary files). `vl_climbing_trimmed_n8_fourclass.csv` and
`vl_climbing_entire_n8_fourclass.csv` also share an identical error
fingerprint (same 2 failed clip_ids) — worth checking they aren't accidental
duplicates of the same run.

No issues found (beyond the general schema note): `vl_dance_entire_n8_structured.csv`,
`vl_dance_trimmed_n8_structured.csv`, `vl_climbing_entire_n16_binary.csv`,
`vl_climbing_entire_n8_fourclass.csv`, `trimmed/fourclass_n16.csv`,
`trimmed/vl_climbing_trimmed_exo_n8_binary.csv` (slightly less collapsed:
45/50 Novice, 5/50 Expert), `trimmed/structured_n16.csv`,
`trimmed/vl_climbing_trimmed_n8_structured.csv`.

---

## Cross-model summary

1. **A shared bad-clip set** (`0ec8c380…`, `e77f05f8…`, `d065cac0…`,
   `368bcbb7…`, `783a070b…`, `50b31b18…`, `fe9cc639…`, `8d755767…`,
   `f994a8f3…`, `06846813…`, `495c922c…`, `6bdb5463…`, `fea346bb…`) fails
   across **both Qwen and VideoLlava** climbing files. **ROOT CAUSE CONFIRMED
   (2026-07-26):** checked all 10 corresponding `uniandes_bouldering_*` take
   folders on the data server under `DATA_DIR/takes/<take_folder>/frame_aligned_videos/`
   — **9 of 10 are missing the video files entirely** (both `cam01.mp4` exo
   and `aria01_214-1.mp4` ego): `uniandes_bouldering_022_12`,
   `uniandes_bouldering_023_53`, `uniandes_bouldering_023_77`,
   `uniandes_bouldering_025_38`, `uniandes_bouldering_023_11`,
   `uniandes_bouldering_024_54`, `uniandes_bouldering_022_5`,
   `uniandes_bouldering_031_30`, `uniandes_bouldering_025_52`. These clips
   never had source video available locally, so Qwen/VideoLlava's OpenCV
   frame extraction failed outright (`ERROR`) — this is a **data-sync gap**,
   not a model or pipeline bug. Gemini didn't fail on these because it
   likely reads from a different/cloud video source rather than local
   OpenCV frame extraction.
   - **One exception**: `uniandes_bouldering_027_87` (clip `06846813…`) *does*
     exist locally with a valid 108.9s duration for both views, yet still
     shows up as a failure in Qwen/VideoLlava reasoning/structured files —
     this one has a genuine pipeline bug (not a missing-file issue) and is
     worth debugging separately (transient API error, OOM on frame
     extraction, etc.).
   - `495c922c…`, `6bdb5463…`, `fea346bb…` not yet checked against the data
     directory — recommend running the same missing-file check for these
     (they weren't in `benchmark_reasoning.json`; look them up in
     `benchmark_binary.json`/`benchmark_3class.json` instead).
2. **VideoLlava's climbing binary/fourclass results are largely
   non-informative** — several files are 100% single-label ("Novice")
   collapse or fully failed generation. Any accuracy numbers computed from
   those files should be treated as near-meaningless rather than a real
   model comparison point.
3. **Reasoning-text parsers (Qwen, VideoLlava) miss paraphrased skill-level
   phrasing** ("intermediate skill level" vs. literal "Intermediate Expert"),
   producing spurious `Unknown`/incorrect labels even when the text is
   clear — a fixable extraction bug, not a model failure.
4. **Schema drift** across nominally-parallel binary/fourclass files (for
   both Qwen and VideoLlava) will break any generic aggregation script that
   assumes one fixed column layout — needs normalizing before cross-model
   comparison.
5. **Gemini results are the most reliable** of the three, with only 1
   duplicate-row bug and 4 rows total reflecting genuine footage/ground-truth
   mismatches (egocentric clips showing the camera operator or an unrelated
   briefing instead of the labeled activity).
