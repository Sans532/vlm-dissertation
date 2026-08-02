"""
Regenerates the Climbing/Dance results text reports using the corrected
(recomputed) accuracy numbers from statistics/master_results_summary.csv,
in the same layout as the originally-supplied PDFs, but with:
  - per-class breakdown counts filled in for every section (the original
    PDFs only showed this for Gemini)
  - the source CSV filename bracketed next to each subheading
  - two noted exceptions: VideoLLaVA climbing trimmed 16-frame structured/
    reasoning, whose CSVs are now empty/all-ERROR on disk (overwritten by a
    later crashed run, per this project's audit). Those two keep the
    original PDF's numbers (sourced from the run log, not the CSV), flagged
    with a note instead of a normal bracket.
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_CSV = REPO_ROOT / "statistics" / "master_results_summary.csv"

rows = list(csv.DictReader(open(MASTER_CSV)))
LOOKUP = {}
for r in rows:
    n = int(r["n"])
    acc = float(r["accuracy"])
    r["accuracy_pct"] = f"{round(acc * n)}/{n} = {acc:.1%}"
    LOOKUP[(r["file"], r["view"])] = r


def get(file_rel, view):
    return LOOKUP.get((file_rel, view))


def derive_binary_dist(file_rel, view):
    """Some binary files have no 'predicted' column (only raw answer + correct).
    Derive a Novice/Expert distribution directly from the raw answer text."""
    from collections import Counter
    path = REPO_ROOT / file_rel
    with open(path, newline="") as f:
        file_rows = list(csv.DictReader(f))
    ans_col = None
    for cand in (f"{view}_answer", f"binary_{view}_answer", "answer"):
        if file_rows and cand in file_rows[0]:
            ans_col = cand
            break
    if not ans_col:
        return "{}"
    counts = Counter()
    for r in file_rows:
        a = (r.get(ans_col) or "").lower()
        has_nov, has_exp = "novice" in a, "expert" in a
        if a == "error":
            counts["Unknown"] += 1
        elif has_nov and not has_exp:
            counts["Novice"] += 1
        elif has_exp and not has_nov:
            counts["Expert"] += 1
        elif has_nov and has_exp:
            pos_n, pos_e = a.find("novice"), a.find("expert")
            counts["Novice" if pos_n < pos_e else "Expert"] += 1
        else:
            counts["Unknown"] += 1
    items = [f"'{k}': {v}" for k, v in counts.most_common()]
    return "{" + ", ".join(items) + "}"


def fmt_breakdown(row):
    if not row or not row["class_breakdown"]:
        return []
    lines = []
    for part in row["class_breakdown"].split("; "):
        label, rest = part.split(": ", 1)
        lines.append(f"{label:22s}: {rest}")
    return lines


def fmt_dist(row):
    if not row:
        return "{}"
    parts = [p for p in row["predicted_distribution"].split("; ") if p]
    items = []
    for p in parts:
        k, v = p.rsplit(":", 1)
        items.append(f"'{k}': {v}")
    return "{" + ", ".join(items) + "}"


def section(prompt_label, trim_label, frames_label, file_rel, note=None,
            exo_view="exo", ego_view="ego"):
    out = []
    frame_suffix = f", {frames_label}" if frames_label else ""
    header = f"{prompt_label}, {trim_label}{frame_suffix}"
    bracket = f"[{Path(file_rel).name}]" if not note else f"[{Path(file_rel).name} -- {note}]"
    out.append(f"{header}  {bracket}")

    exo = get(file_rel, exo_view)
    ego = get(file_rel, ego_view)
    exo_acc = exo["accuracy_pct"] if exo else "n/a"
    ego_acc = ego["accuracy_pct"] if ego else "n/a"

    if prompt_label == "Binary":
        exo_dist = fmt_dist(exo)
        if exo_dist == "{}":
            exo_dist = derive_binary_dist(file_rel, exo_view)
        ego_dist = fmt_dist(ego)
        if ego_dist == "{}":
            ego_dist = derive_binary_dist(file_rel, ego_view)
        out.append("EXO")
        out.append(f"Overall: {exo_acc}")
        out.extend(fmt_breakdown(exo))
        out.append(f"Answers: {exo_dist}")
        out.append("EGO")
        out.append(f"Overall: {ego_acc}")
        out.extend(fmt_breakdown(ego))
        out.append(f"Answers: {ego_dist}")
    else:
        out.append(f" exo: {exo_acc}")
        out.append(f" ego: {ego_acc}")
        for l in fmt_breakdown(exo):
            out.append(f"  exo {l}")
        for l in fmt_breakdown(ego):
            out.append(f"  ego {l}")
        out.append(f"Exo predictions: {fmt_dist(exo)}")
        out.append(f"Ego predictions: {fmt_dist(ego)}")
    out.append("")
    return out


def hardcoded_section(prompt_label, trim_label, frames_label, file_rel, note,
                       exo_acc, ego_acc, exo_breakdown, ego_breakdown, exo_dist, ego_dist):
    """For the two sections we can't recompute (broken CSVs) -- reuse the original PDF numbers."""
    out = []
    frame_suffix = f", {frames_label}" if frames_label else ""
    header = f"{prompt_label}, {trim_label}{frame_suffix}"
    out.append(f"{header}  [{Path(file_rel).name} -- {note}]")
    out.append(f" exo: {exo_acc}")
    out.append(f" ego: {ego_acc}")
    for l in exo_breakdown:
        out.append(f"  exo {l}")
    for l in ego_breakdown:
        out.append(f"  ego {l}")
    out.append(f"Exo predictions: {exo_dist}")
    out.append(f"Ego predictions: {ego_dist}")
    out.append("")
    return out


def gemini_section(prompt_label, file_exo, file_ego):
    out = []
    out.append(prompt_label)
    exo = get(file_exo, "single") or get(file_exo, "exo")
    ego = get(file_ego, "ego")
    exo_acc = exo["accuracy_pct"] if exo else "n/a"
    ego_acc = ego["accuracy_pct"] if ego else "n/a"
    out.append(f"EXO  [{Path(file_exo).name}]" + " " * 10 + f"EGO  [{Path(file_ego).name}]")
    out.append(f"Overall: {exo_acc}")
    out.extend(fmt_breakdown(exo))
    out.append(f"Answers: {fmt_dist(exo)}")
    out.append(f"Overall: {ego_acc}")
    out.extend(fmt_breakdown(ego))
    out.append(f"Answers: {fmt_dist(ego)}")
    out.append("")
    return out


def separate_files_binary_section(trim_label, frames_label, file_exo, file_ego):
    """Binary section where exo and ego live in two separate single-view files."""
    out = []
    frame_suffix = f", {frames_label}" if frames_label else ""
    out.append(f"Binary, {trim_label}{frame_suffix}  [{Path(file_exo).name} / {Path(file_ego).name}]")
    exo = get(file_exo, "single") or get(file_exo, "exo")
    ego = get(file_ego, "single") or get(file_ego, "ego")
    exo_acc = exo["accuracy_pct"] if exo else "n/a"
    ego_acc = ego["accuracy_pct"] if ego else "n/a"
    exo_dist = fmt_dist(exo)
    if exo_dist == "{}":
        exo_dist = derive_binary_dist(file_exo, "exo")
    ego_dist = fmt_dist(ego)
    if ego_dist == "{}":
        ego_dist = derive_binary_dist(file_ego, "ego")
    out.append("EXO")
    out.append(f"Overall: {exo_acc}")
    out.extend(fmt_breakdown(exo))
    out.append(f"Answers: {exo_dist}")
    out.append("EGO")
    out.append(f"Overall: {ego_acc}")
    out.extend(fmt_breakdown(ego))
    out.append(f"Answers: {ego_dist}")
    out.append("")
    return out


# ===========================================================================
def build_climbing():
    d = "dissertation_v2/results"
    lines = []
    lines.append("CLIMBING- QWEN ENTIRE")
    for e in [
        ("Binary", "Entire", "8f", f"{d}/qwen/qwen_climbing_entire_n8_binary.csv"),
        ("Fourclass", "Entire", "8f", f"{d}/qwen/qwen_climbing_entire_n8_fourclass.csv"),
        ("Structured", "Entire", "8f", f"{d}/qwen/structured_eval.csv"),
        ("Reasoning", "Entire", "8f", f"{d}/qwen/qwen_climbing_entire_n8_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("CLIMBING- QWEN TRIMMED")
    lines.extend(separate_files_binary_section(
        "Trimmed", "8f",
        f"{d}/qwen/trimmed/qwen_climbing_trimmed_exo_n8_binary.csv",
        f"{d}/qwen/trimmed/qwen_climbing_trimmed_ego_n8_binary.csv"))
    for e in [
        ("Fourclass", "Trimmed", "8f", f"{d}/qwen/trimmed/fourclass_trimmed.csv"),
        ("Structured", "Trimmed", "8f", f"{d}/qwen/trimmed/qwen_climbing_trimmed_n8_structured.csv"),
        ("Reasoning", "Trimmed", "8f", f"{d}/qwen/trimmed/qwen_climbing_trimmed_n8_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("CLIMBING- VideoLLaVa ENTIRE")
    for e in [
        ("Binary", "Entire", "8f", f"{d}/videollava/vl_climbing_entire_n8_binary.csv"),
        ("Fourclass", "Entire", "8f", f"{d}/videollava/vl_climbing_entire_n8_fourclass.csv"),
        ("Structured", "Entire", "8f", f"{d}/videollava/vl_climbing_entire_n8_structured.csv"),
        ("Reasoning", "Entire", "8f", f"{d}/videollava/vl_climbing_entire_n8_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("CLIMBING- VideoLLaVA TRIMMED")
    lines.extend(separate_files_binary_section(
        "Trimmed", "8f",
        f"{d}/videollava/trimmed/vl_climbing_trimmed_exo_n8_binary.csv",
        f"{d}/videollava/trimmed/vl_climbing_trimmed_ego_n8_binary.csv"))

    for e in [
        ("Fourclass", "Trimmed", "8f", f"{d}/videollava/trimmed/vl_climbing_trimmed_n8_fourclass.csv"),
        ("Structured", "Trimmed", "8f", f"{d}/videollava/trimmed/vl_climbing_trimmed_n8_structured.csv"),
        ("Reasoning", "Trimmed", "8f", f"{d}/videollava/trimmed/vl_climbing_trimmed_n8_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("CLIMBING- Gemini")
    lines.extend(gemini_section("Binary", f"{d}/gemini/gemini_climbing_entire_binary.csv",
                                 f"{d}/gemini/gemini_climbing_entire_binary_ego.csv"))
    lines.extend(gemini_section("Fourclass", f"{d}/gemini/gemini_climbing_entire_fourclass.csv",
                                 f"{d}/gemini/gemini_climbing_entire_fourclass_ego.csv"))
    lines.extend(gemini_section("Structured", f"{d}/gemini/gemini_climbing_entire_structured.csv",
                                 f"{d}/gemini/gemini_climbing_entire_structured_ego.csv"))
    lines.extend(gemini_section("Reasoning", f"{d}/gemini/gemini_climbing_entire_reasoning.csv",
                                 f"{d}/gemini/gemini_climbing_entire_reasoning_ego.csv"))

    lines.append("CLIMBING- QWEN ENTIRE (16 frames)")
    for e in [
        ("Binary", "Entire", "16f", f"{d}/qwen/qwen_climbing_entire_n16_binary.csv"),
        ("Fourclass", "Entire", "16f", f"{d}/qwen/qwen_climbing_entire_n16_fourclass.csv"),
        ("Structured", "Entire", "16f", f"{d}/qwen/qwen_climbing_entire_n16_structured.csv"),
        ("Reasoning", "Entire", "16f", f"{d}/qwen/qwen_climbing_entire_n16_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("CLIMBING- QWEN TRIMMED (16 frames)")
    lines.extend(separate_files_binary_section(
        "Trimmed", "16f",
        f"{d}/qwen/trimmed/qwen_climbing_trimmed_exo_n16_binary.csv",
        f"{d}/qwen/trimmed/qwen_climbing_trimmed_ego_n16_binary.csv"))
    for e in [
        ("Structured", "Trimmed", "16f", f"{d}/qwen/trimmed/qwen_climbing_trimmed_n16_structured.csv"),
        ("Reasoning", "Trimmed", "16f", f"{d}/qwen/trimmed/qwen_climbing_trimmed_n16_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("CLIMBING- VideoLLaVa ENTIRE (16 frames)")
    lines.extend(section("Binary", "Entire", "16f", f"{d}/videollava/vl_climbing_entire_n16_binary.csv"))

    lines.append("CLIMBING- VideoLLaVA TRIMMED (16 frames)")
    lines.extend(separate_files_binary_section(
        "Trimmed", "16f",
        f"{d}/videollava/trimmed/vl_climbing_trimmed_exo_n16_binary.csv",
        f"{d}/videollava/trimmed/vl_climbing_trimmed_ego_n16_binary.csv"))

    # These two: CSVs are now empty/all-ERROR (overwritten by a later crashed job -- see
    # conversation history). Recomputation is impossible; keep the original report's
    # log-sourced numbers with an explicit note.
    lines.extend(hardcoded_section(
        "Structured", "Trimmed", "16f", f"{d}/videollava/trimmed/structured_n16.csv",
        "CSV is now EMPTY (0 rows) -- overwritten by a later crashed job (see job 3541299, prompt "
        "exceeded 4096-token context limit). These are the original run's numbers from log "
        "structured_n16_3541255.log; NOT independently reproducible/recomputable from current data.",
        "24/100 = 24.0%", "25/100 = 25.0%",
        [], [],
        "{'Novice': 98, 'Unknown': 2}", "{'Novice': 89, 'Unknown': 9, 'Late Expert': 2}",
    ))
    lines.extend(hardcoded_section(
        "Reasoning", "Trimmed", "16f", f"{d}/videollava/trimmed/reasoning_n16.csv",
        "CSV is now 100% ERROR (NameError: 'name re is not defined' in the run that overwrote it -- "
        "since fixed in the script, but never re-run). These are the original run's numbers from log "
        "reasoning_n16_3538647.log; NOT independently reproducible/recomputable from current data.",
        "16/100 = 16.0%", "15/100 = 15.0%",
        [], [],
        "{'Novice': 61, 'Unknown': 31, 'Early Expert': 8}",
        "{'Novice': 65, 'Unknown': 33, 'Early Expert': 1, 'Late Expert': 1}",
    ))

    return lines


def build_dance():
    d = "diss_dance/results"
    lines = []
    lines.append("DANCE- QWEN ENTIRE")
    for e in [
        ("Binary", "Entire", "8f", f"{d}/qwen/qwen_dance_entire_n8_binary.csv"),
        ("Fourclass", "Entire", "8f", f"{d}/qwen/qwen_dance_entire_n8_fourclass.csv"),
        ("Structured", "Entire", "8f", f"{d}/qwen/qwen_dance_entire_n8_structured.csv"),
        ("Reasoning", "Entire", "8f", f"{d}/qwen/qwen_dance_entire_n8_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("DANCE- QWEN TRIMMED")
    for e in [
        ("Binary", "Trimmed", "8f", f"{d}/qwen/trimmed/qwen_dance_trimmed_n8_binary.csv"),
        ("Fourclass", "Trimmed", "8f", f"{d}/qwen/trimmed/qwen_dance_trimmed_n8_fourclass.csv"),
        ("Structured", "Trimmed", "8f", f"{d}/qwen/trimmed/qwen_dance_trimmed_n8_structured.csv"),
        ("Reasoning", "Trimmed", "8f", f"{d}/qwen/trimmed/qwen_dance_trimmed_n8_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("DANCE- VideoLLaVa ENTIRE")
    for e in [
        ("Binary", "Entire", "8f", f"{d}/videollava/vl_dance_entire_n8_binary.csv"),
        ("Fourclass", "Entire", "8f", f"{d}/videollava/vl_dance_entire_n8_fourclass.csv"),
        ("Structured", "Entire", "8f", f"{d}/videollava/vl_dance_entire_n8_structured.csv"),
        ("Reasoning", "Entire", "8f", f"{d}/videollava/vl_dance_entire_n8_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("DANCE- VideoLLaVA TRIMMED")
    for e in [
        ("Binary", "Trimmed", "8f", f"{d}/videollava/trimmed/vl_dance_trimmed_n8_binary.csv"),
        ("Fourclass", "Trimmed", "8f", f"{d}/videollava/trimmed/vl_dance_trimmed_n8_fourclass.csv"),
        ("Structured", "Trimmed", "8f", f"{d}/videollava/trimmed/vl_dance_trimmed_n8_structured.csv"),
        ("Reasoning", "Trimmed", "8f", f"{d}/videollava/trimmed/vl_dance_trimmed_n8_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("DANCE- Gemini")
    lines.extend(gemini_section("Binary", f"{d}/gemini/gemini_dance_entire_binary.csv",
                                 f"{d}/gemini/gemini_dance_entire_binary_ego.csv"))
    lines.extend(gemini_section("Fourclass", f"{d}/gemini/gemini_dance_entire_fourclass.csv",
                                 f"{d}/gemini/gemini_dance_entire_fourclass_ego.csv"))
    lines.extend(gemini_section("Structured", f"{d}/gemini/gemini_dance_entire_structured.csv",
                                 f"{d}/gemini/gemini_dance_entire_structured_ego.csv"))
    lines.extend(gemini_section("Reasoning", f"{d}/gemini/gemini_dance_entire_reasoning.csv",
                                 f"{d}/gemini/gemini_dance_entire_reasoning_ego.csv"))

    lines.append("DANCE- QWEN ENTIRE (16 frames)")
    for e in [
        ("Binary", "Entire", "16f", f"{d}/qwen/qwen_dance_entire_n16_binary.csv"),
        ("Fourclass", "Entire", "16f", f"{d}/qwen/qwen_dance_entire_n16_fourclass.csv"),
        ("Structured", "Entire", "16f", f"{d}/qwen/qwen_dance_entire_n16_structured.csv"),
        ("Reasoning", "Entire", "16f", f"{d}/qwen/qwen_dance_entire_n16_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("DANCE- QWEN TRIMMED (16 frames)")
    for e in [
        ("Binary", "Trimmed", "16f", f"{d}/qwen/trimmed/qwen_dance_trimmed_n16_binary.csv"),
        ("Fourclass", "Trimmed", "16f", f"{d}/qwen/trimmed/qwen_dance_trimmed_n16_fourclass.csv"),
        ("Structured", "Trimmed", "16f", f"{d}/qwen/trimmed/qwen_dance_trimmed_n16_structured.csv"),
        ("Reasoning", "Trimmed", "16f", f"{d}/qwen/trimmed/qwen_dance_trimmed_n16_reasoning.csv"),
    ]:
        lines.extend(section(*e))

    lines.append("DANCE- VideoLLaVa ENTIRE (16 frames)")
    lines.extend(section("Binary", "Entire", "16f", f"{d}/videollava/vl_dance_entire_n16_binary.csv"))

    lines.append("DANCE- VideoLLaVA TRIMMED (16 frames)")
    lines.extend(section("Binary", "Trimmed", "16f", f"{d}/videollava/trimmed/vl_dance_trimmed_n16_binary.csv"))

    return lines


def main():
    climbing_lines = build_climbing()
    dance_lines = build_dance()

    out_dir = REPO_ROOT / "statistics"
    (out_dir / "climbing_results_recomputed.txt").write_text("\n".join(climbing_lines))
    (out_dir / "dance_results_recomputed.txt").write_text("\n".join(dance_lines))
    print(f"Wrote {out_dir / 'climbing_results_recomputed.txt'} ({len(climbing_lines)} lines)")
    print(f"Wrote {out_dir / 'dance_results_recomputed.txt'} ({len(dance_lines)} lines)")


if __name__ == "__main__":
    main()
