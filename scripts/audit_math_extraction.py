#!/usr/bin/env python3
"""Audit math recovery extraction quality (categories A–E)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cot_segmentation import detect_degenerate_trace  # noqa: E402
from evaluator import extract_domain_answer, is_correct  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())


def _gold_variants(gold: str) -> list[str]:
    """String forms gold may take in raw text (handles \\frac vs a/b)."""
    g = str(gold).strip()
    variants = {g, _norm(g)}
    frac = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}", g)
    if frac:
        variants.add(f"{frac.group(1)}/{frac.group(2)}")
        variants.add(f"\\frac{{{frac.group(1)}}}{{{frac.group(2)}}}")
        variants.add(f"\\dfrac{{{frac.group(1)}}}{{{frac.group(2)}}}")
    slash = re.fullmatch(r"(-?\d+)/(-?\d+)", g)
    if slash:
        variants.add(f"\\frac{{{slash.group(1)}}}{{{slash.group(2)}}}")
        variants.add(f"\\dfrac{{{slash.group(1)}}}{{{slash.group(2)}}}")
    return [v for v in variants if v]


def _present(haystack: str, gold: str) -> bool:
    hay_norm = _norm(haystack)
    for v in _gold_variants(gold):
        if v in haystack or _norm(v) in hay_norm:
            return True
    g = str(gold).strip()
    if re.search(rf"(?<!\d){re.escape(g)}(?!\d)", haystack):
        return True
    if re.search(rf"=\s*{re.escape(g)}\b", haystack):
        return True
    return False


def gold_in_text(raw: str, gold: str) -> bool:
    return _present(raw, gold)


def get_final_region(raw: str) -> str:
    body = raw
    idx = raw.rfind("</think>")
    if idx != -1:
        body = raw[idx + len("</think>") :].strip() or raw
    markers = (
        r"\\boxed\{",
        r"\*\*Final\s+Answer",
        r"\*\*Answer",
        r"\*\*Conclusion",
        r"Final\s+Answer:",
    )
    last_pos = -1
    for marker in markers:
        for match in re.finditer(marker, body, re.I | re.M):
            last_pos = max(last_pos, match.start())
    # No explicit marker: only the last couple of lines count as "final"
    # (a gold token deep in scratch work must NOT be treated as the final answer).
    return body[last_pos:] if last_pos >= 0 else body[-250:]


def gold_in_final(raw: str, gold: str) -> bool:
    return _present(get_final_region(raw), gold)


def has_contradiction(raw: str, gold: str) -> bool:
    final = get_final_region(raw)
    answers = re.findall(r"\*\*Answer:?\*\*\s*(.+)$", final, re.I | re.M)
    if len(answers) > 1:
        return True
    nums = set(re.findall(r"\b\d{2,4}\b", final[-400:]))
    g = str(gold).strip()
    if g.isdigit() and len(nums) >= 2 and g in nums and any(n != g for n in nums):
        return True
    return False


def classify_run(run: dict) -> str:
    """
    Category by whether GOLD is the model's FINAL answer (not scratch work).

    A: graded correct, gold in final region.
    B: graded incorrect, gold NOT in final region (model's final answer is wrong).
    C: graded incorrect, gold IS in final region (parser failed to pick it) — real bug.
    D: graded correct, gold absent from text (parser pulled gold from scratch by luck).
    E: contradictory finals / truncated (no parseable final answer).

    Gold presence is judged on the FINAL region only. Gold appearing in
    intermediate computation does not count (that was the prior mis-categorization).
    """
    raw = run["raw_output"]
    gold = run["gold_answer"]
    ev = run.get("eval_type", "symbolic")
    extracted = extract_domain_answer(raw, "math", ev)
    graded = bool(is_correct(extracted, gold, ev)) if extracted else False
    gin_anywhere = gold_in_text(raw, gold)
    gfin = gold_in_final(raw, gold)
    contra = has_contradiction(raw, gold)

    if graded and gfin:
        return "A"
    if graded and not gin_anywhere:
        return "D"
    if extracted is None:
        # No parseable answer at all (token-limit truncation or no final marker).
        return "E"
    if not graded and gfin:
        return "C"  # gold IS the final answer but parser missed it → real bug
    if contra:
        return "E"
    # graded incorrect and gold not in final region → model's final answer is wrong
    return "B"


def audit_extraction(input_path: str) -> None:
    runs = []
    with Path(input_path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("variant") == "recovery" and r.get("domain") == "math":
                runs.append(r)

    counts = {k: 0 for k in "ABCDE"}
    by_cat: dict[str, list] = {k: [] for k in "ABCDE"}
    for run in runs:
        cat = classify_run(run)
        counts[cat] += 1
        if cat in {"C", "D", "E"}:
            ex = extract_domain_answer(run["raw_output"], "math", run.get("eval_type"))
            by_cat[cat].append(
                (run["task_id"], run["error_type"], ex, run["gold_answer"])
            )

    total = len(runs)
    print(f"Math recovery runs: {total}\n")
    for key in "ABCDE":
        pct = (100 * counts[key] / total) if total else 0.0
        print(f"  {key}: {counts[key]} ({pct:.1f}%)")
    for key in "CDE":
        print(f"\n{key} ({counts[key]}):")
        for row in by_cat[key][:15]:
            print(f"  {row}")


def audit_baseline_runaway(input_path: str, max_steps: int) -> None:
    """
    Read-only check of baseline_cots.jsonl: flag math baselines that were chosen
    from a runaway/loop trace, and report degenerate samples per task.

    Uses the stored `steps` (recomputed via detect_degenerate_trace) plus the
    `sample_degenerate*` / `baseline_degenerate_fallback` fields written by the
    runaway guard in generate_baselines.py.
    """
    rows = []
    with Path(input_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("domain") == "math":
                rows.append(r)

    print(f"Math baseline rows: {len(rows)} (runaway threshold: step_count>{max_steps})\n")
    print(
        f"{'task_id':45} {'seed':>4} {'steps':>5} {'degen':>5} "
        f"{'reason':18} {'fallback':>8} sample_step_counts"
    )
    flagged = 0
    fallback_count = 0
    for r in rows:
        steps = r.get("steps", [])
        chosen_degen, chosen_reason = detect_degenerate_trace(steps, max_steps=max_steps)
        fallback = bool(r.get("baseline_degenerate_fallback"))
        # Recompute per-sample degeneracy from stored step counts as a cross-check
        # (older files may lack sample_degenerate; recompute from step counts).
        sample_counts = r.get("sample_step_counts", [])
        stored_degen = r.get("sample_degenerate")
        if stored_degen is None:
            n_degen = sum(1 for c in sample_counts if isinstance(c, int) and c > max_steps)
        else:
            n_degen = sum(1 for d in stored_degen if d)
        if chosen_degen or fallback or n_degen:
            flagged += 1
        if fallback:
            fallback_count += 1
        if chosen_degen or fallback or n_degen:
            print(
                f"{str(r.get('task_id'))[:45]:45} {str(r.get('seed')):>4} "
                f"{len(steps):>5} {('YES' if chosen_degen else 'no'):>5} "
                f"{str(chosen_reason or '-'):18} {('YES' if fallback else 'no'):>8} "
                f"{sample_counts}"
            )
    print(
        f"\nFlagged tasks: {flagged}/{len(rows)} | "
        f"chosen baseline degenerate fallbacks: {fallback_count}"
    )
    if flagged == 0:
        print("OK: no runaway/loop markers in any math baseline.")

    # Math sanity-gate report: tasks where the judge said Correct but det disagreed
    # and the extracted answer was off by an order of magnitude (judge=Correct flipped
    # to incorrect before the vote). Surfaces the conflict cases for manual review.
    print("\n--- judge/det conflict + sanity-gate flips (math) ---")
    conflict_tasks = 0
    flip_tasks = 0
    for r in rows:
        conflicts = r.get("sample_judge_det_conflict") or []
        flips = r.get("sample_sanity_gate_flipped") or []
        n_conflict = sum(1 for c in conflicts if c)
        n_flip = sum(1 for fl in flips if fl)
        if n_conflict or n_flip:
            conflict_tasks += 1 if n_conflict else 0
            flip_tasks += 1 if n_flip else 0
            print(
                f"{str(r.get('task_id'))[:45]:45} "
                f"conflicts={n_conflict} flips={n_flip} "
                f"solve_rate={r.get('solve_rate')} verified={r.get('verified')}"
            )
    print(
        f"\nTasks with judge/det conflict: {conflict_tasks} | "
        f"tasks with sanity-gate flips: {flip_tasks}"
    )
    if conflict_tasks == 0:
        print(
            "Note: no conflict fields found (older baseline_cots.jsonl predates the "
            "sanity gate; rerun generate_baselines.py to populate them)."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["extraction", "baseline-runaway"],
        default="extraction",
        help="extraction: math recovery extraction audit (post_injection_runs); "
        "baseline-runaway: flag runaway/loop baselines (baseline_cots)",
    )
    parser.add_argument("--input", default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Runaway step_count threshold for baseline-runaway mode",
    )
    args = parser.parse_args()

    if args.mode == "baseline-runaway":
        audit_baseline_runaway(
            args.input or "data/baseline_cots.jsonl", args.max_steps
        )
    else:
        audit_extraction(args.input or "data/post_injection_runs.jsonl")


if __name__ == "__main__":
    main()
