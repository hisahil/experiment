#!/usr/bin/env python3
"""One-off analysis to prove whether empty/double-separator artefacts are
confined to the mini_run_3seed pre-patch dialogues (10 tasks) or if they also
leak into the post-patch full_math dialogues.

Run:
  python scripts/analyze_pre_post_patch_split.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data/complete_dialogues.jsonl")

MINI_TASK_IDS = {
    "math500_test_algebra_114.json",
    "math500_test_algebra_1934.json",
    "math500_test_algebra_2257.json",
    "math500_test_counting_and_probability_14.json",
    "math500_test_counting_and_probability_1114.json",
    "math500_test_intermediate_algebra_1111.json",
    "math500_test_number_theory_239.json",
    "math500_test_number_theory_820.json",
    "math500_test_number_theory_1055.json",
    "math500_test_prealgebra_2019.json",
}
assert len(MINI_TASK_IDS) == 10, f"expected 10 mini task_ids, got {len(MINI_TASK_IDS)}"


def dialogue_group(task_id: str) -> str:
    return "mini" if task_id in MINI_TASK_IDS else "full"


def has_empty_intermediate_segment(row: dict) -> tuple[bool, int]:
    """Return (any_empty, count_empty) using both explicit flag AND heuristic."""
    n_empty = 0
    for seg in row.get("trace_segments", []) or []:
        if seg.get("type") != "model":
            continue
        # explicit post-patch flag
        if seg.get("intermediate_empty") is True:
            n_empty += 1
            continue
        # heuristic (works for pre-patch rows that lack the flag): the segment
        # is intermediate (i.e. NOT the final stop/recovery — those are type
        # "stop"/"recovery"; intermediate model segments are type "model").
        text = (seg.get("text") or "").strip()
        tokens = seg.get("tokens")
        if not text and (tokens is None or tokens <= 1):
            n_empty += 1
    return (n_empty > 0), n_empty


def has_double_separator(row: dict) -> bool:
    acc = row.get("accumulated_trace") or ""
    return "\n\n\n\n" in acc


def has_intermediate_retried(row: dict) -> bool:
    """Post-patch marker. Written on any model trace_segment that was retried."""
    for seg in row.get("trace_segments", []) or []:
        if seg.get("intermediate_retried") is True:
            return True
    for turn in row.get("turns", []) or []:
        if turn.get("intermediate_retried") is True:
            return True
    return False


def main() -> None:
    total = 0
    mini_rows = 0
    full_rows = 0
    empty_mini = 0
    empty_full = 0
    empty_seg_count_mini = 0
    empty_seg_count_full = 0
    doublesep_mini = 0
    doublesep_full = 0
    retried_mini = 0
    retried_full = 0
    k_dist = Counter()
    mini_task_hits = Counter()
    full_empty_examples: list[dict] = []
    full_doublesep_examples: list[dict] = []
    mini_empty_examples: list[dict] = []
    mini_doublesep_examples: list[dict] = []
    seed_dist_mini = Counter()
    seed_dist_full = Counter()

    with DATA.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            tid = row.get("task_id") or ""
            group = dialogue_group(tid)
            k = row.get("k")
            seed = row.get("seed")
            k_dist[(group, k)] += 1

            has_empty, n_empty = has_empty_intermediate_segment(row)
            has_dbl = has_double_separator(row)
            has_retry = has_intermediate_retried(row)

            if group == "mini":
                mini_rows += 1
                mini_task_hits[tid] += 1
                seed_dist_mini[seed] += 1
                if has_empty:
                    empty_mini += 1
                    empty_seg_count_mini += n_empty
                    if len(mini_empty_examples) < 3:
                        mini_empty_examples.append({
                            "dialogue_id": row.get("dialogue_id"),
                            "task_id": tid, "k": k, "seed": seed,
                            "length_level": row.get("length_level"),
                            "variant_row": row.get("condition"),
                            "n_empty_segments": n_empty,
                        })
                if has_dbl:
                    doublesep_mini += 1
                    if len(mini_doublesep_examples) < 3:
                        mini_doublesep_examples.append({
                            "dialogue_id": row.get("dialogue_id"),
                            "task_id": tid, "k": k, "seed": seed,
                        })
                if has_retry:
                    retried_mini += 1
            else:
                full_rows += 1
                seed_dist_full[seed] += 1
                if has_empty:
                    empty_full += 1
                    empty_seg_count_full += n_empty
                    if len(full_empty_examples) < 5:
                        full_empty_examples.append({
                            "dialogue_id": row.get("dialogue_id"),
                            "task_id": tid, "k": k, "seed": seed,
                            "length_level": row.get("length_level"),
                            "variant_row": row.get("condition"),
                            "n_empty_segments": n_empty,
                        })
                if has_dbl:
                    doublesep_full += 1
                    if len(full_doublesep_examples) < 5:
                        full_doublesep_examples.append({
                            "dialogue_id": row.get("dialogue_id"),
                            "task_id": tid, "k": k, "seed": seed,
                        })
                if has_retry:
                    retried_full += 1

    print("=" * 72)
    print("SPLIT ANALYSIS: data/complete_dialogues.jsonl")
    print("=" * 72)
    print(f"Total rows:                    {total}")
    print(f"Mini-run rows (10 task_ids):   {mini_rows}")
    print(f"Full-math rows (rest):         {full_rows}")
    print()
    print(f"Mini task_id coverage (should be 10):  {len(mini_task_hits)}")
    for tid, n in sorted(mini_task_hits.items()):
        print(f"  {tid}: {n} rows")
    print()
    print("Rows-per-(group, k):")
    for key, n in sorted(k_dist.items()):
        print(f"  {key}: {n}")
    print()
    print(f"Seeds present in mini rows:  {dict(sorted(seed_dist_mini.items()))}")
    print(f"Seeds present in full rows:  {dict(sorted(seed_dist_full.items()))}")
    print()
    print("-" * 72)
    print("EMPTY INTERMEDIATE SEGMENTS (intermediate_empty=True OR text.strip()=''"
          " with tokens<=1)")
    print("-" * 72)
    print(f"  rows w/ empty seg    mini: {empty_mini:>4}   full: {empty_full:>4}"
          f"   total: {empty_mini + empty_full}")
    print(f"  empty-segment count  mini: {empty_seg_count_mini:>4}"
          f"   full: {empty_seg_count_full:>4}"
          f"   total: {empty_seg_count_mini + empty_seg_count_full}")
    print()
    print("-" * 72)
    print(r"DOUBLE SEPARATOR in accumulated_trace  ('\n\n\n\n')")
    print("-" * 72)
    print(f"  rows w/ '\\n\\n\\n\\n'    mini: {doublesep_mini:>4}   full: {doublesep_full:>4}"
          f"   total: {doublesep_mini + doublesep_full}")
    print()
    print("-" * 72)
    print("intermediate_retried=True MARKER (only exists after patch deployment)")
    print("-" * 72)
    print(f"  rows w/ retried flag  mini: {retried_mini:>4}   full: {retried_full:>4}")
    print()
    print("-" * 72)
    print("EXAMPLES (up to 5) of empty-segment rows OUTSIDE mini set:")
    print("-" * 72)
    for ex in full_empty_examples:
        print(f"  {ex}")
    print()
    print("-" * 72)
    print("EXAMPLES (up to 5) of double-separator rows OUTSIDE mini set:")
    print("-" * 72)
    for ex in full_doublesep_examples:
        print(f"  {ex}")
    print()
    if mini_empty_examples:
        print("-" * 72)
        print("EXAMPLES of empty-segment rows INSIDE mini set:")
        print("-" * 72)
        for ex in mini_empty_examples:
            print(f"  {ex}")
    if mini_doublesep_examples:
        print("-" * 72)
        print("EXAMPLES of double-separator rows INSIDE mini set:")
        print("-" * 72)
        for ex in mini_doublesep_examples:
            print(f"  {ex}")


if __name__ == "__main__":
    main()
