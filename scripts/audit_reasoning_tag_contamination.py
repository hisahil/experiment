#!/usr/bin/env python3
"""Audit reasoning-tag contamination in the 1.5B run.

Verifies claims about pseudo `<reasoning>` / `</reasoning>` tags that were
allegedly inherited from 1.5B baselines into dialogues. Does not modify data.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TAG_PATTERNS = {
    "<reasoning>": re.compile(r"<reasoning>", re.IGNORECASE),
    "</reasoning>": re.compile(r"</reasoning>", re.IGNORECASE),
    "<thinking>": re.compile(r"<thinking>", re.IGNORECASE),
    "</thinking>": re.compile(r"</thinking>", re.IGNORECASE),
    "<thought>": re.compile(r"<thought>", re.IGNORECASE),
    "</thought>": re.compile(r"</thought>", re.IGNORECASE),
    "<think>": re.compile(r"<think>"),
    "</think>": re.compile(r"</think>"),
}

STRING_FIELDS_TO_SCAN = ("baseline_cot", "full_text", "dialogue", "continuation_trace")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def scan_text(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, rx in TAG_PATTERNS.items():
        c = len(rx.findall(text))
        if c:
            counts[name] = c
    return counts


def scan_record_text(rec) -> dict[str, int]:
    """Scan every string field of a record (recursing into lists/dicts)."""
    aggregate: dict[str, int] = {}

    def walk(v):
        if isinstance(v, str):
            for name, rx in TAG_PATTERNS.items():
                c = len(rx.findall(v))
                if c:
                    aggregate[name] = aggregate.get(name, 0) + c
        elif isinstance(v, list):
            for item in v:
                walk(item)
        elif isinstance(v, dict):
            for item in v.values():
                walk(item)

    walk(rec)
    return aggregate


# ----------------- SECTION A: BASELINE CONTAMINATION -----------------
def audit_baselines():
    print("=" * 78)
    print("A) BASELINE CONTAMINATION")
    print("=" * 78)
    all_contam_task_ids: dict[str, set[str]] = {}
    for model in ("1.5b", "7b", "14b", "32b"):
        for fname in ("baseline_cots.jsonl", "baseline_cots_continuation.jsonl"):
            p = ROOT / "data" / "baselines" / model / fname
            if not p.exists():
                continue
            tag_counter: Counter[str] = Counter()
            contam_ids: dict[str, dict[str, int]] = {}
            total = 0
            for rec in load_jsonl(p):
                total += 1
                hits = scan_record_text(rec)
                if hits:
                    contam_ids[rec.get("task_id", "?")] = hits
                    for k, v in hits.items():
                        tag_counter[k] += v
            print(f"[{model:5s}] {fname:38s} rows={total:4d}  contaminated_tasks={len(contam_ids):3d}  tag_counts={dict(tag_counter)}")
            if fname == "baseline_cots.jsonl":
                all_contam_task_ids[model] = set(contam_ids.keys())
            elif model == "1.5b":
                all_contam_task_ids[f"{model}_cont"] = set(contam_ids.keys())
    print()
    print("Contaminated task_ids per model (baseline_cots.jsonl):")
    for model in ("1.5b", "7b", "14b", "32b"):
        ids = sorted(all_contam_task_ids.get(model, set()))
        print(f"  {model}: {len(ids)} tasks")
        for tid in ids:
            print(f"    - {tid}")
    if "1.5b_cont" in all_contam_task_ids:
        print()
        print(f"1.5b baseline_cots_continuation.jsonl contaminated tasks: {len(all_contam_task_ids['1.5b_cont'])}")
        symmetric_diff = all_contam_task_ids["1.5b"] ^ all_contam_task_ids["1.5b_cont"]
        if symmetric_diff:
            print(f"  DIFF vs baseline_cots.jsonl: {sorted(symmetric_diff)}")
        else:
            print("  Same task_ids as baseline_cots.jsonl (convert script did NOT strip tags).")
    return all_contam_task_ids.get("1.5b", set())


# ----------------- SECTION B: DIALOGUE CONTAMINATION -----------------
def audit_dialogues(contam_task_ids_1_5b: set[str]):
    print()
    print("=" * 78)
    print("B) DIALOGUE CONTAMINATION")
    print("=" * 78)
    p = ROOT / "data" / "1.5b" / "complete_dialogues.jsonl"

    total = 0
    trace_with_open = 0
    trace_with_close = 0
    dialogues_close_anywhere = 0
    tag_only_intermediate = 0
    tag_only_recovery = 0
    tag_only_stop = 0
    empty_intermediate = 0
    empty_recovery = 0
    model_seg_with_tag = 0

    per_task_stats: dict[str, dict[str, int]] = defaultdict(lambda: {
        "n_dialogues": 0,
        "trace_open": 0,
        "trace_close": 0,
        "dialogue_close_any": 0,
        "tag_only_intermediate": 0,
        "tag_only_recovery": 0,
        "model_seg_with_tag": 0,
    })
    per_error_close = Counter()
    per_k_close = Counter()

    for rec in load_jsonl(p):
        total += 1
        tid = rec.get("task_id", "?")
        per_task_stats[tid]["n_dialogues"] += 1

        trace = rec.get("accumulated_trace") or ""
        has_open = "<reasoning>" in trace.lower()
        has_close = "</reasoning>" in trace.lower()
        if has_open:
            trace_with_open += 1
            per_task_stats[tid]["trace_open"] += 1
        if has_close:
            trace_with_close += 1
            per_task_stats[tid]["trace_close"] += 1

        dialogue_close_flag = False
        tag_only_flag_intermediate = False
        for seg in rec.get("trace_segments") or []:
            text = seg.get("text") or ""
            stype = seg.get("type")
            low = text.lower()
            stripped = text.strip()
            stripped_low = stripped.lower()

            if "</reasoning>" in low or "<reasoning>" in low:
                dialogue_close_flag = True
                if stype == "model":
                    model_seg_with_tag += 1
                    per_task_stats[tid]["model_seg_with_tag"] += 1

            if stype == "model":
                if stripped_low == "</reasoning>" or stripped_low == "<reasoning>":
                    tag_only_intermediate += 1
                    tag_only_flag_intermediate = True
                    per_task_stats[tid]["tag_only_intermediate"] += 1
                if not stripped:
                    empty_intermediate += 1
            elif stype == "recovery":
                if stripped_low == "</reasoning>" or stripped_low == "<reasoning>":
                    tag_only_recovery += 1
                    per_task_stats[tid]["tag_only_recovery"] += 1
                if not stripped:
                    empty_recovery += 1
            elif stype == "stop":
                if stripped_low == "</reasoning>" or stripped_low == "<reasoning>":
                    tag_only_stop += 1

        if dialogue_close_flag:
            dialogues_close_anywhere += 1
            per_task_stats[tid]["dialogue_close_any"] += 1
            per_error_close[rec.get("error_type", "?")] += 1
            per_k_close[rec.get("k")] += 1

    print(f"Total dialogues scanned: {total}")
    print(f"Dialogues with <reasoning> in accumulated_trace: {trace_with_open}")
    print(f"Dialogues with </reasoning> in accumulated_trace: {trace_with_close}")
    print(f"Dialogues with </reasoning> ANY segment: {dialogues_close_anywhere}")
    print(f"'Model' segments containing <reasoning> or </reasoning>: {model_seg_with_tag}")
    print(f"Model (intermediate) segments == '</reasoning>' only: {tag_only_intermediate}")
    print(f"Recovery segments == '</reasoning>' only: {tag_only_recovery}")
    print(f"Stop segments == '</reasoning>' only: {tag_only_stop}")
    print(f"Empty (whitespace) model/intermediate segments: {empty_intermediate}")
    print(f"Empty (whitespace) recovery segments: {empty_recovery}")

    contam_tasks_in_dialogues = [tid for tid, s in per_task_stats.items() if s["dialogue_close_any"] > 0]
    contam_tasks_trace_open = [tid for tid, s in per_task_stats.items() if s["trace_open"] > 0]
    print()
    print(f"Distinct tasks with <reasoning> in accumulated_trace: {len(contam_tasks_trace_open)}")
    print(f"Distinct tasks with </reasoning> anywhere in dialogue: {len(contam_tasks_in_dialogues)}")
    print()
    print("Overlap with baseline-contaminated task_ids:")
    match = set(contam_tasks_in_dialogues) & contam_task_ids_1_5b
    only_in_baseline = contam_task_ids_1_5b - set(contam_tasks_in_dialogues)
    only_in_dialogues = set(contam_tasks_in_dialogues) - contam_task_ids_1_5b
    print(f"  intersect: {len(match)}")
    print(f"  in baseline only (no dialogue tag): {len(only_in_baseline)} -> {sorted(only_in_baseline)}")
    print(f"  in dialogues only (unexpected): {len(only_in_dialogues)} -> {sorted(only_in_dialogues)}")

    print()
    print("Per-task counts (only contaminated tasks):")
    for tid in sorted(contam_tasks_trace_open):
        s = per_task_stats[tid]
        print(f"  {tid:45s} nd={s['n_dialogues']:3d} trace_open={s['trace_open']:3d} trace_close={s['trace_close']:3d} tag_only_inter={s['tag_only_intermediate']:3d} tag_only_rec={s['tag_only_recovery']:3d} model_seg_tag={s['model_seg_with_tag']:3d}")

    print()
    print("D) DISTRIBUTION of dialogues with </reasoning> anywhere:")
    print(f"  by error_type: {dict(per_error_close)}")
    print(f"  by k: {dict(per_k_close)}")

    total_trace_open = trace_with_open
    print()
    print(f"Sanity 17*54 = 918 => observed trace_open = {total_trace_open}")

    return {
        "total": total,
        "trace_open": trace_with_open,
        "trace_close": trace_with_close,
        "dialogues_close_anywhere": dialogues_close_anywhere,
        "tag_only_intermediate": tag_only_intermediate,
        "tag_only_recovery": tag_only_recovery,
        "tag_only_stop": tag_only_stop,
        "empty_intermediate": empty_intermediate,
        "empty_recovery": empty_recovery,
        "contam_tasks": contam_tasks_trace_open,
    }


# ----------------- SECTION C: CONVERT SCRIPT -----------------
def audit_convert_script():
    print()
    print("=" * 78)
    print("C) CONVERT SCRIPT")
    print("=" * 78)
    p = ROOT / "scripts" / "convert_baselines_to_continuation.py"
    text = p.read_text()
    for i, line in enumerate(text.splitlines(), 1):
        if "re.compile" in line or "STEP_PREFIX_RE" in line or "strip_step_prefix" in line:
            print(f"  L{i}: {line}")


if __name__ == "__main__":
    contam = audit_baselines()
    audit_dialogues(contam)
    audit_convert_script()
