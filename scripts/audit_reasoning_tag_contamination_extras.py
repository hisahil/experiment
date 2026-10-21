#!/usr/bin/env python3
"""Follow-up checks: distributions and empty recovery segments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONTAM_TASKS = {
    "math500_test_algebra_1837.json",
    "math500_test_algebra_668.json",
    "math500_test_algebra_791.json",
    "math500_test_algebra_858.json",
    "math500_test_counting_and_probability_1009.json",
    "math500_test_counting_and_probability_894.json",
    "math500_test_counting_and_probability_913.json",
    "math500_test_geometry_483.json",
    "math500_test_geometry_967.json",
    "math500_test_intermediate_algebra_431.json",
    "math500_test_intermediate_algebra_894.json",
    "math500_test_number_theory_1201.json",
    "math500_test_number_theory_45.json",
    "math500_test_number_theory_572.json",
    "math500_test_prealgebra_1840.json",
    "math500_test_prealgebra_192.json",
    "math500_test_prealgebra_307.json",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


p = ROOT / "data" / "1.5b" / "complete_dialogues.jsonl"

# Distribution of tag-only model segments by error_type / k / condition
tag_only_err = Counter()
tag_only_k = Counter()
tag_only_cond = Counter()
tag_only_ins_step = Counter()

# Distribution of model-segments containing tag by error_type / k
seg_close_err = Counter()
seg_close_k = Counter()
seg_open_err = Counter()

# Dialogues with </reasoning> in accumulated_trace (subset of 918)
dlg_close_trace_err = Counter()
dlg_close_trace_k = Counter()

# 51 empty recovery segments: on clean or contaminated tasks?
empty_recovery_task_kind = Counter()  # 'contaminated' vs 'clean'
empty_recovery_err = Counter()
empty_recovery_k = Counter()

# Model segments containing tags (not tag-only) — how many?
model_seg_containing_close = 0
model_seg_containing_open = 0
model_seg_containing_any_tag = 0
model_seg_tag_only = 0

for rec in load_jsonl(p):
    tid = rec.get("task_id", "?")
    err = rec.get("error_type")
    k = rec.get("k")
    cond = rec.get("condition")

    trace = rec.get("accumulated_trace") or ""
    if "</reasoning>" in trace.lower():
        dlg_close_trace_err[err] += 1
        dlg_close_trace_k[k] += 1

    for seg in rec.get("trace_segments") or []:
        text = seg.get("text") or ""
        stype = seg.get("type")
        low = text.lower()
        stripped_low = text.strip().lower()

        if stype == "model":
            has_close = "</reasoning>" in low
            has_open = "<reasoning>" in low
            if has_close:
                model_seg_containing_close += 1
                seg_close_err[err] += 1
                seg_close_k[k] += 1
            if has_open:
                model_seg_containing_open += 1
                seg_open_err[err] += 1
            if has_close or has_open:
                model_seg_containing_any_tag += 1
            if stripped_low == "</reasoning>":
                model_seg_tag_only += 1
                tag_only_err[err] += 1
                tag_only_k[k] += 1
                tag_only_cond[cond] += 1
                tag_only_ins_step[rec.get("insertion_step")] += 1

        elif stype == "recovery":
            if not text.strip():
                empty_recovery_task_kind["contaminated" if tid in CONTAM_TASKS else "clean"] += 1
                empty_recovery_err[err] += 1
                empty_recovery_k[k] += 1


print("=" * 78)
print("Model-segment-level counts (all 5400 dialogues):")
print("=" * 78)
print(f"  model segments containing </reasoning>: {model_seg_containing_close}")
print(f"  model segments containing <reasoning>:  {model_seg_containing_open}")
print(f"  model segments containing either tag:   {model_seg_containing_any_tag}")
print(f"  model segments == '</reasoning>' only:  {model_seg_tag_only}")

print()
print("Tag-only intermediate/model segment breakdown (n=141):")
print(f"  by error_type: {dict(tag_only_err)}")
print(f"  by k:          {dict(tag_only_k)}")
print(f"  by condition:  {dict(tag_only_cond)}")
print(f"  by insertion_step: {dict(tag_only_ins_step)}")

print()
print("Distribution of dialogues with </reasoning> in accumulated_trace (n=249):")
print(f"  by error_type: {dict(dlg_close_trace_err)}")
print(f"  by k:          {dict(dlg_close_trace_k)}")

print()
print("Model segments containing </reasoning> broken down:")
print(f"  by error_type: {dict(seg_close_err)}")
print(f"  by k:          {dict(seg_close_k)}")

print()
print("Empty (whitespace-only) recovery segments (n=51):")
print(f"  task kind: {dict(empty_recovery_task_kind)}")
print(f"  by error_type: {dict(empty_recovery_err)}")
print(f"  by k:          {dict(empty_recovery_k)}")
