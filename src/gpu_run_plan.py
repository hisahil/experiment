"""Helpers for unified GPU full runs (selected tasks, timings, domain limits)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baseline_index import index_verified_baselines
from experiment_scope import get_scope, select_tasks_balanced
from grid_validation import load_jsonl
from run_timings import FULL_THESIS_DIALOGUES, HOURLY_RATE_USD, build_extrapolation

FULL_THESIS_TASKS = 240

# Script 1 candidate pool per domain (schlank).
BASELINE_CANDIDATE_LIMITS: dict[str, int] = {
    "math": 10,
    "science": 15,
    "logic": 24,
}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def resolve_selected_tasks(
    *,
    cfg: dict,
    scope_name: str,
    tasks_path: Path,
    baseline_path: Path,
    model_name: str,
    balanced_per_domain: int,
    experiment_seed: int,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Return (selected tasks, counts) after Script 1."""
    scope = get_scope(cfg, scope_name)
    all_tasks = load_jsonl(tasks_path)
    task_by_id = {t["task_id"]: t for t in all_tasks}
    baseline_index = index_verified_baselines(load_jsonl(baseline_path), model_name)
    eligible = [task_by_id[tid] for tid in sorted(baseline_index.keys()) if tid in task_by_id]
    selected = select_tasks_balanced(eligible, scope, experiment_seed, balanced_per_domain)
    from collections import Counter

    return selected, {
        "verified_by_domain": dict(Counter(t.get("domain", "") for t in eligible)),
        "selected_by_domain": dict(Counter(t.get("domain", "") for t in selected)),
    }


def build_total_extrapolation(
    *,
    script1_elapsed_s: float,
    script1_tasks_processed: int,
    script2_elapsed_s: float,
    script2_tasks_processed: int,
    script3_completed_dialogues: int,
    script3_compute_elapsed_s: float,
    script3_planned_dialogues: int,
    script4_elapsed_s: float,
    model_load_s: float,
    pilot_tasks: int,
    scope: str | None = None,
) -> dict[str, Any]:
    """Extrapolate Script 1–4 wall times to the full thesis run.

    Task-count scaling for Scripts 1/2/4 uses FULL_THESIS_TASKS (240). The
    dialogue total reported as planned_full_thesis_dialogues uses
    FULL_THESIS_DIALOGUES (5_400; see src/run_timings.py and docs/full_run_plan.md).
    """
    script3 = build_extrapolation(
        completed_dialogues=script3_completed_dialogues,
        compute_elapsed_s=script3_compute_elapsed_s,
        planned_tonight=script3_planned_dialogues,
        scope=scope,
    )

    def _scale(elapsed: float, n: int, full_n: int) -> float | None:
        if elapsed <= 0 or n <= 0:
            return None
        return elapsed / n * full_n

    ext1_s = _scale(script1_elapsed_s, script1_tasks_processed, FULL_THESIS_TASKS)
    ext2_s = _scale(script2_elapsed_s, script2_tasks_processed, FULL_THESIS_TASKS)
    ext4_s = _scale(script4_elapsed_s, pilot_tasks, FULL_THESIS_TASKS)
    ext3_s = script3.get("extrapolated_full_thesis_s")

    parts = [ext1_s, ext2_s, ext3_s, ext4_s]
    if model_load_s > 0:
        parts.append(model_load_s)

    if ext3_s is None or ext1_s is None or ext2_s is None:
        total_s = None
    else:
        total_s = sum(p for p in parts if p is not None)

    total_h = total_s / 3600.0 if total_s is not None else None

    return {
        **script3,
        "full_thesis_tasks": FULL_THESIS_TASKS,
        "script1_elapsed_s": round(script1_elapsed_s, 1),
        "script1_tasks_processed": script1_tasks_processed,
        "seconds_per_baseline_task": round(script1_elapsed_s / script1_tasks_processed, 2)
        if script1_tasks_processed > 0 and script1_elapsed_s > 0
        else None,
        "extrapolated_script1_h": round(ext1_s / 3600.0, 2) if ext1_s else None,
        "script2_elapsed_s": round(script2_elapsed_s, 1),
        "script2_tasks_processed": script2_tasks_processed,
        "seconds_per_injection_task": round(script2_elapsed_s / script2_tasks_processed, 2)
        if script2_tasks_processed > 0 and script2_elapsed_s > 0
        else None,
        "extrapolated_script2_h": round(ext2_s / 3600.0, 2) if ext2_s else None,
        "script4_elapsed_s": round(script4_elapsed_s, 1),
        "extrapolated_script4_h": round(ext4_s / 3600.0, 2) if ext4_s else None,
        "model_load_s": round(model_load_s, 1),
        "extrapolated_full_thesis_total_s": round(total_s, 1) if total_s is not None else None,
        "extrapolated_full_thesis_total_h": round(total_h, 2) if total_h is not None else None,
        "extrapolated_full_thesis_total_days": round(total_s / 86400.0, 2) if total_s is not None else None,
        "extrapolated_full_thesis_total_cost_usd": round(total_h * HOURLY_RATE_USD, 2)
        if total_h is not None
        else None,
        "planned_full_thesis_dialogues": FULL_THESIS_DIALOGUES,
    }
