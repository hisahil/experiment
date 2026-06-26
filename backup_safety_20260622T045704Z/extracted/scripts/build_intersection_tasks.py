#!/usr/bin/env python3
"""
Build the final stratified-intersection task set (Step 4 of task-selection).

Intersection approach (von Recum et al., ICLR 2026): keep only tasks that ALL
experiment models verify, so recovery effects are not confounded with model
incapacity. Sampling is stratified by MATH-500 difficulty level (default 20 per
level => 100 tasks).

Inputs:
  * candidate pool JSONL (from prepare_tasks.py --domains math, stratified):
    provides task metadata (level, subject, prompt, gold_answer, decoy_answer).
  * one or more baseline_cots.jsonl files (Script 1 output). Each row carries
    task_id, model, verified, solve_rate. Multiple models may live in one file
    (keyed by task_id|model) or be split across files; both are supported.

Logic:
  1. Index candidate pool by task_id (source of level/subject/prompt/gold).
  2. From baseline rows, collect verified task_ids per model + solve_rate.
  3. Intersection = task_ids verified by EVERY model present.
  4. Per level, sample --per-level (default 20) from the intersection (fixed seed).
  5. Backfill warning: if a level has < per_level in the intersection, report it
     (operator runs Step 3d: sample fresh tasks of that level, re-verify).
  6. Write experiment_tasks.jsonl with per-model solve_rate columns.

Usage:
  python scripts/build_intersection_tasks.py \
    --pool data/experiment_tasks_stratified.jsonl \
    --baselines data/baseline_cots.jsonl \
    --out data/experiment_tasks.jsonl
  # or separate per-model files:
  python scripts/build_intersection_tasks.py \
    --pool data/experiment_tasks_stratified.jsonl \
    --baselines data/baseline_cots_7b.jsonl data/baseline_cots_14b.jsonl data/baseline_cots_32b.jsonl \
    --out data/experiment_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short_model_name(model: str) -> str:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -> 7b (best-effort, else slug)."""
    tail = model.rsplit("/", 1)[-1].lower()
    for token in tail.replace("-", " ").split():
        if token.endswith("b") and token[:-1].replace(".", "").isdigit():
            return token
    return tail


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified-intersection task set.")
    parser.add_argument(
        "--pool",
        type=Path,
        default=ROOT / "data/experiment_tasks_stratified.jsonl",
        help="Candidate pool with level/subject metadata.",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Script-1 baseline_cots.jsonl files (rows carry model + verified).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/experiment_tasks.jsonl",
        help="Final task set output.",
    )
    parser.add_argument("--per-level", type=int, default=20, help="Tasks per difficulty level.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Difficulty levels to balance over.",
    )
    args = parser.parse_args()

    if not args.pool.exists():
        sys.exit(f"[FAIL] candidate pool not found: {args.pool}")

    pool = load_jsonl(args.pool)
    pool_by_id = {t["task_id"]: t for t in pool if t.get("task_id")}

    # Collect verified task_ids + solve_rate per model from all baseline files.
    verified_by_model: dict[str, set[str]] = defaultdict(set)
    solve_rate: dict[tuple[str, str], float] = {}  # (model, task_id) -> rate
    for bpath in args.baselines:
        if not bpath.exists():
            sys.exit(f"[FAIL] baseline file not found: {bpath}")
        for row in load_jsonl(bpath):
            model = row.get("model")
            task_id = row.get("task_id")
            if not model or not task_id:
                continue
            if row.get("verified"):
                verified_by_model[model].add(task_id)
            solve_rate[(model, task_id)] = float(row.get("solve_rate", 0.0))

    if not verified_by_model:
        sys.exit("[FAIL] no verified rows found in baseline files.")

    models = sorted(verified_by_model)
    print(f"Models found: {models}")
    for m in models:
        print(f"  {m}: {len(verified_by_model[m])} verified")

    # Intersection = verified by EVERY model present.
    intersection: set[str] = set.intersection(*(verified_by_model[m] for m in models))
    print(f"Intersection (verified by all {len(models)} models): {len(intersection)} tasks")

    # Group intersection by level.
    by_level: dict[int, list[str]] = defaultdict(list)
    for task_id in intersection:
        task = pool_by_id.get(task_id)
        if task is None:
            # task verified but not in candidate pool (e.g. a backfill task) — skip
            continue
        lv = task.get("level")
        if lv in args.levels:
            by_level[lv].append(task_id)

    # Per-level stratified sample (fixed seed per level, deterministic).
    final_ids: list[str] = []
    shortfall: dict[int, int] = {}
    for lv in args.levels:
        candidates = sorted(by_level.get(lv, []))  # sort for reproducibility
        if len(candidates) < args.per_level:
            shortfall[lv] = args.per_level - len(candidates)
            chosen = candidates  # take all available; backfill needed
        else:
            rng = np.random.RandomState(args.seed + lv)
            idx = rng.choice(len(candidates), size=args.per_level, replace=False)
            chosen = [candidates[int(i)] for i in idx]
        final_ids.extend(chosen)
        print(f"  Level {lv}: {len(candidates)} in intersection -> selected {len(chosen)}")

    # Write final tasks with per-model solve_rate columns.
    out_rows: list[dict] = []
    for task_id in final_ids:
        task = dict(pool_by_id[task_id])
        for m in models:
            task[f"solve_rate_{short_model_name(m)}"] = solve_rate.get((m, task_id))
        out_rows.append(task)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_rows)} tasks to {args.out}")
    if shortfall:
        print("\n[WARN] Backfill needed (Step 3d) — levels below per_level in intersection:")
        for lv, missing in sorted(shortfall.items()):
            print(f"  Level {lv}: missing {missing} (sample fresh Level-{lv} tasks, re-verify all models)")
        print("Final set is NOT yet complete; resolve backfill before the full run.")
    else:
        print(f"All {len(args.levels)} levels filled with {args.per_level} tasks each. Complete.")


if __name__ == "__main__":
    main()
