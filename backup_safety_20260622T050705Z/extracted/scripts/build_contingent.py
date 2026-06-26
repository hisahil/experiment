#!/usr/bin/env python3
"""Build 14B/32B contingent from 7B-verified tasks (task-selection Phase 3a)."""

from __future__ import annotations

import argparse
import json
import sys
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contingent for 14B/32B runs.")
    parser.add_argument(
        "--pool",
        type=Path,
        default=ROOT / "data/task_selection/pool/candidates.jsonl",
    )
    parser.add_argument(
        "--baselines-7b",
        type=Path,
        default=ROOT / "data/task_selection/baselines/7b/baseline_cots.jsonl",
    )
    parser.add_argument(
        "--out-contingent",
        type=Path,
        default=ROOT / "data/task_selection/meta/contingent.jsonl",
    )
    parser.add_argument(
        "--out-reserve",
        type=Path,
        default=ROOT / "data/task_selection/meta/reserve_7b.jsonl",
    )
    parser.add_argument("--cap-l1l4", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    if not args.pool.exists():
        sys.exit(f"[FAIL] pool not found: {args.pool}")
    if not args.baselines_7b.exists():
        sys.exit(f"[FAIL] 7B baselines not found: {args.baselines_7b}")

    pool_by_id = {t["task_id"]: t for t in load_jsonl(args.pool) if t.get("task_id")}
    verified_7b: set[str] = set()
    for row in load_jsonl(args.baselines_7b):
        if row.get("verified") and row.get("task_id"):
            verified_7b.add(row["task_id"])

    by_level: dict[int, list[str]] = {lv: [] for lv in args.levels}
    missing_level = 0
    for task_id in sorted(verified_7b):
        task = pool_by_id.get(task_id)
        if task is None:
            continue
        lv = task.get("level")
        if lv in by_level:
            by_level[lv].append(task_id)
        else:
            missing_level += 1

    contingent_ids: list[str] = []
    reserve_ids: list[str] = []

    for lv in args.levels:
        ids = sorted(by_level.get(lv, []))
        available = len(ids)
        if lv == 5:
            chosen = ids
            reserved: list[str] = []
        else:
            cap = min(args.cap_l1l4, available)
            if available > cap:
                rng = np.random.RandomState(args.seed + lv)
                idx = sorted(int(i) for i in rng.choice(available, size=cap, replace=False))
                chosen = [ids[i] for i in idx]
                chosen_set = set(chosen)
                reserved = [tid for tid in ids if tid not in chosen_set]
            else:
                chosen = ids
                reserved = []
        contingent_ids.extend(chosen)
        reserve_ids.extend(reserved)
        print(
            f"  Level {lv}: verified={available} "
            f"contingent={len(chosen)} reserve={len(reserved)}"
        )

    contingent_rows = [pool_by_id[tid] for tid in contingent_ids if tid in pool_by_id]
    reserve_rows = [pool_by_id[tid] for tid in reserve_ids if tid in pool_by_id]
    write_jsonl(args.out_contingent, contingent_rows)
    write_jsonl(args.out_reserve, reserve_rows)

    print(f"7B verified: {len(verified_7b)}")
    if missing_level:
        print(f"[WARN] {missing_level} verified tasks missing level in pool")
    print(f"Contingent: {len(contingent_rows)} -> {args.out_contingent}")
    print(f"Reserve:    {len(reserve_rows)} -> {args.out_reserve}")


if __name__ == "__main__":
    main()
