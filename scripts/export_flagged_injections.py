#!/usr/bin/env python3
"""Export rows with validation_issues from injections_generated.jsonl.

Adds task-pool context columns so a reviewer can tell a REAL gold leak from the
structural false-positive source (gold is a substring of the decoy) and can see
which decoy values are shared across multiple tasks:

  * gold / decoy               — from the task pool (data/experiment_tasks.jsonl)
  * gold_in_decoy              — gold is a literal substring of decoy (or vice versa)
  * decoy_shared_by_n_tasks    — how many tasks in the pool use this decoy value
  * gold_and_decoy_share_substring / generation_stop_reason — from the record
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _load_tasks(path: Path) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    if not path.exists():
        return tasks
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                tasks[rec.get("task_id")] = rec
    return tasks


def _gold_in_decoy(gold: str, decoy: str) -> bool:
    gold, decoy = gold.strip(), decoy.strip()
    if not gold or not decoy or gold == decoy:
        return False
    return gold in decoy or decoy in gold


def main() -> None:
    parser = argparse.ArgumentParser(description="Export flagged injection rows for manual review.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/injections_generated.jsonl"),
        help="Input JSONL from Script 2.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/experiment_tasks.jsonl"),
        help="Task pool for gold/decoy context columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/flagged_injections.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    tasks = _load_tasks(args.tasks)
    decoy_counts = Counter(
        str(t.get("decoy_answer", "")).strip()
        for t in tasks.values()
        if str(t.get("decoy_answer", "")).strip()
    )

    rows: list[dict] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            issues = rec.get("validation_issues") or []
            if not issues:
                continue
            task = tasks.get(rec.get("task_id"), {})
            gold = str(task.get("gold_answer", "")).strip()
            decoy = str(task.get("decoy_answer", "")).strip()
            rows.append(
                {
                    "task_id": rec.get("task_id"),
                    "target_model": rec.get("target_model"),
                    "error_type": rec.get("error_type"),
                    "k": rec.get("k"),
                    "length_level": rec.get("length_level"),
                    "shift_subtype": rec.get("shift_subtype"),
                    "off_topic_theme": rec.get("off_topic_theme"),
                    "gold": gold,
                    "decoy": decoy,
                    "gold_in_decoy": _gold_in_decoy(gold, decoy),
                    "gold_and_decoy_share_substring": rec.get("gold_and_decoy_share_substring"),
                    "decoy_shared_by_n_tasks": decoy_counts.get(decoy, 0) if decoy else 0,
                    "generation_stop_reason": rec.get("generation_stop_reason"),
                    "validation_issues": "|".join(str(x) for x in issues),
                    "injection_steps": " || ".join(rec.get("injection_steps") or []),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "target_model",
                "error_type",
                "k",
                "length_level",
                "shift_subtype",
                "off_topic_theme",
                "gold",
                "decoy",
                "gold_in_decoy",
                "gold_and_decoy_share_substring",
                "decoy_shared_by_n_tasks",
                "generation_stop_reason",
                "validation_issues",
                "injection_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Flagged rows: {len(rows)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
