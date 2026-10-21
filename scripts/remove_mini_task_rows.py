#!/usr/bin/env python3
"""Remove mini_run_3seed task rows from dialogue JSONL files and optionally sort.

Used when regenerating the 540 pre-patch dialogues inside a full_math run.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

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

ERROR_TYPE_ORDER = {
    "off_topic": 0,
    "paraphrased_shift": 1,
    "false_consensus": 2,
}
VARIANT_ORDER = {"stop": 0, "recovery": 1}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_key(row: dict) -> tuple:
    et = row.get("error_type") or ""
    return (
        row.get("task_id") or "",
        ERROR_TYPE_ORDER.get(et, 99),
        et,
        int(row.get("k") or 0),
        row.get("length_level") or "",
        int(row.get("seed") or 0),
    )


def post_key(row: dict) -> tuple:
    variant = row.get("variant") or ""
    return (*run_key(row), VARIANT_ORDER.get(variant, 99), variant)


def backup_files(data_dir: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "complete_dialogues.jsonl",
        "post_injection_runs.jsonl",
    ):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    timings = Path("results/run_timings.json")
    if timings.exists():
        shutil.copy2(timings, backup_dir / "run_timings.json")


def remove_mini_rows(data_dir: Path) -> dict[str, int]:
    complete_path = data_dir / "complete_dialogues.jsonl"
    post_path = data_dir / "post_injection_runs.jsonl"

    complete_rows = load_jsonl(complete_path)
    post_rows = load_jsonl(post_path)

    complete_keep = [r for r in complete_rows if r.get("task_id") not in MINI_TASK_IDS]
    post_keep = [r for r in post_rows if r.get("task_id") not in MINI_TASK_IDS]

    write_jsonl(complete_path, complete_keep)
    write_jsonl(post_path, post_keep)

    return {
        "complete_before": len(complete_rows),
        "complete_after": len(complete_keep),
        "complete_removed": len(complete_rows) - len(complete_keep),
        "post_before": len(post_rows),
        "post_after": len(post_keep),
        "post_removed": len(post_rows) - len(post_keep),
    }


def sort_rows(data_dir: Path) -> dict[str, int]:
    complete_path = data_dir / "complete_dialogues.jsonl"
    post_path = data_dir / "post_injection_runs.jsonl"

    complete_rows = sorted(load_jsonl(complete_path), key=run_key)
    post_rows = sorted(load_jsonl(post_path), key=post_key)

    write_jsonl(complete_path, complete_rows)
    write_jsonl(post_path, post_rows)

    return {
        "complete_rows": len(complete_rows),
        "post_rows": len(post_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--sort", action="store_true")
    args = parser.parse_args()

    if args.backup_dir:
        backup_files(args.data_dir, args.backup_dir)
        print(f"Backed up to {args.backup_dir}")

    if args.remove:
        stats = remove_mini_rows(args.data_dir)
        print(
            f"Removed mini rows: complete {stats['complete_removed']} "
            f"({stats['complete_before']} -> {stats['complete_after']}), "
            f"post {stats['post_removed']} "
            f"({stats['post_before']} -> {stats['post_after']})"
        )

    if args.sort:
        stats = sort_rows(args.data_dir)
        print(f"Sorted: complete={stats['complete_rows']} post={stats['post_rows']}")


if __name__ == "__main__":
    main()
