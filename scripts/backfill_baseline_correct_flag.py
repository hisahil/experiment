#!/usr/bin/env python3
"""Backfill top-level `correct` on verified baseline rows for acc_clean=1.0.

Script 1 majority-vote already selected a correct sample for verified=True rows,
but baseline_cots.jsonl omitted the `correct` field. evaluate.py then re-graded
parsed_answer vs gold (Path B), causing format false negatives (acc_clean < 1.0).

For verified=True: set correct=True (matches design: verified => clean acc 1.0).
For verified=False: set correct=False when absent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = [
    ROOT / "data" / "baselines" / "1.5b" / "baseline_cots.jsonl",
    ROOT / "data" / "baselines" / "7b" / "baseline_cots.jsonl",
    ROOT / "data" / "baselines" / "14b" / "baseline_cots.jsonl",
    ROOT / "data" / "baselines" / "32b" / "baseline_cots.jsonl",
]


def backfill(path: Path, *, backup_suffix: str, dry_run: bool) -> dict:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    changed = 0
    verified_set = 0
    for row in rows:
        if row.get("correct") is not None:
            continue
        if row.get("verified"):
            row["correct"] = True
            verified_set += 1
            changed += 1
        else:
            row["correct"] = False
            changed += 1

    if dry_run:
        return {"path": str(path), "changed": changed, "verified_true": verified_set}

    if changed:
        backup = path.with_name(path.name + backup_suffix)
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"  [BACKUP] {backup}")
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"path": str(path), "changed": changed, "verified_true": verified_set}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--suffix", default=".pre_correct_backfill")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in args.paths:
        if not path.exists():
            print(f"[SKIP] {path} not found", file=sys.stderr)
            continue
        stats = backfill(path, backup_suffix=args.suffix, dry_run=args.dry_run)
        print(f"[OK] {stats}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
