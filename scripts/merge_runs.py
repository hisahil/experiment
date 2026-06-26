#!/usr/bin/env python3
"""
Per-key merge + dedup of run JSONL files (post_injection_runs.jsonl).

evaluate.py does NOT deduplicate (load_runs only appends), so duplicate rows would
be double-counted in means. This merges two or more run files into one, REPLACING by
the 8-field run key (task_id|model|error_type|k|length_level|seed|variant|condition).
Later inputs (and later rows within a file) win, so a rejudged file passed after the
base overrides the stale rows. The output is asserted to contain each key exactly once.

Usage:
  # base first, rejudged/new last (last wins on conflicts):
  python scripts/merge_runs.py data/post_injection_runs.jsonl data/rejudged.jsonl \
    --out data/post_injection_runs_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_key(rec: dict) -> str:
    """8-field key matching run_adaptive_dialogue.load_existing_keys()."""
    cond = rec.get("condition", "adversarial")
    et = rec.get("error_type") or "neutral"
    return (
        f"{rec.get('task_id')}|{rec.get('model')}|{et}|{rec.get('k')}|"
        f"{rec.get('length_level')}|{rec.get('seed')}|{rec.get('variant')}|{cond}"
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge(inputs: list[Path]) -> tuple[dict[str, dict], dict[str, int]]:
    """Return (key -> winning record, per-input row counts). Later inputs win."""
    merged: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for path in inputs:
        rows = load_jsonl(path)
        counts[str(path)] = len(rows)
        for rec in rows:
            merged[run_key(rec)] = rec  # later occurrence overwrites earlier
    return merged, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-key merge + dedup of run JSONL files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL files (later wins on key conflict).")
    parser.add_argument("--out", type=Path, required=True, help="Output merged JSONL path.")
    args = parser.parse_args()

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        for p in missing:
            print(f"[FAIL] input not found: {p}", file=sys.stderr)
        sys.exit(1)

    merged, counts = merge(args.inputs)

    total_in = sum(counts.values())
    print("Merge inputs (later wins):")
    for path, n in counts.items():
        print(f"  {path}: {n} rows")
    print(f"Total input rows: {total_in}")
    print(f"Unique keys (output rows): {len(merged)}")
    print(f"Duplicate rows collapsed: {total_in - len(merged)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written_keys: Counter = Counter()
    with args.out.open("w", encoding="utf-8") as f:
        for key, rec in merged.items():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written_keys[key] += 1

    # Verification: every key appears exactly once in the output.
    dupes = [k for k, c in written_keys.items() if c != 1]
    assert not dupes, f"output has duplicate keys: {dupes[:5]}"
    print(f"[OK] wrote {len(merged)} rows to {args.out} (each key exactly once).")


if __name__ == "__main__":
    main()
