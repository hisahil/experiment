#!/usr/bin/env python3
"""Remove 1.5B rows for baseline-tag-contaminated tasks (pre-rerun staging)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONTAMINATED_TASKS = frozenset({
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
})


def filter_jsonl(src: Path, dst: Path, *, keep_task) -> tuple[int, int]:
    kept = removed = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_id", "")
            if keep_task(tid):
                fout.write(line + "\n")
                kept += 1
            else:
                removed += 1
    return kept, removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "1.5b",
        help="Directory with complete_dialogues + post_injection runs.",
    )
    parser.add_argument(
        "--suffix",
        default=".pre_tagfix",
        help="Backup suffix before filtering (default: .pre_tagfix).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = [
        "complete_dialogues.jsonl",
        "post_injection_runs.jsonl",
        "post_injection_runs.rejudged.jsonl",
    ]

    def keep(tid: str) -> bool:
        return tid not in CONTAMINATED_TASKS

    for name in files:
        src = args.data_dir / name
        if not src.exists():
            print(f"[SKIP] {src} not found", file=sys.stderr)
            continue
        backup = src.with_name(src.name + args.suffix)
        if not args.dry_run:
            if not backup.exists():
                shutil.copy2(src, backup)
                print(f"[BACKUP] {backup}")
            else:
                print(f"[BACKUP] {backup} already exists, skipping copy")
            kept, removed = filter_jsonl(src, src.with_suffix(".tmp"), keep_task=keep)
            src.with_suffix(".tmp").replace(src)
        else:
            kept = removed = 0
            with src.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    tid = json.loads(line).get("task_id", "")
                    if keep(tid):
                        kept += 1
                    else:
                        removed += 1
        print(f"[{name}] kept={kept} removed={removed}")

    print(f"Contaminated tasks: {len(CONTAMINATED_TASKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
