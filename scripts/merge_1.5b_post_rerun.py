#!/usr/bin/env python3
"""Merge kept rejudged rows with fresh GPU rows after 1.5B tagfix rerun."""

from __future__ import annotations

import json
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


def run_key(rec: dict) -> str:
    cond = rec.get("condition", "adversarial")
    et = rec.get("error_type") or "neutral"
    return (
        f"{rec.get('task_id')}|{rec.get('model')}|{et}|{rec.get('k')}|"
        f"{rec.get('length_level')}|{rec.get('seed')}|{rec.get('variant')}|{cond}"
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    data_dir = ROOT / "data" / "1.5b"
    kept_rejudged = data_dir / "post_injection_runs.rejudged.jsonl"
    runs = data_dir / "post_injection_runs.jsonl"
    out = data_dir / "post_injection_runs.rejudged.jsonl"

    if not kept_rejudged.exists() or not runs.exists():
        sys.exit(f"[FAIL] missing {kept_rejudged} or {runs}")

    kept_by_key = {run_key(r): r for r in load_jsonl(kept_rejudged)}
    run_rows = load_jsonl(runs)
    merged: list[dict] = []
    reused = fresh = 0
    for row in run_rows:
        key = run_key(row)
        tid = row.get("task_id", "")
        if tid not in CONTAMINATED_TASKS and key in kept_by_key:
            merged.append(kept_by_key[key])
            reused += 1
        else:
            merged.append(row)
            fresh += 1

    with out.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] merged {len(merged)} rows -> {out}")
    print(f"     reused_kept_rejudged={reused} fresh_from_runs={fresh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
