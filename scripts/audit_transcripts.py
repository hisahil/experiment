#!/usr/bin/env python3
"""
Guard G4: transcript-consistency audit.

Checks that every dialogue_id present in the run rows (post_injection_runs.jsonl)
also has a transcript in complete_dialogues.jsonl and adaptive_dialogues.jsonl,
and reports orphan transcripts (transcript ids with no run rows). There is no
transcript dedup in the pipeline, so historical partial crashes could leave gaps;
this is read-only and never modifies data.

Usage:
  python scripts/audit_transcripts.py \
    --runs data/post_injection_runs.jsonl \
    --complete data/complete_dialogues.jsonl \
    --adaptive data/adaptive_dialogues.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dialogue_ids(path: Path) -> tuple[set[str], int]:
    """Return (unique dialogue_ids, total rows) for a JSONL file."""
    ids: set[str] = set()
    rows = 0
    if not path.exists():
        return ids, rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = rec.get("dialogue_id")
            if did:
                ids.add(str(did))
    return ids, rows


def _print_sample(label: str, ids: set[str], limit: int = 20) -> None:
    print(f"  {label}: {len(ids)}")
    for did in sorted(ids)[:limit]:
        print(f"    - {did}")
    if len(ids) > limit:
        print(f"    ... and {len(ids) - limit} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dialogue_id consistency across artifacts.")
    parser.add_argument("--runs", type=Path, default=ROOT / "data/post_injection_runs.jsonl")
    parser.add_argument("--complete", type=Path, default=ROOT / "data/complete_dialogues.jsonl")
    parser.add_argument("--adaptive", type=Path, default=ROOT / "data/adaptive_dialogues.jsonl")
    args = parser.parse_args()

    runs_ids, runs_rows = load_dialogue_ids(args.runs)
    complete_ids, complete_rows = load_dialogue_ids(args.complete)
    adaptive_ids, adaptive_rows = load_dialogue_ids(args.adaptive)

    print("Transcript consistency audit")
    print("=" * 60)
    print(f"runs     ({args.runs.name}): {runs_rows} rows, {len(runs_ids)} dialogue_ids")
    print(f"complete ({args.complete.name}): {complete_rows} rows, {len(complete_ids)} dialogue_ids")
    print(f"adaptive ({args.adaptive.name}): {adaptive_rows} rows, {len(adaptive_ids)} dialogue_ids")
    print("")

    problems = False

    missing_complete = runs_ids - complete_ids
    missing_adaptive = runs_ids - adaptive_ids
    if missing_complete:
        problems = True
        print("[WARN] run dialogue_ids with NO complete_dialogues transcript:")
        _print_sample("missing in complete", missing_complete)
    if missing_adaptive:
        problems = True
        print("[WARN] run dialogue_ids with NO adaptive_dialogues transcript:")
        _print_sample("missing in adaptive", missing_adaptive)

    orphan_complete = complete_ids - runs_ids
    orphan_adaptive = adaptive_ids - runs_ids
    if orphan_complete:
        problems = True
        print("[WARN] complete_dialogues transcripts with NO run rows (orphans):")
        _print_sample("orphan in complete", orphan_complete)
    if orphan_adaptive:
        problems = True
        print("[WARN] adaptive_dialogues transcripts with NO run rows (orphans):")
        _print_sample("orphan in adaptive", orphan_adaptive)

    if not problems:
        print("OK — every run dialogue_id has complete + adaptive transcripts, no orphans.")
        return
    # Read-only audit: report only, do not fail CI hard. raw_output in the runs
    # file remains the canonical source; document any gaps before dataset release.
    print("\nAudit found inconsistencies (see above). This is informational only.")
    sys.exit(0)


if __name__ == "__main__":
    main()
