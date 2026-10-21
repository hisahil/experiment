#!/usr/bin/env python3
"""Backfill shift_subtype and off_topic_theme onto legacy dialogue/run files.

The mini-run pre-dating Punkt 3 wrote complete_dialogues.jsonl and
post_injection_runs.jsonl without the two injection sub-label fields, but
injections_generated.jsonl already carries them. This script joins the three
files by the injection cell key (task_id | error_type | k | length_level) and
emits enriched copies alongside the originals so the pilot backup can be
re-evaluated without regenerating anything.

Usage:
    python scripts/backfill_injection_fields.py \\
        --injections data/injections_generated.jsonl \\
        --dialogues  data/complete_dialogues.jsonl \\
        --runs       data/post_injection_runs.jsonl \\
        --out-dir    data/backfilled

Behaviour:
- Reads all three inputs (any of --dialogues/--runs may be omitted).
- Builds a {cell_key: (shift_subtype, off_topic_theme)} map from injections.
- For each dialogue/run row, injects the two fields only when the row's
  error_type + k + length_level + task_id match an injection cell. Rows that
  already have the fields are passed through unchanged.
- Writes <out-dir>/<original-basename>.enriched.jsonl next to the inputs;
  originals are never touched.
- Prints a coverage summary (rows enriched vs. rows without a matching cell).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _cell_key(
    task_id: str,
    error_type: str | None,
    k: int | str | None,
    length_level: str | None,
) -> str:
    return f"{task_id}|{error_type}|{k}|{length_level}"


def build_label_index(injections: Iterable[dict]) -> dict[str, dict[str, str | None]]:
    """cell_key -> {shift_subtype, off_topic_theme}. Missing fields → None."""
    idx: dict[str, dict[str, str | None]] = {}
    for rec in injections:
        key = _cell_key(
            rec.get("task_id", ""),
            rec.get("error_type"),
            rec.get("k"),
            rec.get("length_level"),
        )
        idx[key] = {
            "shift_subtype": rec.get("shift_subtype"),
            "off_topic_theme": rec.get("off_topic_theme"),
        }
    return idx


def enrich_row(row: dict, label_idx: dict[str, dict[str, str | None]]) -> tuple[dict, bool]:
    """Return (possibly-enriched row, whether a matching cell was found)."""
    key = _cell_key(
        row.get("task_id", ""),
        row.get("error_type"),
        row.get("k"),
        row.get("length_level"),
    )
    labels = label_idx.get(key)
    if labels is None:
        return row, False
    out = dict(row)
    # Only set when a value exists and the target row doesn't already have one.
    for field in ("shift_subtype", "off_topic_theme"):
        val = labels.get(field)
        if val is None:
            continue
        if out.get(field) in (None, ""):
            out[field] = val
    return out, True


def process_file(
    path: Path,
    label_idx: dict[str, dict[str, str | None]],
    out_dir: Path,
    tag: str,
) -> None:
    if not path.exists():
        print(f"[{tag}] SKIP — file does not exist: {path}")
        return
    out_path = out_dir / f"{path.stem}.enriched.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    matched = 0
    field_added: Counter[str] = Counter()
    with out_path.open("w", encoding="utf-8") as out_fp:
        for row in _iter_jsonl(path):
            total += 1
            before = {f: row.get(f) for f in ("shift_subtype", "off_topic_theme")}
            enriched, hit = enrich_row(row, label_idx)
            if hit:
                matched += 1
                for f in ("shift_subtype", "off_topic_theme"):
                    if before.get(f) in (None, "") and enriched.get(f) not in (None, ""):
                        field_added[f] += 1
            out_fp.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    added = ", ".join(f"{k}+{v}" for k, v in sorted(field_added.items())) or "none"
    print(
        f"[{tag}] {path.name}: {total} rows, {matched} matched a cell "
        f"(fields added: {added}) -> {out_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--injections", type=Path, required=True)
    parser.add_argument("--dialogues", type=Path, default=None)
    parser.add_argument("--runs", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.injections.exists():
        print(f"error: injections file not found: {args.injections}", file=sys.stderr)
        return 2

    label_idx = build_label_index(_iter_jsonl(args.injections))
    print(
        f"[index] loaded {len(label_idx)} injection cells from {args.injections}"
    )
    if not label_idx:
        print("error: no injection records found — aborting.", file=sys.stderr)
        return 2

    if args.dialogues is not None:
        process_file(args.dialogues, label_idx, args.out_dir, "dialogues")
    if args.runs is not None:
        process_file(args.runs, label_idx, args.out_dir, "runs")
    if args.dialogues is None and args.runs is None:
        print("warning: no --dialogues or --runs given; nothing to enrich.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
