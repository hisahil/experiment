#!/usr/bin/env python3
"""Smoke check for the Punkt 1 fixes (leere Zwischen-Segmente).

Reads a `complete_dialogues.jsonl` and reports:
  1. Empty-intermediate rate                (target: ~0 %; pilot before fix: 4.6 %)
  2. Retried-intermediate rate              (Schicht 2 activity; expect > 0 only
                                             for the rare residual cases)
  3. Double-separator occurrences in traces (Schicht 1 must never create `\\n\\n\\n\\n`)
  4. Token-length distribution of intermediate segments (sanity)
  5. Optional pass/fail against thresholds so it can be used as a gate.

Usage (analysis / gate after a fresh run):

    python scripts/smoke_intermediate_segments.py \\
        --dialogues data/complete_dialogues.jsonl \\
        --max-empty-rate 0.005      # 0.5 % ceiling for the full run
        --max-double-seps 0

The pre-fix pilot backup is expected to FAIL the gate — that's the reference.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def iter_dialogues(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def analyse(dialogues_path: Path) -> dict:
    seg_count = 0
    empty = 0
    retried = 0
    tokens: list[int] = []
    double_sep_dialogues = 0
    empty_by_k: Counter[int] = Counter()
    empty_by_err: Counter[str] = Counter()

    for row in iter_dialogues(dialogues_path):
        trace = row.get("accumulated_trace", "") or ""
        if "\n\n\n\n" in trace:
            double_sep_dialogues += 1
        for seg in row.get("trace_segments", []):
            if seg.get("type") != "model":
                continue
            seg_count += 1
            tokens.append(int(seg.get("tokens", 0) or 0))
            if seg.get("intermediate_retried"):
                retried += 1
            text = seg.get("text", "") or ""
            is_empty = (
                bool(seg.get("intermediate_empty"))
                or text.strip() == ""
                or int(seg.get("tokens", 0) or 0) <= 1
            )
            if is_empty:
                empty += 1
                empty_by_k[int(row.get("k", -1) or -1)] += 1
                empty_by_err[str(row.get("error_type"))] += 1

    if seg_count == 0:
        return {
            "dialogues_path": str(dialogues_path),
            "intermediate_segments": 0,
            "empty": 0,
            "empty_rate": 0.0,
            "retried": 0,
            "double_separator_dialogues": double_sep_dialogues,
            "tokens": {},
            "empty_by_k": {},
            "empty_by_error_type": {},
        }

    return {
        "dialogues_path": str(dialogues_path),
        "intermediate_segments": seg_count,
        "empty": empty,
        "empty_rate": empty / seg_count,
        "retried": retried,
        "retried_rate": retried / seg_count,
        "double_separator_dialogues": double_sep_dialogues,
        "tokens": {
            "min": min(tokens),
            "median": int(statistics.median(tokens)),
            "mean": round(statistics.mean(tokens), 1),
            "max": max(tokens),
        },
        "empty_by_k": dict(empty_by_k),
        "empty_by_error_type": dict(empty_by_err),
    }


def print_report(report: dict) -> None:
    print(f"[smoke] dialogues file: {report['dialogues_path']}")
    n = report["intermediate_segments"]
    if n == 0:
        print("[smoke] no intermediate segments (nothing to check).")
        return
    print(f"[smoke] intermediate segments: {n}")
    print(
        f"[smoke] empty:   {report['empty']} "
        f"({report['empty_rate'] * 100:.2f} %) — "
        f"target ~0, pilot baseline 4.6 %"
    )
    print(
        f"[smoke] retried: {report.get('retried', 0)} "
        f"({report.get('retried_rate', 0.0) * 100:.2f} %)"
    )
    print(
        f"[smoke] traces with '\\n\\n\\n\\n' anywhere: "
        f"{report['double_separator_dialogues']}"
    )
    t = report["tokens"]
    print(
        f"[smoke] intermediate tokens: min={t['min']}, median={t['median']}, "
        f"mean={t['mean']}, max={t['max']}"
    )
    if report["empty_by_k"]:
        print(f"[smoke] empty by k: {report['empty_by_k']}")
    if report["empty_by_error_type"]:
        print(f"[smoke] empty by error_type: {report['empty_by_error_type']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dialogues", type=Path, required=True,
        help="Path to complete_dialogues.jsonl to analyse.",
    )
    parser.add_argument(
        "--max-empty-rate", type=float, default=None,
        help="If set, exit 1 when empty intermediate rate exceeds this fraction.",
    )
    parser.add_argument(
        "--max-double-seps", type=int, default=None,
        help="If set, exit 1 when more than this many dialogues contain '\\n\\n\\n\\n'.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the report as JSON on stdout instead of a human summary.",
    )
    args = parser.parse_args()

    if not args.dialogues.exists():
        print(f"error: dialogues file not found: {args.dialogues}", file=sys.stderr)
        return 2

    report = analyse(args.dialogues)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print_report(report)

    failed = False
    if args.max_empty_rate is not None and report["intermediate_segments"] > 0:
        if report["empty_rate"] > args.max_empty_rate:
            print(
                f"[FAIL] empty rate {report['empty_rate'] * 100:.2f} % "
                f"exceeds threshold {args.max_empty_rate * 100:.2f} %",
                file=sys.stderr,
            )
            failed = True
    if args.max_double_seps is not None:
        if report["double_separator_dialogues"] > args.max_double_seps:
            print(
                f"[FAIL] {report['double_separator_dialogues']} traces contain "
                f"'\\n\\n\\n\\n' (threshold {args.max_double_seps})",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
