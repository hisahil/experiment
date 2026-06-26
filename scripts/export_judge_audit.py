#!/usr/bin/env python3
"""
Export a math judge audit CSV from the backup judge_raw logs (read-only).

Reads backup_gpu_final_20260618/data/judge_raw/{baseline,dialogue}/math500_*.jsonl
and writes results/math_judge_audit.csv, one row per judge call, with a short
parser/judge diagnosis. This documents the math finding from the plan: the math
judge is near-perfect (1/950 wrong) and almost all judge/det disagreements are
parser/det-too-strict, not judge errors.

The backup logs predate the judge_parsed -> judge_verdict rename, so this reader
accepts both field names. It never writes to the backup data.

Usage:
  python scripts/export_judge_audit.py
  python scripts/export_judge_audit.py --backup-dir backup_gpu_final_20260618 \
    --out results/math_judge_audit.csv --no-dialogue
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COLUMNS = [
    "task_id",
    "phase",
    "variant",
    "seed",
    "judge_verdict",
    "det_correct",
    "parsed_status",
    "fallback_status",
    "final_answer",
    "gold",
    "short_diagnosis",
]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _judge_correct(rec: dict) -> bool | None:
    """Resolve the judge verdict as a bool from whichever field is present."""
    out = str(rec.get("judge_output") or "").strip().lower()
    if out == "correct":
        return True
    if out == "incorrect":
        return False
    for key in ("judge_verdict", "judge_parsed"):
        if rec.get(key) is not None:
            return bool(rec[key])
    if rec.get("verifier_score") is not None:
        try:
            return bool(int(rec["verifier_score"]))
        except (TypeError, ValueError):
            return None
    return None


def _parsed_status(extracted: object) -> str:
    return "parsed" if extracted is not None and str(extracted).strip() else "no_parse"


def _rel_err(extracted: object, gold: object) -> float | None:
    """Relative error of two numeric answers, or None when non-numeric / gold==0."""
    try:
        g = abs(float(gold))
        if g == 0:
            return None
        return abs(float(extracted) - float(gold)) / g
    except (TypeError, ValueError):
        return None


def diagnose(rec: dict) -> str:
    """Short read-only diagnosis from verdict/det/parse fields (no raw_output here)."""
    fallback = bool(rec.get("judge_fallback"))
    det = bool(rec.get("det_correct"))
    judge = _judge_correct(rec)
    extracted = rec.get("extracted_answer")
    parsed = _parsed_status(extracted) == "parsed"

    if fallback or judge is None:
        return "fallback: det-only, no judge verdict"
    if judge == det:
        return "agree"
    if judge and not det:
        if not parsed:
            return "parser_failure: no answer parsed (det too strict / boxed-in-think)"
        rel = _rel_err(extracted, rec.get("gold_answer"))
        if rel is not None and rel > 0.5:
            # Large numeric gap: either a real judge error (counting_159 class) or a
            # mis-parsed token while gold is in the text. raw_output is not in the
            # judge logs, so this is flagged for inspection (L3 gate handles it).
            return "magnitude_gap: judge=correct det=wrong, large numeric gap (inspect raw_output)"
        return "parser_format: judge=correct det=wrong (prose/tuple/formatting)"
    # judge wrong, det correct
    return "judge_fn: judge=wrong det=correct (rare; inspect)"


def collect_files(backup_dir: Path, include_dialogue: bool) -> list[tuple[str, Path]]:
    phases = ["baseline"] + (["dialogue"] if include_dialogue else [])
    files: list[tuple[str, Path]] = []
    for phase in phases:
        phase_dir = backup_dir / "data" / "judge_raw" / phase
        if not phase_dir.exists():
            continue
        for path in sorted(phase_dir.glob("math500_*.jsonl")):
            files.append((phase, path))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Export math judge audit CSV (read-only over backups).")
    parser.add_argument("--backup-dir", type=Path, default=ROOT / "backup_gpu_final_20260618")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "math_judge_audit.csv")
    parser.add_argument(
        "--no-dialogue",
        dest="include_dialogue",
        action="store_false",
        help="Only export baseline math rows (default includes dialogue too).",
    )
    args = parser.parse_args()

    files = collect_files(args.backup_dir, args.include_dialogue)
    if not files:
        sys.exit(f"[FAIL] no math500_*.jsonl judge logs under {args.backup_dir}")

    out_rows: list[dict] = []
    n_disagree = 0
    n_magnitude_gap = 0
    for phase, path in files:
        for rec in load_jsonl(path):
            judge = _judge_correct(rec)
            det = bool(rec.get("det_correct"))
            diag = diagnose(rec)
            if judge is not None and not rec.get("judge_fallback") and judge != det:
                n_disagree += 1
            if diag.startswith("magnitude_gap") or diag.startswith("judge_fn"):
                n_magnitude_gap += 1
            out_rows.append(
                {
                    "task_id": rec.get("task_id"),
                    "phase": rec.get("phase", phase),
                    "variant": rec.get("variant", ""),
                    "seed": rec.get("seed"),
                    "judge_verdict": rec.get("judge_output")
                    or ("Correct" if judge else "Incorrect" if judge is not None else ""),
                    "det_correct": det,
                    "parsed_status": _parsed_status(rec.get("extracted_answer")),
                    "fallback_status": bool(rec.get("judge_fallback")),
                    "final_answer": rec.get("extracted_answer"),
                    "gold": rec.get("gold_answer"),
                    "short_diagnosis": diag,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[OK] wrote {len(out_rows)} rows from {len(files)} file(s) to {args.out}")
    print(f"     judge/det disagreements (real verdicts): {n_disagree}")
    print(f"     magnitude-gap rows to inspect (judge=correct, det=wrong, large gap): {n_magnitude_gap}")


if __name__ == "__main__":
    main()
