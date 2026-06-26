#!/usr/bin/env python3
"""
Judge-coverage retry loop (Phase B1) — drives judge_fallback to 0 before evaluate.

Runs judge_batch.py repeatedly with --only-fallbacks until either no rows carry
judge_fallback==true or --max-passes is reached. This is the fix for the pilot
k=3 confound (94% det-only because judge credits ran out mid-run): the full-run
pipeline calls this between Script 3b and evaluate.py so every run is judged by
the LLM judge, not silently dropped to the deterministic placeholder.

Each pass reuses judge_batch.py (same prompt/parsing), overwriting the input in
place so the next pass only re-judges the remaining fallbacks. A pass with no
fallbacks is a cheap passthrough (no API call). The evaluate.py hard gate
(check_judge_coverage) is the backstop if this still leaves fallbacks.

Usage:
  python scripts/ensure_judge_coverage.py \
    --config configs/experiment.yaml \
    --input data/post_injection_runs.jsonl \
    --tasks data/experiment_tasks.jsonl \
    --max-passes 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def count_fallbacks(path: Path) -> tuple[int, int]:
    """Return (n_fallback, n_total) over the JSONL run file."""
    n_total = 0
    n_fb = 0
    if not path.exists():
        return 0, 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)
            if rec.get("judge_fallback"):
                n_fb += 1
    return n_fb, n_total


def run_pass(*, config: Path, input_path: Path, tasks: Path, only_fallbacks: bool, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        str(SCRIPTS / "judge_batch.py"),
        "--config",
        str(config),
        "--input",
        str(input_path),
        "--tasks",
        str(tasks),
        "--out",
        str(input_path),  # overwrite in place so the next pass sees fewer fallbacks
    ]
    if only_fallbacks:
        cmd.append("--only-fallbacks")
    if dry_run:
        cmd.append("--dry-run")
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge-coverage retry loop (Phase B1).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--input", type=Path, default=ROOT / "data/post_injection_runs.jsonl")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/experiment_tasks.jsonl")
    parser.add_argument("--max-passes", type=int, default=5, help="Max --only-fallbacks retries.")
    parser.add_argument(
        "--initial-full-pass",
        action="store_true",
        help="Judge ALL rows once first (use when Script 3b ran with --no-judge).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"[FAIL] input not found: {args.input}")

    if args.initial_full_pass:
        print("[ensure-judge] initial full judge pass (all rows)")
        run_pass(
            config=args.config,
            input_path=args.input,
            tasks=args.tasks,
            only_fallbacks=False,
            dry_run=args.dry_run,
        )

    for attempt in range(1, args.max_passes + 1):
        n_fb, n_total = count_fallbacks(args.input)
        print(f"[ensure-judge] before pass {attempt}: {n_fb}/{n_total} judge_fallback rows")
        if n_fb == 0:
            print("[ensure-judge] 0 fallbacks — judge coverage complete.")
            return
        if args.dry_run:
            print("[ensure-judge] dry-run: not submitting batches; stopping loop.")
            return
        run_pass(
            config=args.config,
            input_path=args.input,
            tasks=args.tasks,
            only_fallbacks=True,
            dry_run=args.dry_run,
        )

    n_fb, n_total = count_fallbacks(args.input)
    if n_fb > 0:
        print(
            f"[ensure-judge] WARNING: {n_fb}/{n_total} fallbacks remain after "
            f"{args.max_passes} passes. evaluate.py judge-coverage gate will block "
            "the full run; inspect these rows (judge API errors / unparsable verdicts)."
        )
    else:
        print("[ensure-judge] 0 fallbacks — judge coverage complete.")


if __name__ == "__main__":
    main()
