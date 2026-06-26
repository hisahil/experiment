#!/usr/bin/env python3
"""
Script 3.5: Post-hoc LLM-as-a-Judge over post_injection_runs.jsonl.

The GPU run (Script 3, run_perturbed.py / run_adaptive_dialogue.py) is now fully
decoupled from the judge (L0): it writes raw_output, extracted_answer and
det_correct only, with no Anthropic dependency. This script adds the LLM verdict
afterwards, so the GPU run can never stall or silently fall back to det on a
credit-out.

What it does
------------
  * reads post_injection_runs.jsonl line by line,
  * deduplicates by the 8-field run key (later row wins, matching the run scripts),
  * for every row without a non-empty judge_raw_output, re-extracts the answer and
    calls grade_sample() with a working judge,
  * writes back correct / verifier_score / judge_output / judge_raw_output /
    judge_verdict / det_correct / judge_fallback,
  * writes the full, deduplicated set back to --out (default: the input file,
    in-place via an atomic temp-file replace).

Robustness (vs rejudge_fallbacks.py, which hard-aborts on credit-out)
---------------------------------------------------------------------
This script NEVER hard-aborts. On an API / credit / billing error it warns LOUDLY
(red + BEL on stderr, plus an appended line in results/judge_alerts.log) and
CONTINUES; the offending row keeps judge_fallback=True and is retried on the next
run (idempotent: rows with a verdict have a non-empty judge_raw_output and are
skipped). The operator decides whether to top up credits and re-run.

Idempotent: a successfully judged row has judge_raw_output != "" and is skipped on
re-run; the 8-field key dedup prevents duplicate rows.

tmux / screen visibility
------------------------
stderr is line-flushed and progress is reported every 50 rows. Launch with tee so
warnings survive a detach:

    python scripts/judge_runs.py 2>&1 | tee -a results/judge_alerts.log

Optionally tail the alert log in a second pane:

    tail -f results/judge_alerts.log | grep --line-buffered -iE 'warn|error|fallback'

DO NOT run against the real Anthropic API casually — it costs money. Use --dry-run
to exercise the join/overwrite logic without any API calls.

Usage:
  python scripts/judge_runs.py --dry-run
  python scripts/judge_runs.py --input data/post_injection_runs.jsonl --limit 100
  python scripts/judge_runs.py 2>&1 | tee -a results/judge_alerts.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer_judge import JudgeCreditError, grade_sample  # noqa: E402
from evaluator import extract_domain_answer  # noqa: E402

ALERT_LOG = ROOT / "results" / "judge_alerts.log"
PROGRESS_EVERY = 50

# ANSI red for loud, tmux-visible warnings (BEL `\a` blinks the tab with
# `set -g visual-bell on`).
_RED = "\033[91m"
_RESET = "\033[0m"


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def run_key(rec: dict) -> str:
    """8-field key matching run_adaptive_dialogue.load_existing_keys()."""
    cond = rec.get("condition", "adversarial")
    et = rec.get("error_type") or "neutral"
    return (
        f"{rec.get('task_id')}|{rec.get('model')}|{et}|{rec.get('k')}|"
        f"{rec.get('length_level')}|{rec.get('seed')}|{rec.get('variant')}|{cond}"
    )


def index_tasks(tasks: list[dict]) -> dict[str, dict]:
    return {t["task_id"]: t for t in tasks if t.get("task_id")}


def task_question(task: dict | None) -> str:
    if not task:
        return ""
    return str(task.get("prompt", task.get("question", "")))


def needs_judging(row: dict) -> bool:
    """A row needs a verdict when it has no non-empty judge_raw_output yet."""
    return not str(row.get("judge_raw_output") or "").strip()


def loud_warn(message: str) -> None:
    """Red + BEL warning to flushed stderr, also appended to the alert log."""
    print(f"\a{_RED}[JUDGE WARN] {message}{_RESET}", file=sys.stderr, flush=True)
    try:
        ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ALERT_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [JUDGE WARN] {message}\n")
    except OSError:
        pass


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def judge_row(
    row: dict,
    task: dict | None,
    judge_cfg: dict,
    *,
    math_tolerance: float,
    science_tolerance: float,
    dry_run: bool,
) -> str:
    """Judge one row in place.

    Returns a status string: "ok" (real verdict), "fallback" (transient/parse
    failure, no verdict) or "credit_out" (non-transient credit/billing/auth).
    On credit_out the row is left with judge_fallback=True and no verdict so a
    later re-run retries it.
    """
    domain = row.get("domain", "math")
    eval_type = row.get("eval_type")
    raw_output = str(row.get("raw_output", "") or "")
    gold = str(row.get("gold_answer", "") or "")
    question = task_question(task) or str(row.get("question", ""))

    # Re-extract (helps recovery extraction_failed cases) but keep the stored
    # extracted_answer when re-extraction finds nothing.
    extracted = extract_domain_answer(raw_output, domain, eval_type)
    if extracted is None:
        extracted = row.get("extracted_answer")

    try:
        grading = grade_sample(
            question=question,
            gold_answer=gold,
            answer_content=raw_output,
            extracted_answer=extracted,
            domain=domain,
            eval_type=eval_type,
            judge_cfg=judge_cfg,
            math_tolerance=math_tolerance,
            science_tolerance=science_tolerance,
            dry_run=dry_run,
        )
    except JudgeCreditError as exc:
        row["judge_fallback"] = True
        row["judge_error"] = str(exc)
        row["extracted_answer"] = extracted
        return "credit_out"

    row["correct"] = grading["correct"]
    row["verifier_score"] = grading["verifier_score"]
    row["judge_output"] = grading.get("judge_output", "")
    row["judge_raw_output"] = grading.get("judge_raw_output", "")
    row["judge_verdict"] = grading.get("correct")
    row["judge_fallback"] = grading.get("judge_fallback")
    if "det_correct" in grading:
        row["det_correct"] = grading["det_correct"]
    row["extracted_answer"] = extracted
    row["judged"] = True

    # In dry-run there is no API, so grade_sample always reports a fallback; that is
    # expected and is not a real fallback to warn about.
    if dry_run:
        return "ok"
    return "fallback" if grading.get("judge_fallback") else "ok"


def write_rows(rows: list[dict], out_path: Path) -> None:
    """Atomically write rows to out_path (temp file + os.replace)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc judge for post_injection_runs.jsonl (Script 3.5).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--input", type=Path, default=ROOT / "data/post_injection_runs.jsonl")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/experiment_tasks.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Output path (default: in-place over --input).")
    parser.add_argument("--limit", type=int, default=None, help="Judge at most N rows (debug).")
    parser.add_argument("--dry-run", action="store_true", help="No API calls (deterministic grading only).")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"[FAIL] input not found: {args.input}")

    cfg = load_yaml(args.config)
    judge_cfg = dict(cfg.get("judge", {}))
    judge_cfg["enabled"] = True
    script4_cfg = cfg.get("script4", {})
    math_tolerance = float(script4_cfg.get("math_tolerance", 1e-4))
    science_tolerance = float(script4_cfg.get("science_tolerance", 1e-2))

    out_path = args.out or args.input

    rows = load_jsonl(args.input)
    # Deduplicate by 8-field key (later wins) so we never emit duplicate rows.
    by_key: dict[str, dict] = {}
    for rec in rows:
        by_key[run_key(rec)] = rec
    deduped = list(by_key.values())

    tasks_by_id = index_tasks(load_jsonl(args.tasks))

    pending = [r for r in deduped if needs_judging(r)]
    if args.limit is not None:
        pending = pending[: args.limit]
    total = len(pending)

    print(f"Input rows: {len(rows)} | unique keys: {len(deduped)}", file=sys.stderr, flush=True)
    print(f"Rows needing a verdict: {total}", file=sys.stderr, flush=True)
    print(f"Tasks loaded for join: {len(tasks_by_id)} ({args.tasks})", file=sys.stderr, flush=True)
    print(f"Mode: {'DRY-RUN (no API)' if args.dry_run else 'LIVE judge API'}", file=sys.stderr, flush=True)

    ok = 0
    fallback = 0
    credit_out = 0
    missing_question = 0
    start = time.time()

    for i, row in enumerate(pending, start=1):
        task = tasks_by_id.get(row.get("task_id"))
        if not task_question(task) and not row.get("question"):
            missing_question += 1
        status = judge_row(
            row,
            task,
            judge_cfg,
            math_tolerance=math_tolerance,
            science_tolerance=science_tolerance,
            dry_run=args.dry_run,
        )
        if status == "ok":
            ok += 1
        elif status == "credit_out":
            credit_out += 1
            fallback += 1
            loud_warn(
                f"credit/billing/auth failure on row {i}/{total} "
                f"(task_id={row.get('task_id')}, seed={row.get('seed')}); "
                f"left judge_fallback=True — top up credits and re-run (idempotent)."
            )
        else:  # fallback
            fallback += 1
            loud_warn(
                f"no verdict (transient/parse) on row {i}/{total} "
                f"(task_id={row.get('task_id')}, seed={row.get('seed')}); left judge_fallback=True."
            )

        if i % PROGRESS_EVERY == 0 or i == total:
            elapsed = fmt_elapsed(time.time() - start)
            print(
                f"[{i}/{total}] verdicts ok={ok} fallback={fallback} elapsed={elapsed}",
                file=sys.stderr,
                flush=True,
            )

    write_rows(deduped, out_path)

    no_verdict = fallback
    print("", file=sys.stderr, flush=True)
    print(
        f"Done. Judged {total} row(s): ok={ok} fallback={fallback} (credit_out={credit_out}).",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"{no_verdict} row(s) without judge verdict, of which {credit_out} due to credit-out.",
        file=sys.stderr,
        flush=True,
    )
    if missing_question:
        print(f"[WARN] {missing_question} row(s) had no question from the tasks join.", file=sys.stderr, flush=True)
    print(f"[OK] wrote {len(deduped)} rows to {out_path}", file=sys.stderr, flush=True)
    if args.dry_run:
        print("Dry-run: grading used the deterministic cross-check, not the judge API.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
