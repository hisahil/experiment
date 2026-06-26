#!/usr/bin/env python3
"""
Re-judge rows that fell back to the deterministic cross-check.

During the pilot the judge API ran out of credit, so ~943 rows have
judge_fallback == True and correct == det_correct. evaluate.py trusts the stored
`correct` and never re-calls the judge, so those labels stay deterministic unless
we re-judge here.

This script:
  * reads post_injection_runs.jsonl,
  * deduplicates by the 8-field run key (later row wins),
  * for every judge_fallback == True row, joins `question` from the tasks file by
    task_id, re-extracts the answer, calls grade_sample() with a working judge, and
  * OVERWRITES all grading fields together (correct, verifier_score, judge_fallback,
    judge_output, judge_raw_output) plus sets rejudged: true,
  * writes the full, deduplicated set to --out (default: <input>.rejudged.jsonl).

It is idempotent: a successfully rejudged row no longer has judge_fallback == True,
so a second run does not touch it; the 8-field key dedup prevents duplicate rows.

DO NOT run against the real Anthropic API casually — it costs money. Use --dry-run
to exercise the join/overwrite logic without any API calls.

Usage:
  python scripts/rejudge_fallbacks.py --dry-run
  python scripts/rejudge_fallbacks.py --config configs/experiment.yaml \
    --input data/post_injection_runs.jsonl --tasks data/experiment_tasks.jsonl \
    --out data/post_injection_runs.rejudged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer_judge import JudgeCreditError, grade_sample  # noqa: E402
from evaluator import extract_domain_answer  # noqa: E402

# Grading fields overwritten together so the agreement report stays consistent.
_GRADING_FIELDS = (
    "correct",
    "verifier_score",
    "judge_output",
    "judge_raw_output",
    "judge_fallback",
)


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


def rejudge_row(
    row: dict,
    task: dict | None,
    judge_cfg: dict,
    *,
    math_tolerance: float,
    science_tolerance: float,
    dry_run: bool,
) -> bool:
    """Re-judge one row in place. Returns True if the `correct` label changed."""
    domain = row.get("domain", "math")
    eval_type = row.get("eval_type")
    raw_output = str(row.get("raw_output", "") or "")
    gold = str(row.get("gold_answer", "") or "")
    question = task_question(task) or str(row.get("question", ""))

    # Re-extract (helps the recovery extraction_failed cases) but keep the stored
    # extracted_answer if re-extraction finds nothing.
    extracted = extract_domain_answer(raw_output, domain, eval_type)
    if extracted is None:
        extracted = row.get("extracted_answer")

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

    old_correct = row.get("correct")
    # Overwrite ALL grading fields together (Gap 0c / G3).
    row["correct"] = grading["correct"]
    row["verifier_score"] = grading["verifier_score"]
    row["judge_output"] = grading.get("judge_output", "")
    row["judge_raw_output"] = grading.get("judge_raw_output", "")
    row["judge_fallback"] = grading.get("judge_fallback")
    if "det_correct" in grading:
        row["det_correct"] = grading["det_correct"]
    row["extracted_answer"] = extracted
    row["rejudged"] = True
    return bool(old_correct) != bool(grading["correct"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-judge judge_fallback rows.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--input", type=Path, default=ROOT / "data/post_injection_runs.jsonl")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/experiment_tasks.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Output path (default <input>.rejudged.jsonl).")
    parser.add_argument("--dry-run", action="store_true", help="No API calls (uses deterministic grading).")
    parser.add_argument("--limit", type=int, default=None, help="Re-judge at most N fallback rows (debug).")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"[FAIL] input not found: {args.input}")

    cfg = load_yaml(args.config)
    judge_cfg = dict(cfg.get("judge", {}))
    judge_cfg["enabled"] = True
    script4_cfg = cfg.get("script4", {})
    math_tolerance = float(script4_cfg.get("math_tolerance", 1e-4))
    science_tolerance = float(script4_cfg.get("science_tolerance", 1e-2))

    out_path = args.out or args.input.with_suffix(".rejudged.jsonl")

    rows = load_jsonl(args.input)
    # Deduplicate by 8-field key (later wins) so we never emit duplicate rows.
    by_key: dict[str, dict] = {}
    for rec in rows:
        by_key[run_key(rec)] = rec
    deduped = list(by_key.values())

    tasks_by_id = index_tasks(load_jsonl(args.tasks))

    fallback_rows = [r for r in deduped if r.get("judge_fallback") is True]
    if args.limit is not None:
        fallback_rows = fallback_rows[: args.limit]

    print(f"Input rows: {len(rows)} | unique keys: {len(deduped)}")
    print(f"judge_fallback rows to re-judge: {len(fallback_rows)}")
    print(f"Tasks loaded for join: {len(tasks_by_id)} ({args.tasks})")
    print(f"Mode: {'DRY-RUN (no API)' if args.dry_run else 'LIVE judge API'}")

    missing_question = 0
    changed = 0
    rejudged = 0
    try:
        for row in fallback_rows:
            task = tasks_by_id.get(row.get("task_id"))
            if not task_question(task) and not row.get("question"):
                missing_question += 1
            if rejudge_row(
                row,
                task,
                judge_cfg,
                math_tolerance=math_tolerance,
                science_tolerance=science_tolerance,
                dry_run=args.dry_run,
            ):
                changed += 1
            rejudged += 1
    except JudgeCreditError as exc:
        print(f"\n[ABORT] Judge credit/billing error: {exc}", file=sys.stderr)
        print("[ABORT] Top up Anthropic credits and re-run (idempotent).", file=sys.stderr)
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nRe-judged: {rejudged} | correct labels changed: {changed}")
    if missing_question:
        print(f"[WARN] {missing_question} fallback row(s) had no question from tasks join.")
    print(f"[OK] wrote {len(deduped)} rows to {out_path}")
    if args.dry_run:
        print("Dry-run: grading used the deterministic cross-check, not the judge API.")


if __name__ == "__main__":
    main()
