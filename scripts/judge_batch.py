#!/usr/bin/env python3
"""
Stage A — Judge Batch Pass.

Instead of calling the judge inline during Script 3b (one synchronous API call
per row), run Script 3b with the judge DISABLED (deterministic grading only),
then re-judge every row here as a single Anthropic Message Batch (50% token
discount + prompt caching on the judge system prompt).

Pipeline:
  1. Script 3b runs with judge.enabled = false  -> rows have judge_fallback=true
     and correct == det_correct (deterministic placeholder).
  2. judge_batch.py builds one judge request per row, submits ONE batch, polls
     until it ends, then overwrites the grading fields (correct, verifier_score,
     judge_output, judge_raw_output, judge_fallback=false, rejudged=true).

Rows that the batch could not grade (errored or unparseable verdict) keep their
deterministic label and judge_fallback stays true, so a re-run picks them up.

This reuses the exact judge prompt + parsing as the inline judge
(answer_judge.build_judge_user_prompt / parse_judge_response), so labels are
identical to the synchronous path — only timing and cost differ.

Usage:
  # dry-run: build requests, no API calls, report counts
  python scripts/judge_batch.py --dry-run
  # real batch pass
  ANTHROPIC_PROMPT_CACHE=1 python scripts/judge_batch.py \
    --input data/post_injection_runs.jsonl --tasks data/experiment_tasks.jsonl \
    --out data/post_injection_runs.rejudged.jsonl
  # resume polling an already-submitted batch
  python scripts/judge_batch.py --resume-batch msgbatch_xxx \
    --input data/post_injection_runs.jsonl --out data/post_injection_runs.rejudged.jsonl
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

from answer_judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
    parse_judge_response,
    resolve_eval_type,
)
from evaluator import extract_domain_answer  # noqa: E402

POLL_INTERVAL_SECONDS = 30


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


def dedupe(rows: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    for r in rows:
        by_key[run_key(r)] = r
    return list(by_key.values())


def build_request(idx: int, row: dict, task: dict | None, judge_cfg: dict) -> dict:
    """Build one Anthropic batch request for a row. custom_id = r{idx}."""
    domain = row.get("domain", "math")
    eval_type = resolve_eval_type(domain, row.get("eval_type"), str(row.get("gold_answer", "")))
    raw_output = str(row.get("raw_output", "") or "")
    gold = str(row.get("gold_answer", "") or "")
    question = ""
    if task:
        question = str(task.get("prompt", task.get("question", "")))
    if not question:
        question = str(row.get("question", ""))

    extracted = extract_domain_answer(raw_output, domain, eval_type)
    if extracted is None:
        extracted = row.get("extracted_answer")

    prompt = build_judge_user_prompt(
        question=question,
        gold_answer=gold,
        answer_content=raw_output,
        extracted_answer=extracted,
        domain=domain,
        eval_type=eval_type,
    )
    system = str(judge_cfg.get("system_prompt", JUDGE_SYSTEM_PROMPT))
    cache = os.environ.get("ANTHROPIC_PROMPT_CACHE", "0") == "1"
    system_param = (
        [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if cache
        else system
    )
    return {
        "custom_id": f"r{idx}",
        "params": {
            "model": str(judge_cfg.get("model", DEFAULT_JUDGE_MODEL)),
            "max_tokens": int(judge_cfg.get("max_tokens", 200)),
            "temperature": float(judge_cfg.get("temperature", 0)),
            "system": system_param,
            "messages": [{"role": "user", "content": prompt}],
        },
        "_extracted": extracted,  # carried locally; stripped before send
    }


def apply_verdict(row: dict, raw_text: str, extracted) -> bool:
    """Apply a parsed judge verdict to a row. Returns True if graded (not fallback)."""
    parsed = parse_judge_response(raw_text)
    raw_stripped = (raw_text or "").strip()
    row["extracted_answer"] = extracted
    if parsed is None:
        row["judge_output"] = raw_stripped
        row["judge_raw_output"] = raw_stripped
        row["judge_fallback"] = True
        row["rejudged"] = True
        return False
    correct = bool(parsed)
    row["correct"] = correct
    row["verifier_score"] = int(correct)
    row["judge_output"] = "Correct" if correct else "Incorrect"
    row["judge_raw_output"] = raw_stripped
    row["judge_fallback"] = False
    row["rejudged"] = True
    return True


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge Batch Pass (Anthropic Message Batches).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--input", type=Path, default=ROOT / "data/post_injection_runs.jsonl")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/experiment_tasks.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Default <input>.rejudged.jsonl")
    parser.add_argument("--only-fallbacks", action="store_true",
                        help="Only re-judge rows with judge_fallback==true (default: all rows).")
    parser.add_argument("--dry-run", action="store_true", help="Build requests, no API calls.")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (debug).")
    parser.add_argument("--resume-batch", type=str, default=None,
                        help="Skip submission; poll + apply results for an existing batch id.")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"[FAIL] input not found: {args.input}")

    cfg = load_yaml(args.config)
    judge_cfg = dict(cfg.get("judge", {}))
    judge_cfg["enabled"] = True
    out_path = args.out or args.input.with_suffix(".rejudged.jsonl")

    rows = dedupe(load_jsonl(args.input))
    tasks = {t["task_id"]: t for t in load_jsonl(args.tasks) if t.get("task_id")}

    # Select rows to judge; remember their position in `rows` via index map.
    targets: list[tuple[int, dict]] = []
    for i, row in enumerate(rows):
        if args.only_fallbacks and not row.get("judge_fallback"):
            continue
        targets.append((i, row))
    if args.limit:
        targets = targets[: args.limit]

    print(f"Rows: {len(rows)} | to judge: {len(targets)} | output: {out_path}")
    if not targets:
        write_jsonl(out_path, rows)
        print("Nothing to judge; wrote passthrough output.")
        return

    # Build requests; custom_id r{idx} -> rows index, plus extracted answer.
    requests = []
    idx_map: dict[str, int] = {}
    extracted_map: dict[str, object] = {}
    for idx, (row_pos, row) in enumerate(targets):
        req = build_request(idx, row, tasks.get(row.get("task_id")), judge_cfg)
        idx_map[req["custom_id"]] = row_pos
        extracted_map[req["custom_id"]] = req.pop("_extracted")
        requests.append(req)

    if args.dry_run:
        sample = requests[0]
        print("--- DRY RUN: first request preview ---")
        print(json.dumps({"custom_id": sample["custom_id"],
                          "model": sample["params"]["model"],
                          "system_cached": isinstance(sample["params"]["system"], list)},
                         indent=2))
        print(f"Would submit {len(requests)} requests as one Message Batch.")
        return

    try:
        import anthropic
    except ImportError:
        sys.exit("Install anthropic: pip install anthropic")
    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("Set ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    batch_id = args.resume_batch
    if batch_id is None:
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"Submitted batch {batch_id} ({len(requests)} requests).")
        print(f"  Resume later with: --resume-batch {batch_id}")

    # Poll until processing ends.
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(f"  batch {batch_id}: {status} | {counts}")
        if status == "ended":
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    graded = 0
    failed = 0
    for entry in client.messages.batches.results(batch_id):
        cid = entry.custom_id
        row_pos = idx_map.get(cid)
        if row_pos is None:
            continue
        result_type = entry.result.type
        if result_type == "succeeded":
            text = entry.result.message.content[0].text
            ok = apply_verdict(rows[row_pos], text, extracted_map.get(cid))
            graded += int(ok)
            failed += int(not ok)
        else:
            failed += 1

    write_jsonl(out_path, rows)
    print(f"\nDone. Graded: {graded} | Failed/fallback: {failed} | Output: {out_path}")


if __name__ == "__main__":
    main()
