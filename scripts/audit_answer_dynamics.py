#!/usr/bin/env python3
"""Audit: LLM-judged answer dynamics on a recovery sample (Phase 2c, exploratory).

The standard Script-4 pipeline does NOT compute post-first-answer / answer-attempt
metrics: regex/phrase heuristics on math reasoning are unreliable (they count
intermediate steps as "answers"). Instead we run a small, judged audit:

  * 200 recovery runs PER MODEL, stratified over (error_type x k) -> ~600 calls.
  * Each call asks Claude for a STRUCTURED verdict on one recovery output:
      - tokens_to_first_final_answer_frac  (0..1, position of first explicit
        final answer as a fraction of the output)
      - post_first_answer_continuation_ratio_judge (0..1, share of text AFTER it)
      - answer_attempt_count_judge          (explicit final-answer attempts/revisions,
                                             NOT reasoning steps)
      - judge_rationale                      (one sentence)

Budget: ~$5-15 total (Claude Sonnet, ~2k in + ~200 out per call). This is an
exploratory / appendix analysis; it is NOT a primary hypothesis test.

Outputs:
  * data/audit/answer_dynamics_judge.jsonl   (one row per judged recovery run)
  * results/answer_dynamics_audit.csv        (aggregate per model x error_type x k)

Idempotent: rows already present in the JSONL (by dialogue_id) are skipped, so the
script can resume after a credit-out. Use --dry-run to exercise sampling + I/O
without any API calls.

Usage:
  python scripts/audit_answer_dynamics.py --dry-run
  python scripts/audit_answer_dynamics.py --per-model 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    JudgeCreditError,
    call_anthropic_judge,
)

DEFAULT_INPUT = ROOT / "data/post_injection_runs.jsonl"
DEFAULT_TASKS = ROOT / "data/experiment_tasks.jsonl"
DEFAULT_OUT = ROOT / "data/audit/answer_dynamics_judge.jsonl"
DEFAULT_CSV = ROOT / "results/answer_dynamics_audit.csv"

AUDIT_SYSTEM_PROMPT = (
    "You analyse the reasoning output a model produced AFTER an interruption, to "
    "study overthinking. You are given the question, the reference (gold) answer, "
    "and the model's full output. Identify the FIRST point where the model states "
    "an explicit FINAL answer (e.g. a boxed answer or 'the answer is X') — NOT an "
    "intermediate calculation. Then quantify how much the model keeps going.\n\n"
    "Respond with ONLY a JSON object (no prose) with these keys:\n"
    '  "first_final_answer_present": true|false,\n'
    '  "tokens_to_first_final_answer_frac": number 0..1 (fraction of the output '
    "before the first explicit final answer; 0 if at the very start),\n"
    '  "post_first_answer_continuation_ratio": number 0..1 (fraction of the output '
    "AFTER the first explicit final answer),\n"
    '  "answer_attempt_count": integer >=0 (how many DISTINCT explicit final-answer '
    "statements / revisions appear, not reasoning steps),\n"
    '  "judge_rationale": one short sentence.\n'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def stratified_sample(
    rows: list[dict], per_model: int, seed: int
) -> list[dict]:
    """Round-robin over (error_type x k) strata within each model, up to per_model."""
    import random

    rng = random.Random(seed)
    by_model: dict[str, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_model[r.get("model")][(r.get("error_type"), r.get("k"))].append(r)

    selected: list[dict] = []
    for model, strata in by_model.items():
        for stratum in strata.values():
            rng.shuffle(stratum)
        keys = sorted(strata.keys(), key=lambda x: tuple(str(v) for v in x))
        picked: list[dict] = []
        cursors = {key: 0 for key in keys}
        while len(picked) < per_model:
            progressed = False
            for key in keys:
                if len(picked) >= per_model:
                    break
                c = cursors[key]
                if c < len(strata[key]):
                    picked.append(strata[key][c])
                    cursors[key] = c + 1
                    progressed = True
            if not progressed:
                break  # exhausted all strata for this model
        selected.extend(picked)
    return selected


def parse_audit_json(text: str) -> dict | None:
    """Extract the first JSON object from the judge response, defensively."""
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def build_audit_prompt(*, question: str, gold: str, recovery_output: str) -> str:
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Reference (gold) answer:\n{gold.strip()}\n\n"
        f"Model output to analyse:\n{recovery_output.strip()}"
    )


def _num(value, lo: float | None = None, hi: float | None = None) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def aggregate(rows: list[dict]) -> list[dict]:
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r.get("model"), r.get("error_type"), r.get("k"))].append(r)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    out: list[dict] = []
    for (model, et, k), items in sorted(by_cell.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out.append({
            "model": model,
            "error_type": et,
            "k": k,
            "n": len(items),
            "post_first_answer_continuation_ratio_judge_mean": _mean(
                [it.get("post_first_answer_continuation_ratio") for it in items]
            ),
            "answer_attempt_count_judge_mean": _mean(
                [it.get("answer_attempt_count") for it in items]
            ),
            "first_final_answer_present_rate": _mean(
                [1.0 if it.get("first_final_answer_present") else 0.0 for it in items]
            ),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judged answer-dynamics audit (recovery sample).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--per-model", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--generation-mode", type=str, default="continuation")
    parser.add_argument("--dry-run", action="store_true", help="No API calls; emit stub verdicts.")
    args = parser.parse_args()

    runs = load_jsonl(args.input)
    tasks = {t.get("task_id"): t for t in load_jsonl(args.tasks)}
    recovery = [
        r for r in runs
        if r.get("variant") == "recovery"
        and (args.generation_mode is None or r.get("generation_mode") == args.generation_mode)
        and (r.get("raw_output") or "").strip()
    ]
    if not recovery:
        sys.exit(f"[FAIL] No recovery rows in {args.input} (mode={args.generation_mode}).")

    sample = stratified_sample(recovery, args.per_model, args.seed)
    print(f"[INFO] Sampled {len(sample)} recovery runs "
          f"({args.per_model}/model x {len({r.get('model') for r in sample})} models).")

    done_ids = {r.get("dialogue_id") for r in load_jsonl(args.out)}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    judged: list[dict] = list(load_jsonl(args.out))
    n_new = 0
    with args.out.open("a", encoding="utf-8") as fout:
        for i, run in enumerate(sample):
            did = run.get("dialogue_id")
            if did in done_ids:
                continue
            task = tasks.get(run.get("task_id"), {})
            question = task.get("prompt", "")
            gold = str(run.get("gold_answer", task.get("gold_answer", "")))
            recovery_output = run.get("raw_output", "")

            if args.dry_run:
                verdict = {
                    "first_final_answer_present": True,
                    "tokens_to_first_final_answer_frac": 0.5,
                    "post_first_answer_continuation_ratio": 0.5,
                    "answer_attempt_count": 1,
                    "judge_rationale": "dry-run stub",
                }
                raw = json.dumps(verdict)
            else:
                prompt = build_audit_prompt(
                    question=question, gold=gold, recovery_output=recovery_output
                )
                try:
                    raw = call_anthropic_judge(
                        prompt=prompt,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=0.0,
                        system=AUDIT_SYSTEM_PROMPT,
                    )
                except JudgeCreditError as exc:
                    print(f"\n[ABORT] {exc}\n[ABORT] Re-run after topping up credits "
                          "(idempotent: judged rows are skipped).", file=sys.stderr)
                    sys.exit(2)
                verdict = parse_audit_json(raw) or {}

            row = {
                "dialogue_id": did,
                "task_id": run.get("task_id"),
                "model": run.get("model"),
                "error_type": run.get("error_type"),
                "k": run.get("k"),
                "length_level": run.get("length_level"),
                "seed": run.get("seed"),
                "recovery_tokens_generated": run.get("tokens_generated"),
                "first_final_answer_present": bool(verdict.get("first_final_answer_present")),
                "tokens_to_first_final_answer_frac": _num(
                    verdict.get("tokens_to_first_final_answer_frac"), 0.0, 1.0
                ),
                "post_first_answer_continuation_ratio": _num(
                    verdict.get("post_first_answer_continuation_ratio"), 0.0, 1.0
                ),
                "answer_attempt_count": _num(verdict.get("answer_attempt_count"), 0.0),
                "judge_rationale": verdict.get("judge_rationale"),
                "judge_raw_output": raw.strip()[:2000],
                "judge_parsed": bool(verdict),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            judged.append(row)
            n_new += 1
            if (i + 1) % 50 == 0:
                print(f"  judged {i + 1}/{len(sample)} (new this run: {n_new})", flush=True)

    print(f"[OK] {args.out}  (+{n_new} new, {len(judged)} total)")

    # Aggregate (pandas optional; fall back to manual CSV).
    agg_rows = aggregate(judged)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(agg_rows).to_csv(args.csv, index=False)
    except ImportError:
        import csv

        if agg_rows:
            with args.csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
                w.writeheader()
                w.writerows(agg_rows)
    print(f"[OK] {args.csv}  ({len(agg_rows)} cells)")


if __name__ == "__main__":
    main()
