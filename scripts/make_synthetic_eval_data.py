#!/usr/bin/env python3
"""
Generate a small synthetic dataset for end-to-end testing of Script 4
(evaluate.py) with NO GPU and NO API. Schema matches Scripts 1 & 3 exactly.

Design (math domain only):
  5 tasks x 3 error_types x k=1 x short x 2 seeds x {stop, recovery}
  = 60 perturbed rows, plus matching baseline (clean) rows.

Properties exercised on purpose:
  - stop is mostly WRONG, recovery mostly RIGHT  -> positive, non-trivial REI
  - a few recovery rows have recovery_terminated_by == "token_limit"
    (high tokens, still wrong) -> overthinking signal + n<min_recovery_seeds path
  - clean accuracy spans easy / medium / hard via multiple baseline rows per task

Outputs:
  data/perturbed_runs_synthetic.jsonl
  data/baseline_cots_synthetic.jsonl

Usage:
  python scripts/make_synthetic_eval_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
ERROR_TYPES = ["incorrect_reasoning", "off_topic", "paraphrased_shift"]
SEEDS = [42, 43]

# task_id -> (gold, decoy, list of baseline correctness flags -> clean accuracy)
# clean accuracy controls the difficulty bin (easy>=0.70, hard<=0.40).
TASKS = {
    "synth_000": {"gold": "10", "decoy": "99", "baseline_correct": [True]},                 # 1.00 easy
    "synth_001": {"gold": "20", "decoy": "98", "baseline_correct": [True]},                 # 1.00 easy
    "synth_002": {"gold": "30", "decoy": "97", "baseline_correct": [True, False]},          # 0.50 medium
    "synth_003": {"gold": "40", "decoy": "96", "baseline_correct": [True, False]},          # 0.50 medium
    "synth_004": {"gold": "50", "decoy": "95", "baseline_correct": [True, False, False]},   # 0.33 hard
}


def baseline_rows() -> list[dict]:
    rows: list[dict] = []
    for tid, spec in TASKS.items():
        gold = spec["gold"]
        for i, correct in enumerate(spec["baseline_correct"]):
            parsed = gold if correct else spec["decoy"]
            rows.append(
                {
                    "task_id": tid,
                    "domain": "math",
                    "model": MODEL,
                    "baseline_cot": f"Step 1: setup.\nStep 2: solve.\nStep 3: check.\nFinal answer: {parsed}",
                    "steps": ["Step 1: setup.", "Step 2: solve.", "Step 3: check."],
                    "step_count": 3,
                    "gold_answer": gold,
                    "parsed_answer": parsed,
                    "verified": bool(correct),
                    "retries": i + 1,
                    "seed": 42 + i,
                    "segmentation_method": "prompt",
                }
            )
    return rows


def _stop_row(tid: str, et: str, seed: int, gold: str, decoy: str) -> dict:
    # stop mostly wrong: only synth_000 + off_topic is correct, everything else wrong
    correct = tid == "synth_000" and et == "off_topic"
    extracted = gold if correct else decoy
    return {
        "task_id": tid,
        "model": MODEL,
        "error_type": et,
        "k": 1,
        "length_level": "short",
        "seed": seed,
        "variant": "stop",
        "extracted_answer": extracted,
        "gold_answer": gold,
        "correct": correct,
        "tokens_generated": 48 + (seed - 42),
        "raw_output": f"\n\nTherefore, the final answer is \\boxed{{{extracted}}}",
        "midpoint_step": 1,
        "domain": "math",
    }


def _recovery_row(tid: str, et: str, seed: int, gold: str, decoy: str) -> dict:
    # token_limit (overthinking) rows: incorrect_reasoning + seed 43 -> many tokens, wrong.
    # This also leaves incorrect_reasoning with only 1 natural seed (< min_recovery_seeds=2).
    token_limit = et == "incorrect_reasoning" and seed == 43
    if token_limit:
        terminated_by = "token_limit"
        correct = False
        extracted = decoy
        tokens = 8192
    else:
        terminated_by = "natural"
        # recovery mostly right; make synth_004 (hard) stay wrong to vary signal
        correct = tid != "synth_004"
        extracted = gold if correct else decoy
        tokens = 300 + (seed - 42) * 12
    return {
        "task_id": tid,
        "model": MODEL,
        "error_type": et,
        "k": 1,
        "length_level": "short",
        "seed": seed,
        "variant": "recovery",
        "extracted_answer": extracted,
        "gold_answer": gold,
        "correct": correct,
        "tokens_generated": tokens,
        "raw_output": f"Step 1: reconsider.\nStep 2: recompute.\nFinal answer: {extracted}\n",
        "midpoint_step": 1,
        "domain": "math",
        "recovery_terminated_by": terminated_by,
    }


def perturbed_rows() -> list[dict]:
    rows: list[dict] = []
    for tid, spec in TASKS.items():
        gold, decoy = spec["gold"], spec["decoy"]
        for et in ERROR_TYPES:
            for seed in SEEDS:
                rows.append(_stop_row(tid, et, seed, gold, decoy))
                rows.append(_recovery_row(tid, et, seed, gold, decoy))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    perturbed = perturbed_rows()
    baselines = baseline_rows()
    perturbed_path = ROOT / "data/perturbed_runs_synthetic.jsonl"
    baseline_path = ROOT / "data/baseline_cots_synthetic.jsonl"
    write_jsonl(perturbed_path, perturbed)
    write_jsonl(baseline_path, baselines)
    print(f"[OK] {perturbed_path}  ({len(perturbed)} rows)")
    print(f"[OK] {baseline_path}  ({len(baselines)} rows)")


if __name__ == "__main__":
    main()
