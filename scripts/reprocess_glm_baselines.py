#!/usr/bin/env python3
"""Offline reprocess of GLM baseline samples.

Applies the (now-configured) baseline_strip_think transformation to existing
raw_output samples WITHOUT re-running vLLM. Because the original 5x baseline
run was executed BEFORE `baseline_strip_think: true` existed, the stored
`steps[]` contain the GLM-specific mess (`</think>` mid-list, "wait/hmm"
prose, draft+fair-copy duplication).

Given the raw_output field is already persisted for every sample, we can
re-derive clean `steps[]` by:
    1. `prepare_baseline_raw_text(raw_output)` — strip everything up to and
       including the last `</think>` (falls back to raw text if no tag).
    2. `segment_cot(...)` — same parser as Script 1 (deterministic; the
       llm_resegment backend fallback is never invoked because
       `parse_numbered_steps` succeeds on 494/500 correct GLM samples).
    3. Preserve every other sample field (correctness, seed, judge output).
       Answer correctness lives in the post-`</think>` region, which the
       strip keeps intact — so `correct`/`parsed_answer` do not change.
    4. Repeat Script-1's baseline picker (max non-degenerate step_count
       among correct samples; majority-vote; min_steps gate).

Writes:
    - <baselines-dir>/baseline_cots_5x.reprocessed.jsonl        (100 rows)
    - <baselines-dir>/baseline_cots_5x_samples.reprocessed.jsonl (500 rows)

Original inputs (baseline_cots_5x.jsonl / baseline_cots_5x_samples.jsonl)
are NOT modified; the report script (Phase 4) diffs old vs new.

Usage (Phase 2 of the GLM plan; math-only run, min_steps=3):
    python scripts/reprocess_glm_baselines.py \\
        --samples backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x_samples.jsonl \\
        --old-baselines backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x.jsonl \\
        --out-baselines backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x.reprocessed.jsonl \\
        --out-samples   backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x_samples.reprocessed.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cot_segmentation import segment_cot, steps_to_baseline_cot  # noqa: E402
from llm_backend import prepare_baseline_raw_text  # noqa: E402
from model_family import resolve_model_family  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _fmt_secs(secs: float) -> str:
    return f"{secs:.1f}s"


def reprocess_sample(sample: dict, family, min_steps: int) -> dict:
    """Return a copy of sample with re-derived steps/step_count/baseline_cot.

    Preserves all other fields — including `correct`, `parsed_answer`,
    `judge_output`, `seed`. Segmentation runs without an LLM resegment
    backend; if `parse_numbered_steps` fails we still get whatever the
    remaining paragraph/sentence-based fallbacks yield.
    """
    updated = dict(sample)
    raw_output = str(sample.get("raw_output", ""))
    if not raw_output.strip():
        updated["steps"] = []
        updated["step_count"] = 0
        updated["baseline_cot"] = ""
        updated["segmentation_method"] = "no_raw_output"
        return updated

    stripped = prepare_baseline_raw_text(raw_output, think_tags=family.think_tags)
    steps, method = segment_cot(
        stripped,
        min_steps=min_steps,
        backend=None,
        model_cfg={"segmentation_backend": "none"},
    )
    updated["steps"] = steps
    updated["step_count"] = len(steps)
    updated["segmentation_method"] = method
    # Rebuild baseline_cot with the SAME final answer (preserved from the
    # original run's grading; the strip does not change the answer region).
    updated["baseline_cot"] = steps_to_baseline_cot(steps, sample.get("parsed_answer"))
    return updated


def pick_baseline(
    samples: list[dict],
    majority: int,
    min_steps: int,
) -> tuple[dict, bool, str | None, bool]:
    """Re-implement Script 1's picker on the reprocessed samples.

    Returns (chosen_sample, verified, exclude_reason, degenerate_fallback).
    """
    correct = [s for s in samples if s.get("correct")]
    non_degen_correct = [s for s in correct if not s.get("degenerate")]
    pool = non_degen_correct or correct
    degenerate_fallback = bool(correct) and not non_degen_correct

    n_correct = len(correct)
    if n_correct >= majority and correct:
        chosen = max(pool, key=lambda s: s.get("step_count", 0))
        if chosen.get("step_count", 0) >= min_steps:
            return chosen, True, None, degenerate_fallback
        return chosen, False, "insufficient_steps", degenerate_fallback

    chosen = (
        max(pool, key=lambda s: s.get("step_count", 0))
        if pool
        else (samples[-1] if samples else {})
    )
    return chosen, False, "below_majority", degenerate_fallback


def build_baseline_record(
    task_id: str,
    old_record: dict | None,
    samples: list[dict],
    chosen: dict,
    verified: bool,
    exclude_reason: str | None,
    degenerate_fallback: bool,
    n_samples: int,
) -> dict:
    """Preserve the exact field layout Script 1 emits."""
    solve_rate = sum(1 for s in samples if s.get("correct")) / n_samples if n_samples else 0.0
    record = {
        "task_id": task_id,
        "domain": (old_record or samples[0]).get("domain"),
        "eval_type": (old_record or samples[0]).get("eval_type"),
        "model": (old_record or samples[0]).get("model"),
        "baseline_cot": chosen.get("baseline_cot", ""),
        "steps": chosen.get("steps", []),
        "step_count": chosen.get("step_count", 0),
        "gold_answer": (old_record or samples[0]).get("gold_answer"),
        "parsed_answer": chosen.get("parsed_answer"),
        "correct": bool(chosen.get("correct")) if verified else False,
        "verified": verified,
        "solve_rate": solve_rate,
        "n_verification_samples": n_samples,
        "seed": chosen.get("seed"),
        "segmentation_method": chosen.get("segmentation_method", "prompt"),
        "sample_step_counts": [s.get("step_count", 0) for s in samples],
        "sample_segmentation_methods": [s.get("segmentation_method", "prompt") for s in samples],
        "sample_correct": [bool(s.get("correct")) for s in samples],
        "sample_seeds": [s.get("seed") for s in samples],
        "sample_degenerate": [bool(s.get("degenerate")) for s in samples],
        "sample_degenerate_reasons": [s.get("degenerate_reason") for s in samples],
        "baseline_degenerate_fallback": degenerate_fallback,
        "sample_judge_det_conflict": [bool(s.get("judge_det_conflict")) for s in samples],
        "sample_sanity_gate_flipped": [bool(s.get("sanity_gate_flipped")) for s in samples],
    }
    if not verified:
        record["exclude_reason"] = exclude_reason
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT
        / "backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x_samples.jsonl",
    )
    parser.add_argument(
        "--old-baselines",
        type=Path,
        default=ROOT
        / "backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x.jsonl",
    )
    parser.add_argument(
        "--out-baselines",
        type=Path,
        default=ROOT
        / "backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x.reprocessed.jsonl",
    )
    parser.add_argument(
        "--out-samples",
        type=Path,
        default=ROOT
        / "backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x_samples.reprocessed.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="zai-org/GLM-Z1-9B-0414",
        help="Used to resolve model family think tags (default: GLM-Z1-9B).",
    )
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=5, help="verification_samples per task")
    args = parser.parse_args()

    t0 = time.monotonic()
    family = resolve_model_family(args.model)
    majority = args.n_samples // 2 + 1

    print(f"[reprocess] samples:       {args.samples}")
    print(f"[reprocess] old baselines: {args.old_baselines}")
    print(f"[reprocess] out baselines: {args.out_baselines}")
    print(f"[reprocess] out samples:   {args.out_samples}")
    print(
        f"[reprocess] model={args.model} family={family.name} "
        f"close_tag={family.think_tags.close!r} strip_think={family.baseline_strip_think} "
        f"min_steps={args.min_steps} majority={majority}/{args.n_samples}"
    )

    samples_raw = load_jsonl(args.samples)
    olds = {r["task_id"]: r for r in load_jsonl(args.old_baselines)}
    print(f"[reprocess] samples loaded: {len(samples_raw)} | old baselines: {len(olds)}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for s in samples_raw:
        grouped[s["task_id"]].append(s)

    args.out_baselines.parent.mkdir(parents=True, exist_ok=True)
    args.out_samples.parent.mkdir(parents=True, exist_ok=True)
    args.out_baselines.write_text("")
    args.out_samples.write_text("")

    verified = 0
    exclude_counts: dict[str, int] = defaultdict(int)
    tasks_processed = 0
    for task_id, samples in grouped.items():
        samples.sort(key=lambda s: (s.get("attempt") or 0, s.get("seed") or 0))
        reprocessed = [reprocess_sample(s, family, args.min_steps) for s in samples]
        chosen, is_verified, exclude_reason, degen_fb = pick_baseline(
            reprocessed, majority, args.min_steps
        )
        rec = build_baseline_record(
            task_id=task_id,
            old_record=olds.get(task_id),
            samples=reprocessed,
            chosen=chosen,
            verified=is_verified,
            exclude_reason=exclude_reason,
            degenerate_fallback=degen_fb,
            n_samples=len(reprocessed),
        )
        with args.out_baselines.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with args.out_samples.open("a", encoding="utf-8") as f:
            for s in reprocessed:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        if is_verified:
            verified += 1
        elif exclude_reason:
            exclude_counts[exclude_reason] += 1
        tasks_processed += 1

    dt = time.monotonic() - t0
    print(
        f"[reprocess] done: tasks={tasks_processed} verified={verified} "
        f"exclude={dict(exclude_counts)} wall={_fmt_secs(dt)}"
    )


if __name__ == "__main__":
    main()
