#!/usr/bin/env python3
"""Offline OLMo baseline repair: post-think fair-copy strip + DeepSeek picker.

Mirrors ``scripts/reprocess_glm_baselines.py`` and Script-1 selection in
``scripts/generate_baselines.py``, with OLMo-specific extras:

  - malformed ``</think#`` close markers (via ``prepare_baseline_raw_text``)
  - recomputed degeneracy on the stripped fair copy
  - audited seed substitutions (untouched stored samples only)
  - optional deterministic + Claude re-grading

Usage:
    python scripts/reprocess_olmo_baselines.py
    python scripts/reprocess_olmo_baselines.py --regrade
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

from answer_judge import grade_sample, write_judge_raw_log  # noqa: E402
from cot_segmentation import (  # noqa: E402
    DEFAULT_RUNAWAY_MAX_STEPS,
    detect_degenerate_trace,
    segment_cot,
    steps_to_baseline_cot,
)
from evaluator import (  # noqa: E402
    extract_domain_answer,
    extract_final_answer,
    gross_numeric_mismatch,
    is_correct,
)
from llm_backend import prepare_baseline_raw_text  # noqa: E402
from model_family import resolve_model_family  # noqa: E402

# Audited untouched seed substitutions. Keys are short task stems.
SEED_OVERRIDES: dict[str, int] = {
    "algebra_114": 43,
    "prealgebra_307": 46,
    "intermediate_algebra_894": 43,
    "counting_and_probability_282": 45,
    "intermediate_algebra_966": 42,
    "geometry_967": 43,
    "geometry_483": 44,
    "intermediate_algebra_690": 42,
    "number_theory_820": 44,
    "number_theory_1002": 42,
    "number_theory_1055": 42,
    "intermediate_algebra_1123": 43,
    "algebra_2102": 46,
    "algebra_1214": 43,
    "number_theory_627": 44,
    "algebra_1275": 44,
    "algebra_2551": 42,
    "prealgebra_2019": 45,
    "algebra_518": 46,
    "intermediate_algebra_1365": 46,
    "algebra_1082": 45,
    "number_theory_769": 45,
}

ADJUDICATIONS: dict[tuple[str, int], dict] = {
    ("math500_test_algebra_2735.json", 44): {
        "winner": "det",
        "reason": "det_wins_parsed_incomplete",
    },
    ("math500_test_algebra_2253.json", 45): {
        "winner": "judge",
        "reason": "judge_wins_boxed_format",
    },
}

REGEN_TASK_STEMS = {
    "counting_and_probability_14",
    "prealgebra_996",
    "algebra_1814",
    "algebra_661",
    "number_theory_931",
    "intermediate_algebra_158",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]



def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def resolve_judge_model() -> str:
    import yaml

    cfg_path = ROOT / "configs/experiment.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        model = (cfg.get("judge") or {}).get("model")
        if model:
            return str(model)
    return "claude-sonnet-4-6"


def apply_adjudication(sample: dict) -> None:
    key = (sample.get("task_id"), sample.get("seed"))
    adj = ADJUDICATIONS.get(key)
    if not adj:
        return
    winner = adj["winner"]
    if winner == "det":
        sample["correct"] = bool(sample.get("det_correct"))
    elif winner == "judge":
        sample["correct"] = True
    sample["adjudication"] = adj
    sample["judge_det_conflict"] = False

def _fmt_secs(secs: float) -> str:
    return f"{secs:.1f}s"


def _task_stem(task_id: str) -> str:
    name = task_id
    if name.startswith("math500_test_"):
        name = name[len("math500_test_") :]
    if name.endswith(".json"):
        name = name[: -len(".json")]
    return name


def reprocess_sample(
    sample: dict,
    family,
    min_steps: int,
    *,
    runaway_max_steps: int,
    regrade: bool,
    judge_cfg: dict | None,
    math_tolerance: float,
    science_tolerance: float,
) -> dict:
    """Strip think channel, resegment, recompute degeneracy, optionally re-grade."""
    updated = dict(sample)
    raw_output = str(sample.get("raw_output", ""))
    updated["original_step_count"] = sample.get("step_count")
    updated["original_degenerate"] = sample.get("degenerate")
    updated["original_degenerate_reason"] = sample.get("degenerate_reason")
    updated["original_correct"] = sample.get("correct")
    updated["original_det_correct"] = sample.get("det_correct")
    updated["original_parsed_answer"] = sample.get("parsed_answer")

    if not raw_output.strip():
        updated["steps"] = []
        updated["step_count"] = 0
        updated["baseline_cot"] = ""
        updated["segmentation_method"] = "no_raw_output"
        updated["fair_copy_extraction"] = "empty_raw"
        updated["stripped_raw"] = ""
        return updated

    stripped = prepare_baseline_raw_text(raw_output, think_tags=family.think_tags)
    extraction = "post_think_fair_copy" if stripped != raw_output.strip() else "no_think_tag"
    updated["stripped_raw"] = stripped
    updated["fair_copy_extraction"] = extraction

    eval_type = sample.get("eval_type") or "symbolic"
    domain = sample.get("domain") or "math"
    gold = str(sample.get("gold_answer", ""))

    # Extract the final answer from the full fair-copy text (includes
    # "Final answer: …") BEFORE segmentation. Parsing from steps alone
    # often mines intermediate numbers and falsely flips correctness.
    parsed = extract_domain_answer(stripped, domain, eval_type)
    if parsed is None:
        parsed = extract_final_answer(stripped)
    if parsed is None:
        parsed = sample.get("parsed_answer")
    updated["parsed_answer"] = parsed

    steps, method = segment_cot(
        stripped,
        min_steps=min_steps,
        backend=None,
        model_cfg={"segmentation_backend": "none"},
    )
    updated["steps"] = steps
    updated["step_count"] = len(steps)
    updated["segmentation_method"] = method
    updated["baseline_cot"] = steps_to_baseline_cot(steps, parsed)

    is_degenerate, degenerate_reason = detect_degenerate_trace(
        steps, max_steps=runaway_max_steps, raw_text=stripped
    )
    updated["degenerate"] = is_degenerate
    updated["degenerate_reason"] = degenerate_reason

    det = is_correct(parsed, gold, eval_type, math_tolerance=math_tolerance)
    updated["det_correct"] = bool(det) if det is not None else False

    if regrade and judge_cfg is not None:
        grading = grade_sample(
            question=str(sample.get("question") or sample.get("prompt") or ""),
            gold_answer=gold,
            answer_content=updated["baseline_cot"],
            extracted_answer=parsed,
            domain=domain,
            eval_type=eval_type,
            judge_cfg=judge_cfg,
            math_tolerance=math_tolerance,
            science_tolerance=science_tolerance,
            dry_run=False,
        )
        updated["correct"] = bool(grading["correct"])
        updated["verifier_score"] = grading["verifier_score"]
        updated["judge_output"] = grading["judge_output"]
        updated["judge_raw_output"] = grading.get("judge_raw_output", "")
        updated["det_correct"] = grading["det_correct"]
        updated["judge_fallback"] = grading["judge_fallback"]
        updated["judge_raw_path"] = write_judge_raw_log(
            judge_cfg=judge_cfg,
            phase="olmo_baseline_reprocess",
            task_id=sample["task_id"],
            grading=grading,
            dry_run=False,
            sample_index=sample.get("attempt"),
            seed=sample.get("seed"),
            extracted_answer=parsed,
            gold_answer=gold,
        )
        judge_det_conflict = (
            eval_type == "symbolic"
            and not grading["judge_fallback"]
            and grading["correct"] is True
            and not grading["det_correct"]
        )
        sanity_gate_flipped = judge_det_conflict and gross_numeric_mismatch(
            parsed, gold
        )
        if sanity_gate_flipped:
            updated["correct"] = False
        updated["judge_det_conflict"] = judge_det_conflict
        updated["sanity_gate_flipped"] = sanity_gate_flipped
        apply_adjudication(updated)
    else:
        # Prefer refreshed deterministic grade; keep prior judge fields as provenance.
        if updated["det_correct"]:
            updated["correct"] = True
        elif sample.get("judge_fallback"):
            # Previous run had no live judge — trust refreshed det only.
            updated["correct"] = False
        else:
            # Preserve prior adjudicated judge verdict when det still fails.
            updated["correct"] = bool(sample.get("correct"))
        updated["grading_mode"] = "det_refresh_no_live_judge"

    return updated


def pick_baseline(
    samples: list[dict],
    majority: int,
    min_steps: int,
) -> tuple[dict, bool, str | None, bool]:
    """Exact Script-1 / DeepSeek picker."""
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


def apply_seed_override(
    samples: list[dict],
    task_id: str,
    chosen: dict,
    verified: bool,
    exclude_reason: str | None,
    degenerate_fallback: bool,
    majority: int,
    min_steps: int,
) -> tuple[dict, bool, str | None, bool, str | None]:
    """Force an audited seed when configured; return selection_tier note."""
    stem = _task_stem(task_id)
    target_seed = SEED_OVERRIDES.get(stem)
    if target_seed is None:
        return chosen, verified, exclude_reason, degenerate_fallback, None

    override = next((s for s in samples if s.get("seed") == target_seed), None)
    if override is None:
        return chosen, verified, exclude_reason, degenerate_fallback, "override_seed_missing"

    n_correct = sum(1 for s in samples if s.get("correct"))
    if n_correct < majority:
        return chosen, verified, exclude_reason, degenerate_fallback, "override_skipped_below_majority"
    if not override.get("correct"):
        return chosen, verified, exclude_reason, degenerate_fallback, "override_seed_incorrect"
    if override.get("step_count", 0) < min_steps:
        return chosen, verified, exclude_reason, degenerate_fallback, "override_seed_insufficient_steps"

    if override.get("degenerate"):
        return chosen, verified, exclude_reason, degenerate_fallback, "override_seed_degenerate"
    return override, True, None, False, f"audited_seed_override:{target_seed}"


def build_baseline_record(
    task_id: str,
    old_record: dict | None,
    samples: list[dict],
    chosen: dict,
    verified: bool,
    exclude_reason: str | None,
    degenerate_fallback: bool,
    n_samples: int,
    selection_tier: str | None,
) -> dict:
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
        "fair_copy_extraction": chosen.get("fair_copy_extraction"),
        "selection_tier": selection_tier or "deepseek_longest_correct",
        "needs_regeneration": _task_stem(task_id) in REGEN_TASK_STEMS,
    }
    if not verified:
        record["exclude_reason"] = exclude_reason
    if old_record and old_record.get("manual_verify_override"):
        record["prior_manual_verify_override"] = True
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots_samples.jsonl",
    )
    parser.add_argument(
        "--old-baselines",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots.jsonl",
    )
    parser.add_argument(
        "--out-baselines",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots.reprocessed.jsonl",
    )
    parser.add_argument(
        "--out-samples",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots_samples.reprocessed.jsonl",
    )
    parser.add_argument("--model", type=str, default="allenai/Olmo-3-7B-Think")
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--runaway-max-steps", type=int, default=DEFAULT_RUNAWAY_MAX_STEPS)
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="Re-run Claude judge + deterministic grade on every sample",
    )
    parser.add_argument("--math-tolerance", type=float, default=1e-4)
    parser.add_argument("--science-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    t0 = time.monotonic()
    family = resolve_model_family(args.model)
    majority = args.n_samples // 2 + 1

    judge_cfg = None
    if args.regrade:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        judge_cfg = {
            "provider": "anthropic",
            "model": resolve_judge_model(),
            "enabled": True,
        }

    print(f"[olmo-reprocess] samples:       {args.samples}")
    print(f"[olmo-reprocess] old baselines: {args.old_baselines}")
    print(f"[olmo-reprocess] out baselines: {args.out_baselines}")
    print(f"[olmo-reprocess] out samples:   {args.out_samples}")
    print(
        f"[olmo-reprocess] model={args.model} family={family.name} "
        f"close_tag={family.think_tags.close!r} strip=True(reprocess) "
        f"min_steps={args.min_steps} majority={majority}/{args.n_samples} "
        f"regrade={args.regrade}"
    )

    samples_raw = load_jsonl(args.samples)
    olds = {r["task_id"]: r for r in load_jsonl(args.old_baselines)}
    print(f"[olmo-reprocess] samples loaded: {len(samples_raw)} | old baselines: {len(olds)}")

    # Attach prompts from experiment_tasks when regrading (judge needs question).
    prompts: dict[str, str] = {}
    tasks_path = ROOT / "data/experiment_tasks.jsonl"
    if tasks_path.exists():
        for row in load_jsonl(tasks_path):
            prompts[row["task_id"]] = row.get("prompt") or row.get("question") or ""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for s in samples_raw:
        if not s.get("prompt") and not s.get("question"):
            s = dict(s)
            s["prompt"] = prompts.get(s["task_id"], "")
        grouped[s["task_id"]].append(s)

    args.out_baselines.parent.mkdir(parents=True, exist_ok=True)
    out_baseline_rows: list[dict] = []
    out_sample_rows: list[dict] = []

    verified = 0
    exclude_counts: dict[str, int] = defaultdict(int)
    override_counts: dict[str, int] = defaultdict(int)
    tasks_processed = 0
    needs_regen = 0

    for task_id, samples in grouped.items():
        samples.sort(key=lambda s: (s.get("attempt") or 0, s.get("seed") or 0))
        reprocessed = [
            reprocess_sample(
                s,
                family,
                args.min_steps,
                runaway_max_steps=args.runaway_max_steps,
                regrade=args.regrade,
                judge_cfg=judge_cfg,
                math_tolerance=args.math_tolerance,
                science_tolerance=args.science_tolerance,
            )
            for s in samples
        ]
        chosen, is_verified, exclude_reason, degen_fb = pick_baseline(
            reprocessed, majority, args.min_steps
        )
        chosen, is_verified, exclude_reason, degen_fb, tier = apply_seed_override(
            reprocessed,
            task_id,
            chosen,
            is_verified,
            exclude_reason,
            degen_fb,
            majority,
            args.min_steps,
        )
        if tier:
            override_counts[tier.split(":")[0]] += 1
        rec = build_baseline_record(
            task_id=task_id,
            old_record=olds.get(task_id),
            samples=reprocessed,
            chosen=chosen,
            verified=is_verified,
            exclude_reason=exclude_reason,
            degenerate_fallback=degen_fb,
            n_samples=len(reprocessed),
            selection_tier=tier,
        )
        out_baseline_rows.append(rec)
        out_sample_rows.extend(reprocessed)
        if is_verified:
            verified += 1
        elif exclude_reason:
            exclude_counts[exclude_reason] += 1
        if rec.get("needs_regeneration"):
            needs_regen += 1
        tasks_processed += 1
        if tasks_processed % 20 == 0:
            print(f"[olmo-reprocess] … {tasks_processed}/100 verified={verified}")

    write_jsonl_atomic(args.out_baselines, out_baseline_rows)
    write_jsonl_atomic(args.out_samples, out_sample_rows)

    dt = time.monotonic() - t0
    print(
        f"[olmo-reprocess] done: tasks={tasks_processed} verified={verified} "
        f"exclude={dict(exclude_counts)} overrides={dict(override_counts)} "
        f"needs_regen_flag={needs_regen} wall={_fmt_secs(dt)}"
    )
    unverified = [json.loads(l) for l in args.out_baselines.open() if l.strip()]
    bad = [r for r in unverified if not r.get("verified")]
    if bad:
        print("[olmo-reprocess] unverified tasks:")
        for r in bad:
            print(
                f"  {_task_stem(r['task_id'])}: reason={r.get('exclude_reason')} "
                f"solve_rate={r.get('solve_rate')} seed={r.get('seed')} "
                f"needs_regen={r.get('needs_regeneration')}"
            )


if __name__ == "__main__":
    main()
