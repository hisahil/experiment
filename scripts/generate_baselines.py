#!/usr/bin/env python3
"""
Script 1: Generate verified baseline CoTs per task per experiment model.

Majority-vote solvability filter (von Recum et al., ICLR 2026, use N=8). Per task we draw a
FIXED verification_samples runs (seeds base_seed+0..N-1, temperature from experiment.yaml — no
early stop). A run's vote counts ONLY answer correctness:
  verified = (#correct >= ceil(verification_samples / 2))   # majority, 3 of 5 by default
The baseline written is the correct run with the MOST steps. The step requirement is decoupled
from the vote and applied only to that chosen baseline:
  - majority reached, best correct run step_count >= min_steps → verified
  - majority reached, but best correct run step_count < min_steps → exclude_reason: insufficient_steps
  - majority not reached → exclude_reason: below_majority
solve_rate (= #correct / N) is logged per task. verified=false tasks are excluded from Script 3;
see exclude_reason in baseline_cots.jsonl.

Outputs:
  data/baseline_cots.jsonl         — chosen baseline per task/model
  data/baseline_cots_samples.jsonl — every verification sample, for audits

step_count is used by Script 3 for midpoint injection. The sample detail file is
observability only; downstream scripts ignore it.

Prompt template: prompts/baseline_generation.txt ({task} = task prompt from JSONL).
Tells the experiment model to answer with numbered "Step N:" lines (>= min_steps)
and a final "Final answer:" line so Script 1 can segment the CoT and grade it.
Not used by Script 3 — see prompts/corrupted_external_cot.txt and stop_after_corruption.txt.

Usage:
  python scripts/generate_baselines.py --dry-run
  python scripts/generate_baselines.py --domain math --limit 5
  python scripts/generate_baselines.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

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
)
from llm_backend import (  # noqa: E402
    build_baseline_prompt,
    create_backend,
    prepare_baseline_raw_text,
)
from model_family import apply_baseline_sampling, resolve_model_family  # noqa: E402

DEFAULT_BASELINE_CONFIG = ROOT / "configs/baseline_generation.yaml"
DEFAULT_EXPERIMENT_CONFIG = ROOT / "configs/experiment.yaml"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs: str) -> str:
    """Replace {key} placeholders without interpreting LaTeX braces."""
    out = template
    for key, val in kwargs.items():
        out = out.replace("{" + key + "}", val)
    return out


def load_tasks(path: Path, domain: str | None) -> list[dict]:
    tasks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            if domain is None or task.get("domain") == domain:
                tasks.append(task)
    return tasks


def load_verified_keys(out_path: Path) -> set[str]:
    """Keys task_id|model already written with verified=true."""
    if not out_path.exists():
        return set()
    keys: set[str] = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("verified"):
                    keys.add(f"{rec['task_id']}|{rec['model']}")
            except (json.JSONDecodeError, KeyError):
                pass
    return keys


def baseline_record_key(record: dict) -> str:
    return f"{record['task_id']}|{record['model']}"


def dedupe_baseline_file(out_path: Path) -> None:
    """Keep the last row per task_id|model (fixes append-on-resume duplicates)."""
    if not out_path.exists():
        return
    by_key: dict[str, dict] = {}
    order: list[str] = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = baseline_record_key(rec)
            if key not in by_key:
                order.append(key)
            by_key[key] = rec
    with out_path.open("w", encoding="utf-8") as f:
        for key in order:
            f.write(json.dumps(by_key[key], ensure_ascii=False) + "\n")


def upsert_baseline_record(out_path: Path, record: dict) -> None:
    key = baseline_record_key(record)
    existing: list[dict] = []
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if baseline_record_key(rec) != key:
                    existing.append(rec)
    existing.append(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in existing:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def replace_sample_records(
    samples_path: Path,
    *,
    task_id: str,
    model: str,
    records: list[dict],
) -> None:
    sample_key = f"{task_id}|{model}"
    kept: list[dict] = []
    if samples_path.exists():
        with samples_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if f"{rec.get('task_id')}|{rec.get('model')}" != sample_key:
                    kept.append(rec)
    kept.extend(records)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_record(out_path: Path, record: dict) -> None:
    upsert_baseline_record(out_path, record)


def sample_detail_path(out_path: Path) -> Path:
    """Sidecar JSONL with one row per verification sample."""
    return out_path.with_name(f"{out_path.stem}_samples{out_path.suffix}")


def _fmt_counts(counter: Counter[str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counter.items())) or "none"


def resolve_models(cfg: dict, experiment_cfg: dict, cli_model: str | None) -> list[str]:
    if cli_model:
        return [cli_model]
    baseline_cfg = cfg.get("models") or experiment_cfg.get("baseline", {}).get("models")
    if baseline_cfg:
        return list(baseline_cfg)
    return [experiment_cfg["model"]["name"]]


def _dry_run_pad_output(raw_text: str, gold_answer: str, min_steps: int) -> str:
    """Dry-run backend returns few steps; pad for pipeline dry-run tests."""
    steps, _ = segment_cot(raw_text, min_steps=1)
    while len(steps) < min_steps:
        n = len(steps) + 1
        steps.append(f"Step {n}: Continue reasoning toward the solution.")
    return steps_to_baseline_cot(steps, gold_answer)


def generate_baseline(
    backend,
    prompt_template: str,
    task: dict,
    model_cfg: dict,
    min_steps: int,
    *,
    dry_run: bool = False,
    domain: str = "math",
    eval_type: str = "exact",
    seed: int | None = None,
    family=None,
) -> tuple[str, list[str], str, str]:
    """Return (baseline_cot, steps, segmentation_method, raw_output)."""
    task_text = task["prompt"]
    use_chat = bool(getattr(family, "baseline_use_chat_template", False)) if family else False
    strip_think = bool(getattr(family, "baseline_strip_think", False)) if family else False
    prompt = build_baseline_prompt(
        backend.tokenizer,
        prompt_template,
        task_text,
        use_chat_template=use_chat,
    )
    top_k = model_cfg.get("top_k")
    rep_pen = model_cfg.get("repetition_penalty")
    result = backend.generate(
        prompt=prompt,
        max_new_tokens=int(model_cfg.get("max_new_tokens", 2048)),
        temperature=float(model_cfg.get("temperature", 0.6)),
        top_p=float(model_cfg.get("top_p", 0.95)),
        top_k=int(top_k) if top_k is not None else None,
        repetition_penalty=float(rep_pen) if rep_pen is not None else None,
        seed=seed,
    )
    raw_text = result.text
    # use_chat (Gemma) implies strip; strip_think (GLM) triggers strip without
    # switching to the chat-template prompt, keeping the flat-text baseline
    # prompt aligned with DeepSeek.
    if use_chat or strip_think:
        raw_text = prepare_baseline_raw_text(raw_text, think_tags=getattr(family, "think_tags", None))
    if dry_run:
        raw_text = _dry_run_pad_output(raw_text, str(task["gold_answer"]), min_steps)

    steps, method = segment_cot(
        raw_text,
        min_steps=min_steps,
        backend=backend,
        model_cfg=model_cfg,
    )
    parsed = extract_domain_answer(raw_text, domain, eval_type)
    if parsed is None and dry_run:
        parsed = str(task["gold_answer"])
    baseline_cot = steps_to_baseline_cot(steps, parsed)
    return baseline_cot, steps, method, result.text


def main() -> None:
    baseline_yaml = load_yaml(DEFAULT_BASELINE_CONFIG)
    experiment_cfg = load_yaml(DEFAULT_EXPERIMENT_CONFIG)

    parser = argparse.ArgumentParser(description="Generate verified baseline CoTs (Script 1).")
    parser.add_argument("--config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="Model and paths from experiment.yaml",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="Task pool JSONL (default from config)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Mock backend (no GPU)")
    parser.add_argument("--model", type=str, default=None, help="Single experiment model")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Override experiment.seed used for verification sample seeds",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    experiment_cfg = load_yaml(args.experiment_config)
    baseline_exp = experiment_cfg.get("baseline", {})
    merged = {**baseline_yaml, **cfg, **baseline_exp}

    paths = experiment_cfg.get("paths", {})
    tasks_path = args.tasks or ROOT / merged.get("tasks", paths.get("tasks", "data/experiment_tasks.jsonl"))
    out_path = args.output or ROOT / merged.get(
        "generated_file", merged.get("output", "data/baseline_cots.jsonl")
    )
    samples_out_path = sample_detail_path(out_path)
    prompt_path = ROOT / merged.get("prompt", "prompts/baseline_generation.txt")
    n_samples = int(merged.get("verification_samples", merged.get("max_retries", 5)))
    majority = n_samples // 2 + 1  # ceil(n_samples / 2): 3 of 5
    min_steps = int(merged.get("min_steps", 3))
    # Runaway/loop guard threshold: samples with more than this many steps (or an
    # internal repetition loop) are excluded as baseline candidates so the
    # max(step_count) selection never picks a degenerate-but-correct sample.
    runaway_max_steps = int(merged.get("runaway_max_steps", DEFAULT_RUNAWAY_MAX_STEPS))
    base_seed = (
        int(args.base_seed)
        if args.base_seed is not None
        else int(experiment_cfg.get("experiment", {}).get("seed", 42))
    )

    model_cfg = dict(experiment_cfg["model"])
    judge_cfg = dict(experiment_cfg.get("judge", {}))
    script4_cfg = experiment_cfg.get("script4", {})
    math_tolerance = float(script4_cfg.get("math_tolerance", 1e-4))
    science_tolerance = float(script4_cfg.get("science_tolerance", 1e-2))
    models = resolve_models(merged, experiment_cfg, args.model)
    # vLLM loads a SINGLE model per process. The backend is created once below,
    # so it MUST use the model we actually run — not whatever model.name happens
    # to be in experiment.yaml (e.g. the 32B default). When exactly one model is
    # resolved (the normal case, incl. --model), make it authoritative for the
    # loaded weights so `--model 7B` truly loads the 7B checkpoint.
    if len(models) == 1:
        model_cfg["name"] = models[0]

    if not tasks_path.exists():
        print(f"Missing task pool: {tasks_path}", file=sys.stderr)
        print("Run: python scripts/prepare_tasks.py", file=sys.stderr)
        sys.exit(1)

    prompt_template = load_prompt_template(prompt_path)
    tasks = load_tasks(tasks_path, args.domain)
    if args.limit:
        tasks = tasks[: args.limit]

    verified_keys = load_verified_keys(out_path)
    dedupe_baseline_file(out_path)
    backend = create_backend(model_cfg, dry_run=args.dry_run)

    total_work = len(tasks) * len(models)
    skipped = sum(
        1
        for task in tasks
        for model_name in models
        if f"{task['task_id']}|{model_name}" in verified_keys
    )

    print(f"Tasks: {len(tasks)} | Models: {len(models)} | Work items: {total_work}")
    print(f"Already verified (skipped): {skipped} | Remaining: {total_work - skipped}")
    print(
        f"Output: {out_path} | verification_samples={n_samples} "
        f"(majority>={majority}) | min_steps={min_steps}"
    )
    print(f"Sample details: {samples_out_path}")
    print(f"Backend: {'dry-run' if args.dry_run else model_cfg.get('backend', 'vllm')}")

    done = 0
    verified_count = 0
    excluded_count = 0
    exclude_reason_counts: dict[str, int] = {}
    seg_counts: Counter[str] = Counter()
    verified_seg_counts: Counter[str] = Counter()
    excluded_seg_counts: Counter[str] = Counter()
    verified_step_counts: Counter[str] = Counter()
    remaining_work = total_work - skipped
    run_start = time.time()

    def print_progress() -> None:
        if done == 0:
            return
        elapsed = time.time() - run_start
        avg = elapsed / done
        left = max(0, remaining_work - done)
        eta = avg * left
        print(
            f"    progress: {done}/{remaining_work} | elapsed {_fmt_duration(elapsed)} "
            f"| avg {avg:.0f}s/task | ETA {_fmt_duration(eta)}"
        )

    for model_name in models:
        # Overlay Card-recommended baseline sampling (Recum-konform, Option A):
        # Gemma temp=1.0 top_k=64, GLM top_k=40, Ministral temp=1.0, DeepSeek
        # 0.6/0.95. Continuation-mode (Script 3b) keeps uniform 0.6/0.95.
        run_model_cfg = apply_baseline_sampling({**model_cfg, "name": model_name}, model_name)
        family = resolve_model_family(model_name)
        if family.baseline_prompt:
            prompt_path = ROOT / family.baseline_prompt
        else:
            prompt_path = ROOT / merged.get("prompt", "prompts/baseline_generation.txt")
        prompt_template = load_prompt_template(prompt_path)
        print(
            f"  [family={family.name} slug={family.slug}] baseline sampling: "
            f"temp={run_model_cfg.get('temperature')} top_p={run_model_cfg.get('top_p')} "
            f"top_k={run_model_cfg.get('top_k')} "
            f"chat_template={family.baseline_use_chat_template} "
            f"prompt={prompt_path.relative_to(ROOT)}"
        )
        for task in tasks:
            key = f"{task['task_id']}|{model_name}"
            if key in verified_keys:
                continue

            gold = str(task["gold_answer"])
            eval_type = task.get("eval_type", "exact")
            question = str(task.get("prompt", task.get("question", "")))

            # Majority-vote solvability filter: draw a FIXED n_samples runs (no early stop).
            # Each sample's vote counts ONLY answer correctness; the min_steps requirement is
            # decoupled from the vote and applied later to the chosen baseline.
            samples: list[dict] = []
            n_correct = 0
            for attempt in range(n_samples):
                seed = base_seed + attempt
                try:
                    baseline_cot, steps, seg_method, raw_output = generate_baseline(
                        backend,
                        prompt_template,
                        task,
                        run_model_cfg,
                        min_steps,
                        dry_run=args.dry_run,
                        domain=task.get("domain", "math"),
                        eval_type=eval_type,
                        seed=seed,
                        family=family,
                    )
                except Exception as exc:
                    # An errored run is a No vote, but never aborts the loop.
                    print(f"  [ERROR] {key} sample {attempt + 1}: {exc}")
                    samples.append(
                        {
                            "correct": False,
                            "step_count": 0,
                            "baseline_cot": "",
                            "steps": [],
                            "parsed_answer": None,
                            "seed": seed,
                            "segmentation_method": "prompt",
                            "degenerate": False,
                            "degenerate_reason": None,
                            "error": str(exc),
                        }
                    )
                    continue

                # Re-extract from the formatted baseline_cot. After the domain-aware
                # fix in generate_baseline(), baseline_cot ends with
                # "Final answer: <clean_answer>" (no \boxed{} wrapper), so
                # extract_domain_answer here is belt-and-suspenders correctness.
                parsed = extract_domain_answer(baseline_cot, task.get("domain", "math"), eval_type)
                if parsed is None:
                    parsed = extract_final_answer(baseline_cot)
                grading = grade_sample(
                    question=question,
                    gold_answer=gold,
                    answer_content=baseline_cot,
                    extracted_answer=parsed,
                    domain=task.get("domain", "math"),
                    eval_type=eval_type,
                    judge_cfg=judge_cfg,
                    math_tolerance=math_tolerance,
                    science_tolerance=science_tolerance,
                    dry_run=args.dry_run,
                )
                judge_raw_path = write_judge_raw_log(
                    judge_cfg=judge_cfg,
                    phase="baseline",
                    task_id=task["task_id"],
                    grading=grading,
                    dry_run=args.dry_run,
                    sample_index=attempt + 1,
                    seed=seed,
                    extracted_answer=parsed,
                    gold_answer=gold,
                )
                correct = bool(grading["correct"])
                # Math sanity gate: the LLM judge stays the primary verdict, but a
                # judge=Correct / det=Wrong verdict must not survive when the
                # extracted symbolic answer is off by an order of magnitude
                # (e.g. counting_159: final 800 vs gold 28800). Equivalent or
                # non-numeric answers are left to the judge (gross_numeric_mismatch
                # returns False). Applied before the vote and selection so a
                # grossly-wrong trace can neither verify nor be chosen for injection.
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
                    correct = False
                if correct:
                    n_correct += 1
                is_degenerate, degenerate_reason = detect_degenerate_trace(
                    steps, max_steps=runaway_max_steps, raw_text=raw_output
                )
                samples.append(
                    {
                        "correct": correct,
                        "verifier_score": grading["verifier_score"],
                        "judge_output": grading["judge_output"],
                        "judge_raw_output": grading.get("judge_raw_output", ""),
                        "det_correct": grading["det_correct"],
                        "judge_fallback": grading["judge_fallback"],
                        "judge_raw_path": judge_raw_path,
                        "step_count": len(steps),
                        "baseline_cot": baseline_cot,
                        "steps": steps,
                        "parsed_answer": parsed,
                        "seed": seed,
                        "segmentation_method": seg_method,
                        "degenerate": is_degenerate,
                        "degenerate_reason": degenerate_reason,
                        "judge_det_conflict": judge_det_conflict,
                        "sanity_gate_flipped": sanity_gate_flipped,
                        "raw_output": raw_output,
                    }
                )

            solve_rate = n_correct / n_samples if n_samples else 0.0
            correct_samples = [s for s in samples if s["correct"]]

            # Runaway/loop guard: exclude degenerate samples (step explosion or
            # generation loop) as baseline candidates so max(step_count) never
            # picks one. The solvability vote (n_correct/majority) is untouched.
            # Fallback: if ALL correct samples are degenerate, fall back to the
            # full correct set (longest-correct behaviour) and flag the record.
            non_degenerate_correct = [s for s in correct_samples if not s.get("degenerate")]
            baseline_pool = non_degenerate_correct or correct_samples
            degenerate_fallback = bool(correct_samples) and not non_degenerate_correct

            # Baseline = correct run with the most steps; step requirement applied only here.
            verified = False
            exclude_reason: str | None = None
            if n_correct >= majority and correct_samples:
                chosen = max(baseline_pool, key=lambda s: s["step_count"])
                if chosen["step_count"] >= min_steps:
                    verified = True
                else:
                    exclude_reason = "insufficient_steps"
            else:
                # Diagnostic baseline: longest non-degenerate correct run if any, else last sample.
                chosen = (
                    max(baseline_pool, key=lambda s: s["step_count"])
                    if baseline_pool
                    else (samples[-1] if samples else {})
                )
                exclude_reason = "below_majority"

            record = {
                "task_id": task["task_id"],
                "domain": task.get("domain"),
                "eval_type": eval_type,
                "model": model_name,
                "baseline_cot": chosen.get("baseline_cot", ""),
                "steps": chosen.get("steps", []),
                "step_count": chosen.get("step_count", 0),
                "gold_answer": gold,
                "parsed_answer": chosen.get("parsed_answer"),
                "correct": bool(chosen.get("correct")) if verified else False,
                "verified": verified,
                "solve_rate": solve_rate,
                "n_verification_samples": n_samples,
                "seed": chosen.get("seed"),
                "segmentation_method": chosen.get("segmentation_method", "prompt"),
                "sample_step_counts": [s.get("step_count", 0) for s in samples],
                "sample_segmentation_methods": [
                    s.get("segmentation_method", "prompt") for s in samples
                ],
                "sample_correct": [bool(s.get("correct")) for s in samples],
                "sample_seeds": [s.get("seed") for s in samples],
                "sample_degenerate": [bool(s.get("degenerate")) for s in samples],
                "sample_degenerate_reasons": [
                    s.get("degenerate_reason") for s in samples
                ],
                "baseline_degenerate_fallback": degenerate_fallback,
                "sample_judge_det_conflict": [
                    bool(s.get("judge_det_conflict")) for s in samples
                ],
                "sample_sanity_gate_flipped": [
                    bool(s.get("sanity_gate_flipped")) for s in samples
                ],
            }
            if not verified:
                record["exclude_reason"] = exclude_reason

            append_record(out_path, record)
            sample_rows = []
            for attempt_idx, sample in enumerate(samples, start=1):
                sample_rows.append(
                    {
                        "task_id": task["task_id"],
                        "domain": task.get("domain"),
                        "eval_type": eval_type,
                        "model": model_name,
                        "attempt": attempt_idx,
                        "seed": sample.get("seed"),
                        "correct": bool(sample.get("correct")),
                        "verifier_score": sample.get("verifier_score"),
                        "judge_output": sample.get("judge_output"),
                        "judge_raw_path": sample.get("judge_raw_path"),
                        "det_correct": sample.get("det_correct"),
                        "judge_fallback": sample.get("judge_fallback"),
                        "step_count": sample.get("step_count", 0),
                        "segmentation_method": sample.get("segmentation_method", "prompt"),
                        "parsed_answer": sample.get("parsed_answer"),
                        "gold_answer": gold,
                        "baseline_cot": sample.get("baseline_cot", ""),
                        "steps": sample.get("steps", []),
                        "degenerate": bool(sample.get("degenerate")),
                        "degenerate_reason": sample.get("degenerate_reason"),
                        "judge_det_conflict": bool(sample.get("judge_det_conflict")),
                        "sanity_gate_flipped": bool(sample.get("sanity_gate_flipped")),
                        "error": sample.get("error"),
                        "raw_output": sample.get("raw_output"),
                    }
                )
            replace_sample_records(
                samples_out_path,
                task_id=task["task_id"],
                model=model_name,
                records=sample_rows,
            )
            done += 1
            short_model = model_name.split("/")[-1]
            method = str(record["segmentation_method"])
            seg_counts[method] += 1
            # [OK]  = verified: both filters passed (majority vote on answer correctness,
            #         then chosen baseline step_count >= min_steps).
            # [FAIL] = verified: false — below_majority or insufficient_steps.
            if verified:
                verified_keys.add(key)
                verified_count += 1
                verified_seg_counts[method] += 1
                verified_step_counts[str(record["step_count"])] += 1
                print(
                    f"[OK] {task['task_id']:20s} | {short_model:30s} | "
                    f"solve_rate={solve_rate:.2f} ({n_correct}/{n_samples}) | "
                    f"steps={record['step_count']} | seg={record['segmentation_method']}"
                )
            else:
                excluded_count += 1
                excluded_seg_counts[method] += 1
                exclude_reason_counts[exclude_reason] = (
                    exclude_reason_counts.get(exclude_reason, 0) + 1
                )
                print(
                    f"[FAIL] {task['task_id']:20s} | {short_model:30s} | "
                    f"solve_rate={solve_rate:.2f} ({n_correct}/{n_samples}) | "
                    f"steps={record['step_count']} | seg={record['segmentation_method']} | "
                    f"exclude_reason={exclude_reason}"
                )
            print_progress()

    print(f"\nDone. Processed: {done} | Verified: {verified_count} | Excluded: {excluded_count}")
    print(f"Segmentation methods (chosen baselines): {_fmt_counts(seg_counts)}")
    print(f"Segmentation methods (verified): {_fmt_counts(verified_seg_counts)}")
    print(f"Segmentation methods (excluded): {_fmt_counts(excluded_seg_counts)}")
    print(f"Verified step_count distribution: {_fmt_counts(verified_step_counts)}")
    if exclude_reason_counts:
        print("Excluded by reason:", ", ".join(f"{k}={v}" for k, v in sorted(exclude_reason_counts.items())))
    if excluded_count:
        print(
            f"Thesis note: {excluded_count} task(s) excluded — majority of {n_samples} "
            f"samples not correct (>= {majority}) or chosen baseline too short"
        )
    print(f"Output: {out_path}")
    print(f"Sample details: {samples_out_path}")
    print("Post-run checks:")
    print(f"  jq -r 'select(.verified==true) | .segmentation_method' {out_path} | sort | uniq -c")
    print(f"  jq -r 'select(.verified==false) | .exclude_reason' {out_path} | sort | uniq -c")
    print(f"  jq -r 'select(.verified==true) | .step_count' {out_path} | sort -n | uniq -c")


if __name__ == "__main__":
    main()
