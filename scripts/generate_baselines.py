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

from cot_segmentation import segment_cot, steps_to_baseline_cot  # noqa: E402
from evaluator import extract_domain_answer, extract_final_answer, is_correct  # noqa: E402
from llm_backend import create_backend  # noqa: E402

DEFAULT_BASELINE_CONFIG = ROOT / "configs/baseline_generation.yaml"
DEFAULT_EXPERIMENT_CONFIG = ROOT / "configs/experiment.yaml"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs: str) -> str:
    return template.format(**kwargs)


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


def append_record(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
) -> tuple[str, list[str], str]:
    prompt = render_prompt(prompt_template, task=task["prompt"])
    result = backend.generate(
        prompt=prompt,
        max_new_tokens=int(model_cfg.get("max_new_tokens", 2048)),
        temperature=float(model_cfg.get("temperature", 0.6)),
        top_p=float(model_cfg.get("top_p", 0.95)),
        seed=seed,
    )
    raw_text = result.text
    if dry_run:
        raw_text = _dry_run_pad_output(raw_text, str(task["gold_answer"]), min_steps)

    steps, method = segment_cot(
        raw_text,
        min_steps=min_steps,
        backend=backend,
        model_cfg=model_cfg,
    )
    # Use domain-aware extraction so math \boxed{} answers are stripped of their
    # wrapper (extract_final_answer would leave the \boxed{} intact, making
    # is_correct fail for symbolic MATH-500 answers like \frac{3}{4}).
    parsed = extract_domain_answer(raw_text, domain, eval_type)
    if parsed is None and dry_run:
        parsed = str(task["gold_answer"])
    baseline_cot = steps_to_baseline_cot(steps, parsed)
    return baseline_cot, steps, method


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
    # Fixed number of majority-vote samples per task; falls back to legacy max_retries.
    n_samples = int(merged.get("verification_samples", merged.get("max_retries", 5)))
    majority = n_samples // 2 + 1  # ceil(n_samples / 2): 3 of 5
    min_steps = int(merged.get("min_steps", 3))
    base_seed = int(experiment_cfg.get("experiment", {}).get("seed", 42))

    model_cfg = dict(experiment_cfg["model"])
    models = resolve_models(merged, experiment_cfg, args.model)

    if not tasks_path.exists():
        print(f"Missing task pool: {tasks_path}", file=sys.stderr)
        print("Run: python scripts/prepare_tasks.py", file=sys.stderr)
        sys.exit(1)

    prompt_template = load_prompt_template(prompt_path)
    tasks = load_tasks(tasks_path, args.domain)
    if args.limit:
        tasks = tasks[: args.limit]

    verified_keys = load_verified_keys(out_path)
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
        run_model_cfg = {**model_cfg, "name": model_name}
        for task in tasks:
            key = f"{task['task_id']}|{model_name}"
            if key in verified_keys:
                continue

            gold = str(task["gold_answer"])
            eval_type = task.get("eval_type", "exact")

            # Majority-vote solvability filter: draw a FIXED n_samples runs (no early stop).
            # Each sample's vote counts ONLY answer correctness; the min_steps requirement is
            # decoupled from the vote and applied later to the chosen baseline.
            samples: list[dict] = []
            n_correct = 0
            for attempt in range(n_samples):
                seed = base_seed + attempt
                try:
                    baseline_cot, steps, seg_method = generate_baseline(
                        backend,
                        prompt_template,
                        task,
                        run_model_cfg,
                        min_steps,
                        dry_run=args.dry_run,
                        domain=task.get("domain", "math"),
                        eval_type=eval_type,
                        seed=seed,
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
                correct = bool(is_correct(parsed, gold, eval_type))
                if correct:
                    n_correct += 1
                samples.append(
                    {
                        "correct": correct,
                        "step_count": len(steps),
                        "baseline_cot": baseline_cot,
                        "steps": steps,
                        "parsed_answer": parsed,
                        "seed": seed,
                        "segmentation_method": seg_method,
                    }
                )

            solve_rate = n_correct / n_samples if n_samples else 0.0
            correct_samples = [s for s in samples if s["correct"]]

            # Baseline = correct run with the most steps; step requirement applied only here.
            verified = False
            exclude_reason: str | None = None
            if n_correct >= majority and correct_samples:
                chosen = max(correct_samples, key=lambda s: s["step_count"])
                if chosen["step_count"] >= min_steps:
                    verified = True
                else:
                    exclude_reason = "insufficient_steps"
            else:
                # Diagnostic baseline: longest correct run if any, else last sample.
                chosen = (
                    max(correct_samples, key=lambda s: s["step_count"])
                    if correct_samples
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
            }
            if not verified:
                record["exclude_reason"] = exclude_reason

            append_record(out_path, record)
            for attempt_idx, sample in enumerate(samples, start=1):
                append_record(
                    samples_out_path,
                    {
                        "task_id": task["task_id"],
                        "domain": task.get("domain"),
                        "eval_type": eval_type,
                        "model": model_name,
                        "attempt": attempt_idx,
                        "seed": sample.get("seed"),
                        "correct": bool(sample.get("correct")),
                        "step_count": sample.get("step_count", 0),
                        "segmentation_method": sample.get("segmentation_method", "prompt"),
                        "parsed_answer": sample.get("parsed_answer"),
                        "gold_answer": gold,
                        "baseline_cot": sample.get("baseline_cot", ""),
                        "steps": sample.get("steps", []),
                        "error": sample.get("error"),
                    },
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
