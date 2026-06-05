#!/usr/bin/env python3
"""
Script 1: Generate verified baseline CoTs per task per experiment model.

Output: data/baseline_cots.jsonl (step_count used by Script 3 for midpoint injection).

Usage:
  python scripts/generate_baselines.py --dry-run
  python scripts/generate_baselines.py --domain math --limit 5
  python scripts/generate_baselines.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

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


def resolve_models(cfg: dict, experiment_cfg: dict, cli_model: str | None) -> list[str]:
    if cli_model:
        return [cli_model]
    baseline_cfg = cfg.get("models") or experiment_cfg.get("baseline", {}).get("models")
    if baseline_cfg:
        return list(baseline_cfg)
    return [experiment_cfg["model"]["name"]]


def _dry_run_pad_output(raw_text: str, gold_answer: str, min_steps: int) -> str:
    """Dry-run backend returns few steps; pad for pipeline smoke tests."""
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
) -> tuple[str, list[str], str]:
    prompt = render_prompt(prompt_template, task=task["prompt"])
    result = backend.generate(
        prompt=prompt,
        max_new_tokens=int(model_cfg.get("max_new_tokens", 2048)),
        temperature=float(model_cfg.get("temperature", 0.6)),
        top_p=float(model_cfg.get("top_p", 0.95)),
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
    prompt_path = ROOT / merged.get("prompt", "prompts/baseline_generation.txt")
    max_retries = int(merged.get("max_retries", 5))
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
    print(f"Output: {out_path} | max_retries={max_retries} | min_steps={min_steps}")
    print(f"Backend: {'dry-run' if args.dry_run else model_cfg.get('backend', 'vllm')}")

    done = 0
    verified_count = 0
    excluded_count = 0

    for model_name in models:
        run_model_cfg = {**model_cfg, "name": model_name}
        for task in tasks:
            key = f"{task['task_id']}|{model_name}"
            if key in verified_keys:
                continue

            gold = str(task["gold_answer"])
            eval_type = task.get("eval_type", "exact")
            success = False
            last_record: dict | None = None

            for attempt in range(max_retries):
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
                    )
                except Exception as exc:
                    print(f"  [ERROR] {key} attempt {attempt + 1}: {exc}")
                    last_record = {
                        "task_id": task["task_id"],
                        "domain": task.get("domain"),
                        "eval_type": eval_type,
                        "model": model_name,
                        "baseline_cot": "",
                        "steps": [],
                        "step_count": 0,
                        "gold_answer": gold,
                        "parsed_answer": None,
                        "verified": False,
                        "retries": attempt + 1,
                        "seed": seed,
                        "segmentation_method": "prompt",
                        "error": str(exc),
                    }
                    continue

                # Re-extract from the formatted baseline_cot. After the domain-aware
                # fix in generate_baseline(), baseline_cot ends with
                # "Final answer: <clean_answer>" (no \boxed{} wrapper), so
                # extract_domain_answer here is belt-and-suspenders correctness.
                parsed = extract_domain_answer(baseline_cot, task.get("domain", "math"), eval_type)
                if parsed is None:
                    parsed = extract_final_answer(baseline_cot)
                correct = is_correct(parsed, gold, eval_type)
                step_count = len(steps)
                verified = bool(correct) and step_count >= min_steps

                last_record = {
                    "task_id": task["task_id"],
                    "domain": task.get("domain"),
                    "eval_type": eval_type,
                    "model": model_name,
                    "baseline_cot": baseline_cot,
                    "steps": steps,
                    "step_count": step_count,
                    "gold_answer": gold,
                    "parsed_answer": parsed,
                    "verified": verified,
                    "retries": attempt + 1,
                    "seed": seed,
                    "segmentation_method": seg_method,
                }

                if verified:
                    append_record(out_path, last_record)
                    verified_keys.add(key)
                    verified_count += 1
                    success = True
                    done += 1
                    status = "OK"
                    print(
                        f"[{status}] {task['task_id']:20s} | {model_name.split('/')[-1]:30s} | "
                        f"steps={step_count} | seg={seg_method} | retry={attempt + 1}"
                    )
                    break

            if not success and last_record is not None:
                append_record(out_path, last_record)
                excluded_count += 1
                done += 1
                print(
                    f"[FAIL] {task['task_id']:20s} | {model_name.split('/')[-1]:30s} | "
                    f"verified=false after {max_retries} retries (excluded from Script 3)"
                )

    print(f"\nDone. Processed: {done} | Verified: {verified_count} | Excluded: {excluded_count}")
    if excluded_count:
        print(
            f"Thesis note: {excluded_count} task(s) excluded because no verified correct "
            f"baseline within {max_retries} attempts"
        )
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
