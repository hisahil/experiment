#!/usr/bin/env python3
"""
Experiment 1 runner — legacy prompt-template path (smoke / injection_templates).

For Script 3 perturbed CoT (baselines + generated injections + chat dialog):
  python scripts/run_perturbed.py --dry-run --domain math --limit 5
  python scripts/run_perturbed.py --scope pilot --domain math

Usage:
  python scripts/prepare_tasks.py
  python src/run_experiment.py --config configs/experiment.yaml --dry-run
  python src/run_experiment.py --config configs/experiment.yaml --scope pilot
  python src/run_experiment.py --config configs/smoke_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluator import extract_final_answer, is_correct  # noqa: E402
from experiment_scope import (  # noqa: E402
    design_lists,
    get_scope,
    print_run_plan,
    select_tasks,
)
from injection_builder import load_injection_templates, resolve_injection_text  # noqa: E402
from llm_backend import create_backend  # noqa: E402
from trace_builder import build_trace, load_injections  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs: str) -> str:
    return template.format(**kwargs)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_raw_output(outputs_dir: Path, run_id: str, text: str) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / f"{run_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def run_variant(
    backend,
    template: str,
    task: dict,
    external_reasoning: str | None,
    variant: str,
    model_cfg: dict,
) -> dict:
    prompt_kwargs = {"task": task["prompt"]}
    if external_reasoning is not None:
        prompt_kwargs["external_reasoning"] = external_reasoning
    prompt = render_prompt(template, **prompt_kwargs)

    result = backend.generate(
        prompt=prompt,
        max_new_tokens=int(model_cfg.get("max_new_tokens", 2048)),
        temperature=float(model_cfg.get("temperature", 0.6)),
        top_p=float(model_cfg.get("top_p", 0.95)),
    )
    parsed = extract_final_answer(result.text)
    correct = is_correct(parsed, task["gold_answer"], task.get("eval_type", "exact"))
    return {
        "raw_text": result.text,
        "parsed_answer": parsed,
        "correct": correct,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "wall_time_s": round(result.wall_time_s, 3),
        "variant": variant,
    }


def resolve_injection(
    cfg: dict,
    paths: dict,
    error_type: str,
    task: dict,
    rng: random.Random,
) -> str:
    templates_path = paths.get("injection_templates")
    if templates_path:
        templates = load_injection_templates(ROOT / templates_path)
        return resolve_injection_text(templates, error_type, task, rng)
    legacy = load_injections(ROOT / paths["injections"])
    return legacy[error_type][0]


def record_run(
    *,
    experiment_name: str,
    task: dict,
    model_name: str,
    variant: str,
    error_type: str | None,
    k: int,
    length_level: str | None,
    seed: int,
    injected_step_ids: list[int],
    outcome: dict,
    pair_key: str | None,
    logging_cfg: dict,
    outputs_dir: Path,
    runs_file: Path,
) -> dict:
    run_id = str(uuid.uuid4())[:8]
    started_at = datetime.now(timezone.utc).isoformat()
    record = {
        "run_id": run_id,
        "experiment": experiment_name,
        "started_at": started_at,
        "task_id": task["task_id"],
        "domain": task["domain"],
        "model": model_name,
        "variant": variant,
        "error_type": error_type,
        "k": k,
        "length_level": length_level,
        "seed": seed,
        "injected_step_ids": injected_step_ids,
        "parsed_answer": outcome["parsed_answer"],
        "gold_answer": task["gold_answer"],
        "correct": outcome["correct"],
        "prompt_tokens": outcome["prompt_tokens"],
        "completion_tokens": outcome["completion_tokens"],
        "wall_time_s": outcome["wall_time_s"],
        "pair_key": pair_key,
        "source_dataset": task.get("source_dataset"),
    }
    if logging_cfg.get("save_raw_outputs", True):
        raw_path = save_raw_output(outputs_dir, run_id, outcome["raw_text"])
        record["raw_output_path"] = str(raw_path.relative_to(ROOT))
    append_jsonl(runs_file, record)
    return record


def run_smoke_style(cfg: dict, args: argparse.Namespace) -> None:
    """Legacy smoke configs (flat run block, static injections)."""
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    run_cfg = cfg["run"]
    tasks = load_tasks(ROOT / paths["tasks"])
    max_tasks = run_cfg.get("max_tasks")
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]

    injections = load_injections(ROOT / paths["injections"])
    prompt_templates = {
        "baseline": load_prompt_template(ROOT / paths["prompts"]["baseline"]),
        "recovery": load_prompt_template(ROOT / paths["prompts"]["recovery"]),
        "stop": load_prompt_template(ROOT / paths["prompts"]["stop"]),
    }
    backend = create_backend(model_cfg, dry_run=args.dry_run)
    logs_dir = ROOT / paths["logs"]
    runs_file = logs_dir / Path(cfg.get("logging", {}).get("runs_file", "runs.jsonl")).name
    outputs_dir = ROOT / paths["outputs"]
    experiment_name = cfg["experiment"]["name"]
    seed = cfg["experiment"].get("seed", 42)

    print(f"Experiment: {experiment_name} (legacy smoke config)")
    print(f"Tasks: {len(tasks)}")
    print("")

    all_records: list[dict] = []
    variants = run_cfg.get("variants", ["baseline"])
    perturbed = [v for v in variants if v in ("stop", "recovery")]

    for task in tasks:
        if "baseline" in variants:
            outcome = run_variant(
                backend, prompt_templates["baseline"], task, None, "baseline", model_cfg
            )
            record = record_run(
                experiment_name=experiment_name,
                task=task,
                model_name=model_cfg["name"],
                variant="baseline",
                error_type=None,
                k=0,
                length_level=None,
                seed=seed,
                injected_step_ids=[],
                outcome=outcome,
                pair_key=None,
                logging_cfg=cfg.get("logging", {}),
                outputs_dir=outputs_dir,
                runs_file=runs_file,
            )
            all_records.append(record)
            _log_line(task, "baseline", None, outcome)

        for error_type in run_cfg.get("error_types", []):
            injection_text = injections[error_type][0]
            external_trace, injected_step_ids = build_trace(
                task["prompt"],
                injection_text,
                k=run_cfg.get("k_injections", 1),
                length_level=run_cfg.get("length_level", "short"),
            )
            pair_key = f"{task['task_id']}::{error_type}::k{run_cfg.get('k_injections', 1)}"
            stop_record: dict | None = None
            for variant in perturbed:
                outcome = run_variant(
                    backend,
                    prompt_templates[variant],
                    task,
                    external_trace,
                    variant,
                    model_cfg,
                )
                record = record_run(
                    experiment_name=experiment_name,
                    task=task,
                    model_name=model_cfg["name"],
                    variant=variant,
                    error_type=error_type,
                    k=run_cfg.get("k_injections", 1),
                    length_level=run_cfg.get("length_level", "short"),
                    seed=seed,
                    injected_step_ids=injected_step_ids,
                    outcome=outcome,
                    pair_key=pair_key,
                    logging_cfg=cfg.get("logging", {}),
                    outputs_dir=outputs_dir,
                    runs_file=runs_file,
                )
                all_records.append(record)
                if variant == "stop":
                    stop_record = record
                elif variant == "recovery" and stop_record:
                    record["_paired_stop"] = stop_record
                _log_line(task, variant, error_type, outcome)
    _print_summary(all_records)
    if args.dry_run:
        print("\nDry-run complete.")


def run_scoped_experiment(cfg: dict, args: argparse.Namespace) -> None:
    scope = get_scope(cfg, args.scope)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    experiment_seed = int(cfg["experiment"].get("seed", 42))

    tasks_path = ROOT / paths["tasks"]
    if not tasks_path.exists():
        print(f"Missing task pool: {tasks_path}")
        print("Run: python scripts/prepare_tasks.py")
        sys.exit(1)

    tasks = select_tasks(load_tasks(tasks_path), scope, experiment_seed)
    lists = design_lists(cfg, scope)

    prompt_templates = {
        "baseline": load_prompt_template(ROOT / paths["prompts"]["baseline"]),
        "recovery": load_prompt_template(ROOT / paths["prompts"]["recovery"]),
        "stop": load_prompt_template(ROOT / paths["prompts"]["stop"]),
    }
    backend = create_backend(model_cfg, dry_run=args.dry_run)
    logs_dir = ROOT / paths["logs"]
    logging_cfg = cfg.get("logging", {})
    runs_file = logs_dir / Path(logging_cfg.get("runs_file", "experiment_runs.jsonl")).name
    outputs_dir = ROOT / paths["outputs"]
    experiment_name = cfg["experiment"]["name"]

    print_run_plan(scope, tasks, lists, str(runs_file))
    print(f"Backend: {'dry-run' if args.dry_run else model_cfg.get('backend', 'vllm')}")
    print("")

    all_records: list[dict] = []
    variants = lists["variants"]
    perturbed_variants = [v for v in variants if v in ("stop", "recovery")]

    for task in tasks:
        for run_seed in lists["seeds"]:
            inj_rng = random.Random(f"{run_seed}-{task['task_id']}")

            if "baseline" in variants:
                outcome = run_variant(
                    backend, prompt_templates["baseline"], task, None, "baseline", model_cfg
                )
                record = record_run(
                    experiment_name=experiment_name,
                    task=task,
                    model_name=model_cfg["name"],
                    variant="baseline",
                    error_type=None,
                    k=0,
                    length_level=None,
                    seed=run_seed,
                    injected_step_ids=[],
                    outcome=outcome,
                    pair_key=None,
                    logging_cfg=logging_cfg,
                    outputs_dir=outputs_dir,
                    runs_file=runs_file,
                )
                all_records.append(record)
                _log_line(task, "baseline", None, outcome)

            for error_type in lists["error_types"]:
                for k in lists["k_injections"]:
                    for length_level in lists["length_levels"]:
                        injection_text = resolve_injection(cfg, paths, error_type, task, inj_rng)
                        external_trace, injected_step_ids = build_trace(
                            task["prompt"],
                            injection_text,
                            k=k,
                            length_level=length_level,
                        )
                        pair_key = f"{task['task_id']}::{error_type}::k{k}::{length_level}::s{run_seed}"
                        stop_record: dict | None = None

                        for variant in perturbed_variants:
                            outcome = run_variant(
                                backend,
                                prompt_templates[variant],
                                task,
                                external_trace,
                                variant,
                                model_cfg,
                            )
                            record = record_run(
                                experiment_name=experiment_name,
                                task=task,
                                model_name=model_cfg["name"],
                                variant=variant,
                                error_type=error_type,
                                k=k,
                                length_level=length_level,
                                seed=run_seed,
                                injected_step_ids=injected_step_ids,
                                outcome=outcome,
                                pair_key=pair_key,
                                logging_cfg=logging_cfg,
                                outputs_dir=outputs_dir,
                                runs_file=runs_file,
                            )
                            all_records.append(record)
                            if variant == "stop":
                                stop_record = record
                            elif variant == "recovery" and stop_record:
                                record["_paired_stop"] = stop_record
                            _log_line(task, variant, error_type, outcome)

    _print_summary(all_records)
    if args.dry_run:
        print("\nDry-run complete. Scope OK — run on GPU without --dry-run when ready.")


def _log_line(task: dict, variant: str, error_type: str | None, outcome: dict) -> None:
    status = "OK" if outcome["correct"] else "WRONG" if outcome["correct"] is False else "NO_PARSE"
    print(
        f"[{status}] {task['task_id']} | {variant:8} | {error_type or '-':18} | "
        f"tokens={outcome['completion_tokens']} | {outcome['wall_time_s']}s"
    )


def _print_summary(all_records: list[dict]) -> None:
    by_variant: dict[str, list] = {}
    for record in all_records:
        by_variant.setdefault(record["variant"], []).append(record)
    print("\n=== Summary ===")
    for variant, records in by_variant.items():
        parsed = [r for r in records if r["parsed_answer"] is not None]
        correct = [r for r in records if r["correct"] is True]
        avg_tokens = sum(r["completion_tokens"] for r in records) / len(records) if records else 0
        avg_time = sum(r["wall_time_s"] for r in records) / len(records) if records else 0
        print(
            f"{variant:8} | runs={len(records):3} | parsed={len(parsed):3} | "
            f"correct={len(correct):3} | avg_tokens={avg_tokens:.1f} | avg_time={avg_time:.2f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSc thesis Experiment 1.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--scope", type=str, default=None, help="Override active_scope (pilot/full).")
    parser.add_argument("--dry-run", action="store_true", help="Mock backend (no GPU).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if "scopes" in cfg:
        run_scoped_experiment(cfg, args)
    else:
        run_smoke_style(cfg, args)


if __name__ == "__main__":
    main()
