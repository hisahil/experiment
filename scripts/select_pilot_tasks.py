#!/usr/bin/env python3
"""
Select the exact GPU-pilot task subset (math20 / science8 / logic8) so that
Script 1 only generates baselines for the tasks the pilot actually uses.

This replicates experiment_scope.select_tasks() with the pilot_gpu scope and
experiment seed, then writes the chosen full task records to an output file.
Because run_perturbed.py restricts to tasks that HAVE a verified baseline and
re-applies select_tasks with the same per-domain caps, generating baselines for
exactly this subset makes the pilot use precisely these tasks.

Usage:
  python scripts/select_pilot_tasks.py \
    --config configs/experiment.yaml \
    --scope pilot_gpu \
    --tasks data/experiment_tasks_math500.jsonl \
    --output data/pilot_gpu_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_scope import get_scope, select_tasks  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_tasks(path: Path) -> list[dict]:
    tasks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Select GPU-pilot task subset.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--scope", type=str, default="pilot_gpu")
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "data/pilot_gpu_tasks.jsonl")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    scope = get_scope(cfg, args.scope)
    experiment_seed = int(cfg.get("experiment", {}).get("seed", 42))

    tasks_path = args.tasks or (ROOT / scope.get("tasks", "data/experiment_tasks_math500.jsonl"))
    if not tasks_path.exists():
        print(f"Missing task pool: {tasks_path}", file=sys.stderr)
        sys.exit(1)

    all_tasks = load_tasks(tasks_path)
    selected = select_tasks(all_tasks, scope, experiment_seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for task in selected:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    by_domain: dict[str, int] = {}
    for t in selected:
        by_domain[t["domain"]] = by_domain.get(t["domain"], 0) + 1

    print(f"Scope: {scope.get('name')} | seed: {experiment_seed}")
    print(f"Selected {len(selected)} tasks -> {args.output}")
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
