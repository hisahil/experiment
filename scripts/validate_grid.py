#!/usr/bin/env python3
"""Validate pipeline JSONL grids against the active experiment scope."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grid_validation import (  # noqa: E402
    format_validation_report,
    load_jsonl,
    resolve_eligible_tasks,
    validate_baselines_grid,
    validate_injections_grid,
    validate_runs_grid,
)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_paths(cfg: dict) -> tuple[Path, Path, Path]:
    script3 = cfg.get("script3", {})
    baseline = cfg.get("baseline", {})
    injection = cfg.get("injection", {})
    script4 = cfg.get("script4", {})
    paths = cfg.get("paths", {})
    baselines_path = ROOT / baseline.get("generated_file", "data/baseline_cots.jsonl")
    injections_path = ROOT / injection.get(
        "generated_file", paths.get("injections", "data/injections_generated.jsonl")
    )
    runs_path = ROOT / script4.get(
        "input_path", script3.get("output_file", "data/post_injection_runs.jsonl")
    )
    return baselines_path, injections_path, runs_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate experiment JSONL grids for a scope.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--scope", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--balanced-per-domain",
        type=int,
        default=None,
        help="Validate against the balanced N-tasks-per-domain selection used by the "
        "GPU run (e.g. 5) instead of the full eligible pool.",
    )
    parser.add_argument(
        "--kind",
        choices=["injections", "runs", "baselines", "all"],
        default="all",
    )
    parser.add_argument("--input", type=Path, default=None, help="Override input for single-kind checks")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    baselines_path, injections_path, runs_path = resolve_paths(cfg)
    scope, tasks, axes, model = resolve_eligible_tasks(
        cfg,
        args.scope,
        domain=args.domain,
        limit=args.limit,
        baseline_path=baselines_path,
        model_name=cfg.get("model", {}).get("name"),
        balanced_per_domain=args.balanced_per_domain,
    )

    print(f"Scope: {scope.get('name')} | tasks: {len(tasks)} | model: {model}")
    results = []
    failed = False

    if args.kind in {"baselines", "all"}:
        path = args.input if args.kind == "baselines" and args.input else baselines_path
        if not path.exists():
            print(f"[baselines] missing file: {path}", file=sys.stderr)
            failed = True
        else:
            result = validate_baselines_grid(load_jsonl(path), tasks, model=model)
            print(format_validation_report(result))
            results.append(result)
            failed = failed or not result.ok

    if args.kind in {"injections", "all"}:
        path = args.input if args.kind == "injections" and args.input else injections_path
        if not path.exists():
            print(f"[injections] missing file: {path}", file=sys.stderr)
            failed = True
        else:
            result = validate_injections_grid(
                load_jsonl(path),
                tasks,
                axes,
                target_model=model,
                pregenerate_steps=(cfg.get("injection") or {}).get(
                    "pregenerate_steps", "auto"
                ),
                dialogue_mode=cfg.get("script3", {}).get(
                    "dialogue_mode", "three_turn"
                ),
            )
            print(format_validation_report(result))
            results.append(result)
            failed = failed or not result.ok

    if args.kind in {"runs", "all"}:
        path = args.input if args.kind == "runs" and args.input else runs_path
        if not path.exists():
            print(f"[runs] missing file: {path}", file=sys.stderr)
            failed = True
        else:
            script3_variants = cfg.get("script3", {}).get("variants", ["stop", "recovery"])
            result = validate_runs_grid(
                load_jsonl(path),
                tasks,
                axes,
                model=model,
                variants=script3_variants,
            )
            print(format_validation_report(result))
            results.append(result)
            failed = failed or not result.ok

    if failed:
        sys.exit(1)
    if results and all(r.ok for r in results):
        print("Grid validation passed.")


if __name__ == "__main__":
    main()
