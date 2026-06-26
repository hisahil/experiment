#!/usr/bin/env python3
"""Local pre-GPU checks — run on Mac before HyperStack deploy."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_index import index_verified_baselines  # noqa: E402
from experiment_scope import (  # noqa: E402
    count_planned_dialogues,
    design_lists,
    get_scope,
    select_tasks_balanced,
)
from grid_validation import load_jsonl, preflight_injections  # noqa: E402
from run_perturbed import index_injections, load_tasks, report_injection_preflight  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/experiment.yaml"
REQUIRED_DATA = (
    "data/baseline_cots.jsonl",
    "data/injections_generated.jsonl",
    "data/experiment_tasks_pilot_gpu.jsonl",
)
GENERATED_DATA = (
    "data/baseline_cots.jsonl",
    "data/injections_generated.jsonl",
)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_env() -> list[str]:
    errors: list[str] = []
    env_path = ROOT / ".env"
    if not env_path.exists():
        errors.append("Missing .env — create with ANTHROPIC_API_KEY=...")
        return errors
    from dotenv import load_dotenv

    load_dotenv(env_path)
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        errors.append(".env exists but ANTHROPIC_API_KEY is empty")
    return errors


def check_data_files() -> tuple[list[str], list[str], bool]:
    """Return (errors, notes, baselines_pending). baselines_pending=True if Script 1+2 outputs missing."""
    errors: list[str] = []
    notes: list[str] = []
    baselines_pending = False
    for rel in REQUIRED_DATA:
        path = ROOT / rel
        if not path.exists():
            if rel in GENERATED_DATA:
                baselines_pending = True
                notes.append(f"{rel} missing — run_gpu_baseline_prep.sh on GPU will create it")
            else:
                errors.append(f"Missing {rel}")
        elif path.stat().st_size == 0:
            if rel in GENERATED_DATA:
                baselines_pending = True
                notes.append(f"{rel} empty — regenerate on GPU via run_gpu_baseline_prep.sh")
            else:
                errors.append(f"Empty file: {rel}")
    return errors, notes, baselines_pending


def check_scope_readiness(
    cfg: dict,
    *,
    scope_name: str,
    balanced_per_domain: int,
) -> tuple[list[str], list[str], dict]:
    """Return (errors, warnings, summary dict)."""
    errors: list[str] = []
    warnings: list[str] = []
    scope = get_scope(cfg, scope_name)
    script3 = cfg.get("script3", {})
    model_name = cfg.get("model", {}).get("name", "")
    experiment_seed = int(cfg.get("experiment", {}).get("seed", 42))
    lists = design_lists(cfg, scope)

    tasks_path = ROOT / (scope.get("tasks") or cfg.get("paths", {}).get("tasks", "data/experiment_tasks.jsonl"))
    baseline_path = ROOT / script3.get("baseline_file", "data/baseline_cots.jsonl")
    injections_path = ROOT / script3.get("injections_file", "data/injections_generated.jsonl")

    all_tasks = load_tasks(tasks_path, None)
    task_by_id = {t["task_id"]: t for t in all_tasks}
    baseline_index = index_verified_baselines(load_jsonl(baseline_path), model_name)
    eligible = [task_by_id[tid] for tid in sorted(baseline_index.keys()) if tid in task_by_id]
    tasks = select_tasks_balanced(eligible, scope, experiment_seed, balanced_per_domain)

    from collections import Counter

    by_domain = Counter(t["domain"] for t in tasks)
    for domain in scope.get("domains", []):
        if by_domain.get(domain, 0) < balanced_per_domain:
            errors.append(
                f"Only {by_domain.get(domain, 0)} tasks for domain '{domain}' "
                f"(need {balanced_per_domain} with verified baseline)"
            )

    missing_baseline = [t["task_id"] for t in tasks if t["task_id"] not in baseline_index]
    if missing_baseline:
        errors.append(f"{len(missing_baseline)} selected tasks lack verified baseline")

    injection_index = index_injections(load_jsonl(injections_path), target_model=model_name)
    missing, invalid, mismatches = preflight_injections(
        tasks, lists, injection_index, baseline_index,
        pregenerate_steps=(cfg.get("injection") or {}).get("pregenerate_steps", "auto"),
        dialogue_mode=script3.get("dialogue_mode", "three_turn"),
    )
    if report_injection_preflight(missing, invalid, mismatches, strict_insertion=False):
        errors.append(f"Injection preflight failed: {len(missing)} missing, {len(invalid)} invalid")

    planned = count_planned_dialogues(
        tasks,
        lists,
        include_neutral=bool(scope.get("include_neutral_control")),
    )
    summary = {
        "scope": scope_name,
        "tasks": len(tasks),
        "by_domain": dict(by_domain),
        "planned_dialogues": planned,
        "k": lists["k_injections"],
        "length_levels": lists["length_levels"],
        "seeds": lists["seeds"],
        "missing_injections": len(missing),
        "invalid_injections": len(invalid),
        "insertion_mismatches": len(mismatches),
        "task_ids": [t["task_id"] for t in tasks],
    }
    if mismatches:
        warnings.append(f"{len(mismatches)} injection insertion_step mismatches (warn only)")
    return errors, warnings, summary


def run_pytest(quick: bool) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    if quick:
        cmd.extend(["--tb=no", "-x"])
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-GPU readiness checks (run locally on Mac).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scope", type=str, default="pilot_gpu_full_factorial")
    parser.add_argument("--balanced-per-domain", type=int, default=5)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    all_errors: list[str] = []
    all_warnings: list[str] = []

    print("=== GPU Preflight (local) ===\n")

    print("[1/4] Data files")
    data_errors, data_notes, baselines_pending = check_data_files()
    for note in data_notes:
        print(f"  INFO: {note}")
    for err in data_errors:
        all_errors.append(err)
        print(f"  FAIL: {err}")
    if not data_errors and not baselines_pending:
        for rel in REQUIRED_DATA:
            size_mb = (ROOT / rel).stat().st_size / 1e6
            print(f"  OK   {rel} ({size_mb:.1f} MB)")
    elif not data_errors:
        tasks_mb = (ROOT / "data/experiment_tasks_pilot_gpu.jsonl").stat().st_size / 1e6
        print(f"  OK   data/experiment_tasks_pilot_gpu.jsonl ({tasks_mb:.1f} MB)")

    print("\n[2/4] Environment")
    env_errors = check_env()
    for err in env_errors:
        all_errors.append(err)
        print(f"  FAIL: {err}")
    if not env_errors:
        print("  OK   ANTHROPIC_API_KEY set in .env")

    print(f"\n[3/4] Scope readiness ({args.scope}, {args.balanced_per_domain}/domain)")
    if baselines_pending:
        print("  SKIP baseline/injections not generated yet (expected after clean_for_gpu_run.sh --full)")
        print("       On GPU: bash scripts/run_gpu_baseline_prep.sh before run_gpu_pilot.sh")
    else:
        scope_errors, scope_warnings, summary = check_scope_readiness(
            cfg,
            scope_name=args.scope,
            balanced_per_domain=args.balanced_per_domain,
        )
        all_errors.extend(scope_errors)
        all_warnings.extend(scope_warnings)
        if scope_errors:
            for err in scope_errors:
                print(f"  FAIL: {err}")
        else:
            print(f"  OK   {summary['tasks']} tasks | {summary['planned_dialogues']} dialogues planned")
            for domain, count in sorted(summary["by_domain"].items()):
                print(f"       {domain}: {count} tasks")
            print(f"       k={summary['k']} | length={summary['length_levels']} | seeds={len(summary['seeds'])}")

    if not args.skip_tests:
        print("\n[4/4] Tests")
        ok, out = run_pytest(quick=False)
        if ok:
            print(f"  OK   {out.splitlines()[-1] if out else 'passed'}")
        else:
            all_errors.append("pytest failed")
            print("  FAIL: pytest")
            print(out[-2000:])

    print("\n=== Summary ===")
    if all_warnings:
        for w in all_warnings:
            print(f"  WARN: {w}")
    if all_errors:
        print(f"  NOT READY — {len(all_errors)} error(s). Fix before GPU deploy.")
        print("\n  Likely fix (on GPU after setup_gpu.sh):")
        print("    bash scripts/run_gpu_baseline_prep.sh")
        print("  Fallback (math-only timing, 5 tasks):")
        print("    bash scripts/run_gpu_pilot_math_only.sh")
        sys.exit(1)

    if baselines_pending:
        print("  READY for deploy — generate baselines on GPU first.")
        print("  GPU: bash scripts/run_gpu_full.sh")
    else:
        print("  READY for HyperStack deploy.")
    print("\nNext steps:")
    print("  1. Deploy HyperStack A100, note IP")
    print("  2. rsync (see scripts/transfer_to_gpu.sh.example)")
    print("  3. SSH → source .venv/bin/activate")
    if baselines_pending:
        print("  4. SSH → bash scripts/run_gpu_full.sh")
    else:
        print("  4. SSH → bash scripts/run_gpu_full.sh")


if __name__ == "__main__":
    main()
