#!/usr/bin/env python3
"""Unified GPU full run: Script 1 → select → Script 2 → preflight → Script 3 → Script 4.

Resume-safe (no auto-clean). Timings written to results/run_timings.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_design import resolve_design_axes  # noqa: E402
from experiment_scope import count_planned_dialogues, get_scope  # noqa: E402
from gpu_run_plan import (  # noqa: E402
    BASELINE_CANDIDATE_LIMITS,
    build_total_extrapolation,
    resolve_selected_tasks,
    write_jsonl,
)
from run_timings import fmt_duration, write_run_timings  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_step(cmd: list[str], *, label: str) -> float:
    print(f"\n{'=' * 60}\n>>> {label}\n{'=' * 60}")
    print(" ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    elapsed = time.time() - t0
    print(f"--- {label} done in {fmt_duration(elapsed)}")
    return elapsed


def count_baseline_tasks_processed(baseline_path: Path, model_name: str) -> int:
    if not baseline_path.exists():
        return 0
    seen: set[str] = set()
    with baseline_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("model") != model_name:
                continue
            seen.add(rec["task_id"])
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified GPU full run (Scripts 1–4, resume-safe).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--scope", type=str, default="pilot_gpu_full_factorial")
    parser.add_argument("--balanced-per-domain", type=int, default=5)
    parser.add_argument("--budget-hours", type=float, default=0.0)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/experiment_tasks_pilot_gpu.jsonl")
    parser.add_argument(
        "--selected-tasks",
        type=Path,
        default=ROOT / "data/selected_pilot_tasks.jsonl",
        help="Written after Script 1; used for selected-only Script 2.",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest in preflight gate.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    scope = get_scope(cfg, args.scope)
    axes = resolve_design_axes(cfg, scope)
    experiment_seed = int(cfg.get("experiment", {}).get("seed", 42))
    model_name = cfg.get("model", {}).get("name", "")
    baseline_path = ROOT / cfg.get("baseline", {}).get("generated_file", "data/baseline_cots.jsonl")
    injections_path = ROOT / cfg.get("injection", {}).get(
        "generated_file", "data/injections_generated.jsonl"
    )
    timings_path = ROOT / cfg.get("script4", {}).get("output_dir", "results") / "run_timings.json"

    run_t0 = time.time()
    timings: dict = {
        "timestamp_start": datetime.now(timezone.utc).isoformat(),
        "script": "run_gpu_full.py",
        "scope": args.scope,
        "model": model_name,
        "balanced_per_domain": args.balanced_per_domain,
        "budget_hours": args.budget_hours,
        "baseline_candidate_limits": dict(BASELINE_CANDIDATE_LIMITS),
    }

    # ── Judge smoke test (fail fast before Script 1) ──────────────────────────
    smoke_cmd = [
        sys.executable,
        str(ROOT / "scripts/smoke_judge.py"),
        "--config",
        str(args.config),
    ]
    run_step(smoke_cmd, label="Judge smoke test")

    # ── Script 1: baselines per domain (candidate pool with buffer) ───────────
    script1_t0 = time.time()
    for domain, limit in BASELINE_CANDIDATE_LIMITS.items():
        run_step(
            [
                sys.executable,
                str(ROOT / "scripts/generate_baselines.py"),
                "--experiment-config",
                str(args.config),
                "--tasks",
                str(args.tasks),
                "--domain",
                domain,
                "--limit",
                str(limit),
            ],
            label=f"Script 1 baselines ({domain}, limit={limit})",
        )
    script1_elapsed = time.time() - script1_t0
    script1_tasks = count_baseline_tasks_processed(baseline_path, model_name)
    timings["script1_elapsed_s"] = round(script1_elapsed, 1)
    timings["script1_tasks_processed"] = script1_tasks

    # ── Select tasks (5/domain) from verified baselines ───────────────────────
    selected, counts = resolve_selected_tasks(
        cfg=cfg,
        scope_name=args.scope,
        tasks_path=args.tasks,
        baseline_path=baseline_path,
        model_name=model_name,
        balanced_per_domain=args.balanced_per_domain,
        experiment_seed=experiment_seed,
    )
    write_jsonl(args.selected_tasks, selected)
    timings["task_selection"] = counts
    timings["selected_task_ids"] = [t["task_id"] for t in selected]
    print(f"\nSelected {len(selected)} tasks for injections + dialogues:")
    for domain, n in sorted(counts["selected_by_domain"].items()):
        print(f"  {domain}: {n} (verified pool: {counts['verified_by_domain'].get(domain, 0)})")

    # ── Script 2: injections for SELECTED tasks only ──────────────────────────
    script2_cmd = [
        sys.executable,
        str(ROOT / "scripts/generate_injections.py"),
        "--baselines",
        str(baseline_path),
        "--tasks",
        str(args.selected_tasks),
        "--output",
        str(injections_path),
        "--k-values",
        *[str(k) for k in axes.k_injections],
        "--error-types",
        *axes.error_types,
        "--length-levels",
        *axes.length_levels,
    ]
    script2_elapsed = run_step(script2_cmd, label="Script 2 injections (selected-only)")
    timings["script2_elapsed_s"] = round(script2_elapsed, 1)
    timings["script2_tasks_processed"] = len(selected)

    # ── Preflight gate (hard stop before expensive Script 3) ──────────────────
    preflight_cmd = [
        sys.executable,
        str(ROOT / "scripts/preflight_gpu.py"),
        "--scope",
        args.scope,
        "--balanced-per-domain",
        str(args.balanced_per_domain),
    ]
    if args.skip_tests:
        preflight_cmd.append("--skip-tests")
    print(f"\n{'=' * 60}\n>>> Preflight gate (must PASS)\n{'=' * 60}")
    result = subprocess.run(preflight_cmd, cwd=ROOT)
    if result.returncode != 0:
        print("\n[ABORT] Preflight failed — dialog phase NOT started.", file=sys.stderr)
        timings["preflight_passed"] = False
        timings["total_elapsed_s"] = round(time.time() - run_t0, 1)
        write_run_timings(timings_path, timings)
        sys.exit(1)
    timings["preflight_passed"] = True

    # ── Script 3 + 4 via pipeline ─────────────────────────────────────────────
    pipeline_cmd = [
        sys.executable,
        str(ROOT / "scripts/run_pipeline.py"),
        "--config",
        str(args.config),
        "--scope",
        args.scope,
        "--script3",
        "adaptive",
        "--from-stage",
        "3",
        "--record-timings",
        "--budget-hours",
        str(args.budget_hours),
        "--balanced-per-domain",
        str(args.balanced_per_domain),
    ]
    script34_t0 = time.time()
    run_step(pipeline_cmd, label="Script 3+4 (adaptive dialogues + evaluate)")
    script34_elapsed = time.time() - script34_t0

    # Merge Script 3 timings from run_timings.json if present
    script3_payload: dict = {}
    if timings_path.exists():
        try:
            script3_payload = json.loads(timings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            script3_payload = {}

    planned_dialogues = count_planned_dialogues(
        selected,
        axes.as_dict(),
        include_neutral=bool(scope.get("include_neutral_control")),
    )

    total_extrap = build_total_extrapolation(
        script1_elapsed_s=script1_elapsed,
        script1_tasks_processed=script1_tasks,
        script2_elapsed_s=script2_elapsed,
        script2_tasks_processed=len(selected),
        script3_completed_dialogues=int(script3_payload.get("dialogues_completed") or 0),
        script3_compute_elapsed_s=float(script3_payload.get("compute_elapsed_s") or 0),
        script3_planned_dialogues=planned_dialogues,
        script4_elapsed_s=0.0,  # folded into script34; evaluate is fast
        model_load_s=float(script3_payload.get("model_load_s") or 0),
        pilot_tasks=len(selected),
        scope=args.scope,
    )

    # Script 4 wall time from pipeline_stages if recorded
    script4_s = 0.0
    for stage in script3_payload.get("pipeline_stages") or []:
        if stage.get("stage") == "evaluate.py":
            script4_s = float(stage.get("wall_time_s") or 0)
    if script4_s > 0:
        total_extrap = build_total_extrapolation(
            script1_elapsed_s=script1_elapsed,
            script1_tasks_processed=script1_tasks,
            script2_elapsed_s=script2_elapsed,
            script2_tasks_processed=len(selected),
            script3_completed_dialogues=int(script3_payload.get("dialogues_completed") or 0),
            script3_compute_elapsed_s=float(script3_payload.get("compute_elapsed_s") or 0),
            script3_planned_dialogues=planned_dialogues,
            script4_elapsed_s=script4_s,
            model_load_s=float(script3_payload.get("model_load_s") or 0),
            pilot_tasks=len(selected),
            scope=args.scope,
        )

    final_payload = {
        **script3_payload,
        **timings,
        **total_extrap,
        "script34_elapsed_s": round(script34_elapsed, 1),
        "script4_elapsed_s": round(script4_s, 1),
        "total_elapsed_s": round(time.time() - run_t0, 1),
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
    }
    write_run_timings(timings_path, final_payload)

    print(f"\n{'=' * 60}")
    print("GPU full run complete.")
    print(f"  Timings: {timings_path}")
    if total_extrap.get("extrapolated_full_thesis_total_h") is not None:
        print(
            f"  FULL THESIS estimate: ~{total_extrap['extrapolated_full_thesis_total_h']}h "
            f"(~{total_extrap.get('extrapolated_full_thesis_total_days')}d, "
            f"~${total_extrap.get('extrapolated_full_thesis_total_cost_usd')} @ $1.36/h)"
        )
    print(f"  Total wall time: {fmt_duration(final_payload['total_elapsed_s'])}")


if __name__ == "__main__":
    main()
