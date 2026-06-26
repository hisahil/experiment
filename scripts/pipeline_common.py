"""Shared helpers for run_pipeline.py and validate_grid.py."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))

from experiment_design import resolve_design_axes  # noqa: E402
from experiment_scope import get_scope  # noqa: E402
from grid_validation import resolve_tasks_path  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_prepare_output(cfg: dict, scope: dict) -> Path:
    """Task pool path for Stage 0 (matches scope.tasks when set)."""
    return resolve_tasks_path(cfg, scope)


def resolve_script3_mode(cfg: dict, override: str | None = None) -> str:
    if override:
        return override
    return cfg.get("script3", {}).get("dialogue_mode", "three_turn")


def script3_executable(mode: str) -> str:
    if mode == "adaptive_alternating":
        return "run_adaptive_dialogue.py"
    if mode == "three_turn":
        return "run_perturbed.py"
    raise ValueError(f"Unknown script3 mode: {mode}")


def common_forward_args(
    *,
    config: Path,
    scope: str | None,
    domain: str | None,
    limit: int | None,
    dry_run: bool,
    batch_size: int | None,
    balanced_per_domain: int | None = None,
    budget_hours: float | None = None,
    record_timings: bool = False,
) -> list[str]:
    cmd: list[str] = ["--config", str(config)]
    if scope:
        cmd.extend(["--scope", scope])
    if domain:
        cmd.extend(["--domain", domain])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if balanced_per_domain is not None:
        cmd.extend(["--balanced-per-domain", str(balanced_per_domain)])
    if budget_hours is not None:
        cmd.extend(["--budget-hours", str(budget_hours)])
    if record_timings:
        cmd.append("--record-timings")
    if dry_run:
        cmd.append("--dry-run")
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    return cmd


def build_stage_commands(
    *,
    experiment_config: Path,
    datasets_config: Path,
    scope: str | None,
    domain: str | None,
    limit: int | None,
    dry_run: bool,
    batch_size: int | None,
    script3_override: str | None,
    from_stage: int,
    skip_evaluate: bool,
    skip_validate: bool,
    balanced_per_domain: int | None = None,
    budget_hours: float | None = None,
    record_timings: bool = False,
    stage_a: bool = False,
) -> list[tuple[int, str, list[str]]]:
    cfg = load_yaml(experiment_config)
    scope_block = get_scope(cfg, scope)
    axes = resolve_design_axes(cfg, scope_block)
    prepare_output = resolve_prepare_output(cfg, scope_block)
    baselines_path = ROOT / cfg.get("baseline", {}).get("generated_file", "data/baseline_cots.jsonl")
    injections_path = ROOT / cfg.get("injection", {}).get(
        "generated_file", cfg.get("paths", {}).get("injections", "data/injections_generated.jsonl")
    )
    script3_out = ROOT / cfg.get("script3", {}).get("output_file", "data/post_injection_runs.jsonl")
    script3_mode = resolve_script3_mode(cfg, script3_override)
    script3_name = script3_executable(script3_mode)
    if stage_a and script3_mode != "adaptive_alternating":
        raise ValueError("--stage-a (judge-batch + step1-only) requires the adaptive script3 mode.")

    shared = common_forward_args(
        config=experiment_config,
        scope=scope,
        domain=domain,
        limit=limit,
        dry_run=dry_run,
        batch_size=batch_size,
        balanced_per_domain=balanced_per_domain,
        budget_hours=budget_hours,
        record_timings=record_timings,
    )

    stages: list[tuple[int, str, list[str]]] = []

    if from_stage <= 0:
        stages.append(
            (
                0,
                "prepare_tasks.py",
                [
                    sys.executable,
                    str(SCRIPTS / "prepare_tasks.py"),
                    "--config",
                    str(datasets_config),
                    "--output",
                    str(prepare_output),
                ],
            )
        )

    if from_stage <= 1:
        baseline_cmd = [
            sys.executable,
            str(SCRIPTS / "generate_baselines.py"),
            "--experiment-config",
            str(experiment_config),
            "--tasks",
            str(prepare_output),
        ]
        if domain:
            baseline_cmd.extend(["--domain", domain])
        if limit is not None:
            baseline_cmd.extend(["--limit", str(limit)])
        if dry_run:
            baseline_cmd.append("--dry-run")
        stages.append((1, "generate_baselines.py", baseline_cmd))

    if from_stage <= 2:
        injection_cmd = [
            sys.executable,
            str(SCRIPTS / "generate_injections.py"),
            "--baselines",
            str(baselines_path),
            "--tasks",
            str(prepare_output),
            "--output",
            str(injections_path),
            "--k-values",
            *[str(k) for k in axes.k_injections],
            "--error-types",
            *axes.error_types,
            "--length-levels",
            *axes.length_levels,
        ]
        if domain:
            injection_cmd.extend(["--domain", domain])
        if limit is not None:
            injection_cmd.extend(["--limit", str(limit)])
        if dry_run:
            injection_cmd.append("--dry-run")
        if stage_a:
            # Async pre-pass: ONE Message Batch (50% off), only step 1 per cell.
            # This subprocess blocks (polls) until injections are fully written,
            # so it acts as the poll-gate before the GPU dialogue stage.
            injection_cmd.extend(["--batch", "--step1-only"])
        stages.append((2, "generate_injections.py", injection_cmd))

    if from_stage <= 3:
        script3_cmd = [sys.executable, str(SCRIPTS / script3_name), *shared]
        if stage_a:
            # Disable inline judge; deterministic grading now, judge runs as a
            # downstream batch (Stage 3b) so the GPU never blocks on the judge API.
            script3_cmd.append("--no-judge")
        stages.append((3, script3_name, script3_cmd))

        if stage_a:
            # Phase B1: judge ALL rows once, then retry --only-fallbacks until the
            # fallback count hits 0 (max 5 passes). This drives judge_fallback to 0
            # before evaluate, fixing the pilot k=3 det-only confound. The
            # evaluate.py judge-coverage gate is the backstop if any remain.
            judge_cov_cmd = [
                sys.executable,
                str(SCRIPTS / "ensure_judge_coverage.py"),
                "--config",
                str(experiment_config),
                "--input",
                str(script3_out),
                "--tasks",
                str(prepare_output),
                "--initial-full-pass",
                "--max-passes",
                "5",
            ]
            if dry_run:
                judge_cov_cmd.append("--dry-run")
            stages.append((3, "ensure_judge_coverage.py", judge_cov_cmd))

    if from_stage <= 4 and not skip_evaluate:
        evaluate_cmd = [
            sys.executable,
            str(SCRIPTS / "evaluate.py"),
            "--config",
            str(experiment_config),
        ]
        if dry_run:
            evaluate_cmd.append("--dry-run")
        stages.append((4, "evaluate.py", evaluate_cmd))

    if not skip_validate:
        validate_cmd = [
            sys.executable,
            str(SCRIPTS / "validate_grid.py"),
            "--config",
            str(experiment_config),
            "--kind",
            "all",
        ]
        if scope:
            validate_cmd.extend(["--scope", scope])
        if domain:
            validate_cmd.extend(["--domain", domain])
        if limit is not None:
            validate_cmd.extend(["--limit", str(limit)])
        stages.append((5, "validate_grid.py", validate_cmd))

    return stages


def run_command(command: list[str], *, dry_run: bool, record_timings: bool = False) -> float | None:
    printable = " ".join(command)
    print(f"\n>>> {printable}")
    if dry_run:
        return None
    t0 = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.time() - t0
    if record_timings:
        timings_path = ROOT / "results" / "run_timings.json"
        timings_path.parent.mkdir(parents=True, exist_ok=True)
        stage_name = Path(command[1]).name if len(command) > 1 else "unknown"
        entry = {
            "stage": stage_name,
            "wall_time_s": round(elapsed, 1),
            "command": printable,
        }
        stages: list = []
        if timings_path.exists():
            try:
                existing = json.loads(timings_path.read_text(encoding="utf-8"))
                if isinstance(existing.get("pipeline_stages"), list):
                    stages = existing["pipeline_stages"]
            except json.JSONDecodeError:
                stages = []
        stages.append(entry)
        payload = {"pipeline_stages": stages}
        if timings_path.exists():
            try:
                payload = {**json.loads(timings_path.read_text(encoding="utf-8")), **payload}
            except json.JSONDecodeError:
                pass
        timings_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return elapsed


def print_plan(stages: list[tuple[int, str, list[str]]]) -> None:
    print("Pipeline plan:")
    for stage_id, name, _ in stages:
        print(f"  Stage {stage_id}: {name}")
