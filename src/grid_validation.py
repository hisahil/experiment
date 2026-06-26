"""Post-hoc and preflight grid validation for pipeline JSONL artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from baseline_index import index_verified_baselines
from experiment_design import (
    DesignAxes,
    expected_baseline_keys,
    expected_injection_keys,
    expected_pregenerated_steps,
    expected_run_keys,
    injection_cell_key,
    resolve_design_axes,
    run_cell_key,
)
from experiment_scope import get_scope, select_tasks, select_tasks_balanced


def compute_midpoint(steps: list[str]) -> tuple[int, str]:
    midpoint = len(steps) // 2
    prefix = "\n".join(steps[:midpoint])
    return midpoint, prefix


def injection_insertion_step(rec: dict | None) -> int | None:
    """Return Script 2 insertion_step for audit only."""
    if not rec:
        return None
    value = rec.get("insertion_step", rec.get("midpoint_step"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_injection_record(
    rec: dict,
    k: int,
    allow_step1_only: bool = False,
    *,
    expected_steps: int | None = None,
) -> list[str]:
    """Return human-readable issues when an injection row is structurally unusable.

    allow_step1_only: when True, records flagged step1_only (pre-pass for the
    adaptive dialogue, which only consumes injection_steps[0]) are accepted with a
    single non-empty step regardless of k. Only the adaptive path (run_adaptive_dialogue)
    sets this; run_perturbed keeps the strict len==k requirement because it injects all k.

    expected_steps: number of pre-generated steps the current dialogue_mode requires
    (see experiment_design.expected_pregenerated_steps). When given, a record is valid
    if it has exactly expected_steps OR the legacy k steps (transition tolerance, so an
    older full-k injection set still validates against an adaptive run). Defaults to the
    strict len==k requirement when omitted.
    """
    issues: list[str] = []
    steps = list(rec.get("injection_steps") or [])
    n = len(steps)
    if allow_step1_only and rec.get("step1_only"):
        if n < 1:
            issues.append("step1_only record has no injection step")
    elif expected_steps is not None:
        if n != expected_steps and n != k:
            issues.append(
                f"expected {expected_steps} injection_steps (or legacy {k}), got {n}"
            )
    elif n != k:
        issues.append(f"expected {k} injection_steps, got {n}")
    if any(not str(s).strip() for s in steps):
        issues.append("empty injection step(s)")
    return issues


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(path: Path, domain: str | None = None) -> list[dict]:
    tasks = load_jsonl(path)
    if domain:
        tasks = [t for t in tasks if t.get("domain") == domain]
    return tasks


def resolve_tasks_path(cfg: dict, scope: dict) -> Path:
    paths = cfg.get("paths", {})
    raw = scope.get("tasks") or paths.get("tasks", "data/experiment_tasks.jsonl")
    return Path(raw)


def resolve_eligible_tasks(
    cfg: dict,
    scope_name: str | None,
    *,
    domain: str | None = None,
    limit: int | None = None,
    baseline_path: Path | None = None,
    model_name: str | None = None,
    balanced_per_domain: int | None = None,
) -> tuple[dict, list[dict], DesignAxes, str]:
    """Scope, selected tasks (verified baseline only), design axes, model name.

    When balanced_per_domain is set, the same per-domain balanced selection used by
    the GPU run (run_adaptive_dialogue.py --balanced-per-domain / gpu_run_plan.py)
    is applied, so grid validation matches the actual N-task run instead of the
    full eligible pool (which would otherwise report thousands of false "missing
    cells").
    """
    scope = get_scope(cfg, scope_name)
    if domain:
        scope = dict(scope)
        scope["domains"] = [domain]

    experiment_seed = int(cfg.get("experiment", {}).get("seed", 42))
    model = model_name or cfg.get("model", {}).get("name", "")
    axes = resolve_design_axes(cfg, scope)

    tasks_path = resolve_tasks_path(cfg, scope)
    all_tasks = load_tasks(tasks_path, domain)
    task_by_id = {t["task_id"]: t for t in all_tasks}

    baseline_index: dict[str, dict] = {}
    if baseline_path and baseline_path.exists():
        baseline_index = index_verified_baselines(load_jsonl(baseline_path), model)

    eligible = [
        task_by_id[tid]
        for tid in sorted(baseline_index.keys())
        if tid in task_by_id
    ]
    pool = eligible if baseline_index else all_tasks
    if balanced_per_domain is not None:
        tasks = select_tasks_balanced(pool, scope, experiment_seed, balanced_per_domain)
    else:
        tasks = select_tasks(pool, scope, experiment_seed)
    if limit:
        tasks = tasks[:limit]
    return scope, tasks, axes, model


@dataclass
class GridValidationResult:
    kind: str
    expected_count: int
    actual_count: int
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    wrong_counts: list[tuple[str, int, int]] = field(default_factory=list)
    structural_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.wrong_counts or self.structural_errors)


def preflight_injections(
    tasks: list[dict],
    lists: dict[str, list[Any]],
    injection_index: dict[str, dict],
    baseline_index: dict[str, dict],
    *,
    pregenerate_steps: int | str = "k",
    dialogue_mode: str = "three_turn",
) -> tuple[list[str], list[str], list[str]]:
    """Pre-GPU validation for Script 2 coverage and baseline/injection alignment.

    pregenerate_steps / dialogue_mode control how many injection steps each cell is
    expected to carry. Defaults reproduce the legacy strict len==k check (run_perturbed);
    the adaptive path passes the real config so trimmed single-step records validate.
    """
    missing: list[str] = []
    invalid: list[str] = []
    insertion_mismatches: list[str] = []

    for task in tasks:
        task_id = task["task_id"]
        baseline = baseline_index.get(task_id)
        live_insertion_step: int | None = None
        if baseline:
            steps = list(baseline.get("steps") or [])
            if steps:
                live_insertion_step, _ = compute_midpoint(steps)

        for error_type in lists["error_types"]:
            for k in lists["k_injections"]:
                k_int = int(k)
                for length_level in lists["length_levels"]:
                    key = injection_cell_key(task_id, error_type, k_int, length_level)
                    rec = injection_index.get(key)
                    if rec is None:
                        missing.append(key)
                        continue
                    expected_steps = expected_pregenerated_steps(
                        k_int,
                        pregenerate_steps=pregenerate_steps,
                        dialogue_mode=dialogue_mode,
                    )
                    issues = validate_injection_record(
                        rec, k_int, expected_steps=expected_steps
                    )
                    if issues:
                        invalid.append(f"{key} ({'; '.join(issues)})")
                        continue
                    recorded = injection_insertion_step(rec)
                    if (
                        live_insertion_step is not None
                        and recorded is not None
                        and recorded != live_insertion_step
                    ):
                        insertion_mismatches.append(
                            f"{key}: injection metadata insertion_step={recorded}, "
                            f"baseline insertion_step={live_insertion_step}"
                        )
    return missing, invalid, insertion_mismatches


def index_injections(
    records: list[dict],
    target_model: str | None = None,
) -> dict[str, dict]:
    """task_id|error_type|k|length_level -> injection record."""
    out: dict[str, dict] = {}
    for rec in records:
        rec_model = rec.get("target_model")
        if target_model and rec_model and rec_model != target_model:
            continue
        key = injection_cell_key(
            rec["task_id"],
            rec["error_type"],
            int(rec["k"]),
            rec["length_level"],
        )
        out[key] = rec
    return out


def validate_injections_grid(
    records: list[dict],
    tasks: list[dict],
    axes: DesignAxes,
    *,
    target_model: str,
    pregenerate_steps: int | str = "k",
    dialogue_mode: str = "three_turn",
) -> GridValidationResult:
    expected = expected_injection_keys(tasks, axes)
    index = index_injections(records, target_model=target_model)
    actual = set(index.keys())

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    wrong_counts: list[tuple[str, int, int]] = []
    structural_errors: list[str] = []

    for key in sorted(expected & actual):
        rec = index[key]
        parts = key.split("|")
        k_int = int(parts[2])
        expected_steps = expected_pregenerated_steps(
            k_int,
            pregenerate_steps=pregenerate_steps,
            dialogue_mode=dialogue_mode,
        )
        issues = validate_injection_record(rec, k_int, expected_steps=expected_steps)
        if issues:
            structural_errors.append(f"{key} ({'; '.join(issues)})")

    counts = Counter(index.keys())
    for key, count in sorted(counts.items()):
        if count != 1:
            wrong_counts.append((key, count, 1))

    return GridValidationResult(
        kind="injections",
        expected_count=len(expected),
        actual_count=len(actual),
        missing=missing,
        extra=extra,
        wrong_counts=wrong_counts,
        structural_errors=structural_errors,
    )


def validate_runs_grid(
    records: list[dict],
    tasks: list[dict],
    axes: DesignAxes,
    *,
    model: str,
    variants: list[str] | None = None,
) -> GridValidationResult:
    perturbed_variants = variants or [v for v in axes.variants if v in ("stop", "recovery")]
    expected = expected_run_keys(tasks, axes, model=model, variants=perturbed_variants)

    actual_keys: list[tuple[str, str, int, str, int, str, str]] = []
    for rec in records:
        if rec.get("model") != model:
            continue
        variant = rec.get("variant")
        if variant not in perturbed_variants:
            continue
        actual_keys.append(
            run_cell_key(
                rec["task_id"],
                rec["error_type"],
                int(rec["k"]),
                rec["length_level"],
                int(rec["seed"]),
                variant,
                model,
            )
        )

    counts = Counter(actual_keys)
    actual = set(counts.keys())
    missing = sorted(str(k) for k in (expected - actual))
    extra = sorted(str(k) for k in (actual - expected))
    wrong_counts = [
        (str(key), counts[key], 1)
        for key in sorted(expected & actual)
        if counts[key] != 1
    ]

    return GridValidationResult(
        kind="runs",
        expected_count=len(expected),
        actual_count=len(actual),
        missing=missing,
        extra=extra,
        wrong_counts=wrong_counts,
    )


def validate_baselines_grid(
    records: list[dict],
    tasks: list[dict],
    *,
    model: str,
) -> GridValidationResult:
    expected = expected_baseline_keys(tasks, model=model)
    verified = {
        (rec["task_id"], rec.get("model", ""))
        for rec in records
        if rec.get("verified") and rec.get("model") == model
    }
    missing = sorted(f"{tid}|{model}" for tid, m in (expected - verified))
    extra = sorted(f"{tid}|{m}" for tid, m in (verified - expected))

    return GridValidationResult(
        kind="baselines",
        expected_count=len(expected),
        actual_count=len(verified),
        missing=missing,
        extra=extra,
    )


def format_validation_report(result: GridValidationResult) -> str:
    lines = [
        f"[{result.kind}] expected cells: {result.expected_count}, actual unique: {result.actual_count}",
    ]
    if result.missing:
        lines.append(f"  missing ({len(result.missing)}):")
        for item in result.missing[:10]:
            lines.append(f"    - {item}")
        if len(result.missing) > 10:
            lines.append(f"    ... and {len(result.missing) - 10} more")
    if result.extra:
        lines.append(f"  extra ({len(result.extra)}):")
        for item in result.extra[:10]:
            lines.append(f"    - {item}")
        if len(result.extra) > 10:
            lines.append(f"    ... and {len(result.extra) - 10} more")
    if result.wrong_counts:
        lines.append(f"  wrong_counts ({len(result.wrong_counts)}):")
        for key, got, want in result.wrong_counts[:10]:
            lines.append(f"    - {key}: got {got}, expected {want}")
    if result.structural_errors:
        lines.append(f"  structural_errors ({len(result.structural_errors)}):")
        for item in result.structural_errors[:10]:
            lines.append(f"    - {item}")
    if result.ok:
        lines.append("  OK")
    return "\n".join(lines)
