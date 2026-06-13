#!/usr/bin/env python3
"""
Script 4: Evaluation + REI (Recovery Efficiency Index) analysis.

Reads post_injection_runs.jsonl (Script 3 output) and baseline_cots.jsonl (Script 1
output), computes three orthogonal metrics per condition, runs paired
significance tests, an overthinking analysis, and descriptive sanity checks,
then writes CSV/TXT summaries for the thesis.

Metrics (per task-condition, then aggregated):
  - Accuracy Drop = Acc_Clean - Acc_Stop
  - REI           = (Acc_Recovery - Acc_Stop) / (Tokens_Recovery - Tokens_Stop)
  - Recovery Rate = (Acc_Recovery - Acc_Stop) / (Acc_Clean - Acc_Stop)

Two REI flavours are reported:
  - rei_natural (PRIMARY): recovery rows with recovery_terminated_by == "natural"
  - rei_all (TRANSPARENCY): all recovery rows regardless of termination

SCHEMA RECONCILIATION (with Scripts 1 & 3, verified against the actual files):
  - post_injection_runs.jsonl has variant in {"stop", "recovery"} only. There is NO
    "clean"/"baseline" variant. Clean accuracy is therefore derived from
    baseline_cots.jsonl (Script 1), per (task_id, model). See
    clean_accuracy_by_task_model() for the documented choice.
  - There is no "run_type" column (supplementary positions were cut). Any such
    filter is applied only if the column is present.
  - recovery_terminated_by in {"natural", "token_limit", "extraction_failed"}.
  - code_tests_passed / code_tests_total are NOT emitted by Script 3; they are
    read if present and otherwise left null.

Usage:
  python scripts/evaluate.py --dry-run            # uses synthetic files, no GPU/API
  python scripts/evaluate.py                       # uses config input/output paths
  python scripts/evaluate.py --input data/post_injection_runs_synthetic.jsonl \
                             --baselines data/baseline_cots_synthetic.jsonl

Two-level grouping (see CONDITION_KEYS below):
  - A "condition" / design cell = one tuple of (model, domain, error_type, k, length_level).
    Example: DeepSeek · math · incorrect_reasoning · k=1 · short.
  - results_per_task.csv: one row per TASK inside that cell (algebra_2391 in the cell above).
  - results_per_condition.csv: mean + CI over all tasks sharing the same CONDITION_KEYS
    (13 math tasks in that cell → 13 per-task rows collapse to 1 condition row).

Outputs (in output_dir, default results/):
  results_per_task.csv        per task-condition metrics + difficulty bin;
                              cot_length_inflation_*, recovery_harmed_natural
  results_per_condition.csv   aggregated metrics + CI per condition
  sanity_checks.csv           corruption rates per error_type (warnings, not gates)
  significance_tests.csv      paired t-tests + n_eligible/n_skipped/skip-reason breakdown
  overthinking.csv            recovery termination breakdown (token_limit signal)
  rei_summary.txt             human-readable summary for thesis writing
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required — run: pip install pandas")

try:
    from scipy import stats as scipy_stats
except ImportError:
    sys.exit("scipy is required — run: pip install scipy")

try:
    import yaml
except ImportError:
    yaml = None  # config is optional; defaults below are sufficient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Same grading rules as Script 3 (is_correct, answer extraction).
from evaluator import (  # noqa: E402
    effective_recovery_termination,
    extract_domain_answer,
    infer_math_eval_type,
    is_correct,
)
# Shared verified-baseline selection (identical to Script 3's injection set).
from baseline_index import clean_accuracy_map  # noqa: E402

DEFAULT_EXPERIMENT_CONFIG = ROOT / "configs/experiment.yaml"

# One factorial design cell, without task_id. Script 4 groups post_injection_runs rows
# by these five fields, then adds task_id for the per-task table.
# Example cell: (DeepSeek, math, incorrect_reasoning, 1, short) holds many tasks;
# aggregate_condition() averages REI/accuracy across those tasks into one condition row.
CONDITION_KEYS = ("model", "domain", "error_type", "k", "length_level")

# Defaults mirror configs/experiment.yaml -> script4. Overridden by resolve_config().
DEFAULT_CONFIG: dict[str, Any] = {
    "input_path": "data/post_injection_runs.jsonl",
    "baseline_path": "data/baseline_cots.jsonl",
    "output_dir": "results",
    "ci_z": 1.96,  # 95% CI half-width: ci_z * SEM per condition row
    # Descriptive threshold only (no filtering): count tasks in a condition where
    # n_recovery_natural_seeds >= this. Full run uses 3 seeds (max 3 natural recoveries
    # per task); pilot_gpu uses seed 42 only → max 1, so n_tasks_min_recovery_seeds is
    # usually 0 and the threshold is not reachable on pilot scale.
    "min_recovery_seeds": 2,
    # REI = (acc_recovery - acc_stop) / (tokens_recovery - tokens_stop); null if delta < this (1 = only delta<=0)
    "min_denominator_tokens": 1,
    # Warn per condition if fewer than N tasks have non-null rei_natural (pilot ~6-13)
    "min_valid_tasks": 60,
    "difficulty_bins": {"easy": 0.70, "hard": 0.40},
    "sanity_thresholds": {
        "incorrect_reasoning": 0.70,
        "off_topic": 0.60,
        "paraphrased_shift": 0.50,
    },
    "math_tolerance": 1e-4,
    # Recovery cap from Script 3; used to reclassify token_limit after parser fixes.
    "recovery_max_tokens": 8192,
    # Stop cap from Script 3; used to label truncated stop runs in ungradable analysis.
    "stop_max_new_tokens": 100,
}

# Warn in rei_summary when this fraction of stop runs hit the token cap (not EOS).
STOP_LENGTH_WARN_FRACTION = 0.05

# Dry-run reads these synthetic files (no GPU, no API). Generate with
# scripts/make_synthetic_eval_data.py.
DRY_RUN_INPUT = "data/post_injection_runs_synthetic.jsonl"
DRY_RUN_BASELINES = "data/baseline_cots_synthetic.jsonl"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    if yaml is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_runs(path: Path) -> list[dict]:
    """Load a JSONL file of run records."""
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Answer grading (_run_correct uses src.evaluator.extract_domain_answer as fallback)
# ---------------------------------------------------------------------------


def compute_accuracy(
    extracted: str | None,
    gold: str | None,
    domain: str,
    math_tolerance: float,
    eval_type: str | None = None,
) -> bool:
    """
    Compare one already-extracted model answer to the reference (gold).

    This is the low-level grader: it assumes parsing is done. It does NOT read
    raw_output — that happens in _run_correct() before this is called.

    Inputs:
      extracted — answer string from Script 3 (extracted_answer) or fallback parse
      gold      — reference from the task row (gold_answer)
      domain    — math | science | logic (selects grading rule)
      eval_type — optional; refines math/logic/science matching (from task row)

    Grading (via src.evaluator.is_correct, same rules as Scripts 1 & 3):
      math + numeric   — float compare within math_tolerance (GSM8K)
      math + symbolic  — SymPy equivalence, else normalized string (MATH-500)
      math, no eval_type — infer numeric vs symbolic from gold shape
      science          — normalized numeric/string match (eval_type from task)
      logic            — knight/knave assignment equivalence

    Returns True if correct, False if wrong OR if extracted/gold is missing.
    Only ever bool — never None. Callers that need "ungradable" (no parse at all)
    must check for missing extracted/gold before calling (see _run_correct).
    """
    if extracted is None or gold is None:
        return False
    if domain == "math":
        resolved_eval_type = eval_type or infer_math_eval_type(str(gold))
        result = is_correct(
            extracted,
            str(gold),
            resolved_eval_type,
            math_tolerance=math_tolerance,
        )
        return bool(result)
    resolved_eval_type = eval_type or "exact"
    result = is_correct(extracted, str(gold), resolved_eval_type, math_tolerance=math_tolerance)
    return bool(result)


def _gold_missing(gold: object) -> bool:
    """True when there is no reference answer to grade against."""
    return gold is None or not str(gold).strip()


def _run_correct(run: dict, math_tolerance: float) -> bool | None:
    """
    Was this run correct? Used by score_runs() to set _correct (1.0 / 0.0 / None).

    Path A — normal (Script 3 already graded):
      If run["correct"] is true or false, return it as-is. raw_output is not re-read.

    Path B — fallback (correct is null or missing, e.g. extraction_failed):
      1. Use extracted_answer from the row, or re-parse raw_output via
         extract_domain_answer() (evaluator.py).
      2. If an answer is found, compare to gold_answer via compute_accuracy().
      3. If parsing still fails (or gold is None/blank), return None — ungradable,
         not wrong. score_runs() writes _correct=None; _mean() excludes these rows.
    """
    if "correct" in run and run["correct"] is not None:
        return bool(run["correct"])
    extracted = run.get("extracted_answer")
    if extracted is None:
        raw = run.get("raw_output")
        if raw is not None:
            extracted = extract_domain_answer(
                str(raw), run.get("domain", "math"), run.get("eval_type")
            )
    if extracted is None or _gold_missing(run.get("gold_answer")):
        return None
    return compute_accuracy(
        extracted,
        run.get("gold_answer"),
        run.get("domain", "math"),
        math_tolerance,
        eval_type=run.get("eval_type"),
    )


def _classify_ungradable(run: dict, *, stop_max: int, recovery_max: int) -> str:
    """
    Why is this run ungradable (_correct is None)? Used to rule out parser weakness.

    Ungradable always means extract_domain_answer() found no answer. Categorise:
      missing_gold      — gold_answer is None or blank (defensive; ~never on real data)
      empty_output      — model produced no text at all
      truncated         — hit the variant token cap mid-generation (no final answer
                          reached) → genuine no-answer, NOT a parser problem
      no_answer_natural — model ended on its own (tokens < cap) yet nothing parsed.
                          The ONLY bucket where a parser miss cannot be excluded —
                          inspect these raw_outputs.
    """
    if _gold_missing(run.get("gold_answer")):
        return "missing_gold"
    raw = run.get("raw_output")
    if raw is None or not str(raw).strip():
        return "empty_output"
    if run.get("finish_reason") == "length":
        return "truncated"
    cap = recovery_max if run.get("variant") == "recovery" else stop_max
    tokens = int(run.get("tokens_generated") or 0)
    if cap and tokens >= cap:
        return "truncated"
    return "no_answer_natural"


def score_runs(runs: list[dict], config: dict) -> list[dict]:
    """
    Annotate every loaded run with _correct — the numeric score used by REI.

    Called once in main() after loading post_injection_runs.jsonl, before any
    per-task or per-condition aggregation. Mutates runs in place.

    Per run:
      1. _run_correct(run, math_tolerance) → True / False / None
      2. Map to _correct:
           True  → 1.0  (correct)
           False → 0.0  (wrong)
           None  → None (ungradable — e.g. extraction_failed; _mean() skips these)
         Ungradable runs also get _ungradable_reason (see _classify_ungradable).
      3. Recovery rows also get _recovery_terminated_by, re-derived from raw_output
         with the current parser (so historical JSONL benefits from parser fixes).

    Downstream consumers: build_per_task_condition() (acc_stop, acc_recovery),
    run_sanity_checks() (corruption on stop rows), analyze_overthinking(),
    analyze_ungradable() (reason breakdown).
    """
    tol = float(config["math_tolerance"])
    recovery_max = int(config["recovery_max_tokens"])
    stop_max = int(config["stop_max_new_tokens"])
    for run in runs:
        result = _run_correct(run, tol)
        if result is None:
            run["_correct"] = None
            run["_ungradable_reason"] = _classify_ungradable(
                run, stop_max=stop_max, recovery_max=recovery_max
            )
        else:
            run["_correct"] = 1.0 if result else 0.0
            run["_ungradable_reason"] = None
        if run.get("variant") == "recovery":
            run["_recovery_terminated_by"] = effective_recovery_termination(
                run,
                recovery_max_tokens=recovery_max,
            )
    return runs


# ---------------------------------------------------------------------------
# Clean accuracy (from baselines — no clean variant in post_injection_runs)
# ---------------------------------------------------------------------------


def baseline_tokens_by_task_model(baselines: list[dict]) -> dict[tuple[str, str], float]:
    """
    (task_id, model) -> baseline CoT length proxy for length-inflation metrics.

    Uses verified baselines only (same set as Script 3 / clean_accuracy_map).
    baseline_cots.jsonl has no tokens_generated field; we use word count on
    baseline_cot (or joined steps) as a reproducible proxy comparable across
    tasks. cot_length_inflation = (tokens_recovery - tokens_baseline) / tokens_baseline.
    """
    out: dict[tuple[str, str], float] = {}
    for rec in baselines:
        if not rec.get("verified"):
            continue
        key = (rec.get("task_id", ""), rec.get("model", ""))
        cot = rec.get("baseline_cot") or ""
        if not str(cot).strip() and rec.get("steps"):
            cot = "\n".join(str(s) for s in rec["steps"])
        words = len(str(cot).split()) if str(cot).strip() else 0
        out[key] = float(words)
    return out


def compute_cot_length_inflation(tokens_recovery: float, tokens_baseline: float) -> float | None:
    """
    Relative CoT length change vs clean baseline (Recum-style length inflation).

    Returns (tokens_recovery - tokens_baseline) / tokens_baseline, or None when
    baseline length is zero or inputs are missing.
    """
    for v in (tokens_recovery, tokens_baseline):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    if tokens_baseline <= 0:
        return None
    return (tokens_recovery - tokens_baseline) / tokens_baseline


def clean_accuracy_by_task_model(baselines: list[dict], config: dict) -> dict[tuple[str, str], float]:
    """
    Clean (un-perturbed) accuracy per (task_id, model), derived from Script 1.

    DESIGN CHOICE: post_injection_runs.jsonl has no clean variant, so clean
    accuracy comes from baseline_cots.jsonl. It uses ONLY verified=True rows —
    exactly the baseline that Script 3 injected into — via the shared
    baseline_index.clean_accuracy_map(). Because verified means correct by
    definition, clean accuracy is 1.0 for every perturbed task. Failed retries
    (verified=False) are NOT averaged in (which would otherwise dilute the value
    and make it inconsistent with the trace Script 3 actually used). Tasks
    without a verified baseline are absent from the map → acc_clean = NaN.
    """
    tol = float(config["math_tolerance"])

    def _grader(extracted, gold, domain, eval_type):
        return compute_accuracy(extracted, gold, domain, tol, eval_type=eval_type)

    return clean_accuracy_map(baselines, grader=_grader)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    """
    Arithmetic mean over a list of numeric scores, skipping missing data.

    Drops None and NaN before averaging; returns NaN if nothing valid remains.

    Used at two levels:
      - Per task (build_per_task_condition): mean over runs/seeds, e.g. _correct
        (accuracy) or tokens_generated for stop vs recovery variants.
      - Per condition (aggregate_condition): mean over tasks that share the same
        CONDITION_KEYS, e.g. acc_stop or rei_natural from results_per_task rows.
    """
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _ci_halfwidth(values: list[float], ci_z: float) -> float:
    """
    Normal-approximation CI half-width: z * SEM (standard error of the mean).

    Same None/NaN filtering as _mean(). Returns NaN when n < 2 (variance undefined).

    Only used in aggregate_condition(): spread is across *tasks* within one design
    cell (model, domain, error_type, k, length_level), not across run seeds.
    Example: rei_natural from 13 tasks → mean ± this half-width in
    results_per_condition.csv (ci_z from config, typically 1.96 for ~95% CI).
    """
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n < 2:
        return float("nan")
    return float(ci_z * np.std(vals, ddof=1) / math.sqrt(n))


def bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for the mean of per-task values within one condition.

    Resamples the (None/NaN-filtered) task values with replacement n_boot times,
    takes the mean of each resample, and returns the (1-ci)/2 and (1+ci)/2
    percentiles. Complements the normal-approximation _ci_halfwidth() and makes
    no symmetry/normality assumption, which suits the small, skewed per-condition
    samples (e.g. rei_natural). Returns (NaN, NaN) when n < 2.
    """
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    arr = np.asarray(vals, dtype=float)
    means = arr[rng.randint(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (1.0 - ci) / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 + ci) / 2.0))
    return lo, hi


def compute_rei(
    acc_recovery: float,
    acc_stop: float,
    tokens_recovery: float,
    tokens_stop: float,
    min_denominator_tokens: int,
) -> float | None:
    """
    Recovery Efficiency Index: accuracy gain per extra token spent on recovery.

    REI = (acc_recovery - acc_stop) / (tokens_recovery - tokens_stop).
    Primary metric uses natural-termination recovery; rei_all includes token_limit runs.
    Returns None when inputs are missing or delta_tokens < min_denominator_tokens.
    With min_denominator_tokens=1 (default), only non-positive deltas are excluded
    (division by zero or negative denominator); small positive deltas are kept.
    """
    for v in (acc_recovery, acc_stop, tokens_recovery, tokens_stop):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    delta_tokens = tokens_recovery - tokens_stop
    if delta_tokens < min_denominator_tokens:
        return None
    return (acc_recovery - acc_stop) / delta_tokens


def compute_recovery_rate(acc_recovery: float, acc_stop: float, acc_clean: float) -> float | None:
    """
    Fraction of injection damage recovered by continuing reasoning.

    RR = (acc_recovery - acc_stop) / (acc_clean - acc_stop).
    1.0 = back to clean accuracy; 0.0 = no gain over stop.
    None when any input is missing or acc_clean <= acc_stop (no measurable drop).
    """
    for v in (acc_recovery, acc_stop, acc_clean):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    denom = acc_clean - acc_stop
    if denom <= 0:
        return None
    return (acc_recovery - acc_stop) / denom


def assign_difficulty(clean_acc: float, bins: dict) -> str:
    """
    Descriptive easy/medium/hard label from Script 1 baseline accuracy (clean_map).

    Thresholds from config difficulty_bins (default: easy >= 0.70, hard <= 0.40).
    Used only to slice results_per_task rows — never filters or drops runs.
    """
    if clean_acc is None or (isinstance(clean_acc, float) and math.isnan(clean_acc)):
        return "unknown"
    if clean_acc >= bins.get("easy", 0.70):
        return "easy"
    if clean_acc <= bins.get("hard", 0.40):
        return "hard"
    return "medium"


# ---------------------------------------------------------------------------
# Per task-condition table  ->  results_per_task.csv
# ---------------------------------------------------------------------------


def _round(value: float | None, ndigits: int) -> float | None:
    """CSV formatting only: round floats; pass None/NaN through unchanged."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), ndigits)


def _acc_missing_reason(
    acc: float | None,
    *,
    kind: str,
    n_stop: int,
    n_recovery_all: int,
    n_recovery_natural_seeds: int,
    has_baseline: bool,
) -> str | None:
    """
    Why a per-task accuracy is None (task-cell excluded from paired t-tests).

    kind: stop | recovery_natural | recovery_all | clean
    """
    if acc is not None:
        return None
    if kind == "stop":
        return "no_stop_runs" if n_stop == 0 else "stop_runs_ungradable"
    if kind == "recovery_all":
        if n_recovery_all == 0:
            return "no_recovery_runs"
        return "recovery_runs_ungradable"
    if kind == "recovery_natural":
        if n_recovery_all == 0:
            return "no_recovery_runs"
        if n_recovery_natural_seeds == 0:
            return "no_natural_recovery"
        return "natural_runs_ungradable"
    if kind == "clean":
        return "no_verified_baseline" if not has_baseline else "unknown"
    return "unknown"


# Maps acc_* column -> per-task missing_reason column (results_per_task.csv).
_ACC_MISSING_REASON_COL = {
    "acc_stop": "acc_stop_missing_reason",
    "acc_recovery_natural": "acc_recovery_natural_missing_reason",
    "acc_recovery_all": "acc_recovery_all_missing_reason",
    "acc_clean": "acc_clean_missing_reason",
}


def build_per_task_condition(
    runs: list[dict],
    clean_map: dict[tuple[str, str], float],
    baseline_token_map: dict[tuple[str, str], float],
    config: dict,
) -> list[dict]:
    """
    Stage 1 aggregation -> results_per_task.csv (one row per task x design cell).

    Step 1 — group runs by (CONDITION_KEYS + task_id): same model/domain/error/k/length
    and same task, possibly multiple seeds and stop+recovery variants.
    Step 2 — split each group into stop vs recovery (natural vs token_limit for REI).
    Step 3 — mean _correct and tokens over seeds within each split; combine with
    clean_map baseline into REI, recovery rate, accuracy_drop, difficulty_bin.

    acc_*_missing_reason columns document why a task-cell lacks an accuracy
    (e.g. no_natural_recovery) for significance_tests skip reporting.
    """
    bins = config["difficulty_bins"]
    min_tokens = int(config["min_denominator_tokens"])

    # Step 1: one bucket per experimental cell + task (e.g. 5 seeds -> 5 stop + 5 recovery).
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        key = tuple(r.get(col) for col in CONDITION_KEYS) + (r.get("task_id"),)
        groups[key].append(r)

    rows: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        model, domain, error_type, k, length_level, task_id = key

        # Step 2: variant splits. rei_natural / acc_recovery_natural use natural only.
        stop_rows = [r for r in group if r.get("variant") == "stop"]
        rec_all = [r for r in group if r.get("variant") == "recovery"]
        rec_natural = [
            r for r in rec_all if r.get("_recovery_terminated_by", r.get("recovery_terminated_by")) == "natural"
        ]
        rec_token_limit = [
            r for r in rec_all if r.get("_recovery_terminated_by", r.get("recovery_terminated_by")) == "token_limit"
        ]

        # Step 3a: per-task means over run seeds (not over tasks — that is aggregate_condition).
        acc_stop = _mean([r["_correct"] for r in stop_rows])
        acc_rec_all = _mean([r["_correct"] for r in rec_all])
        acc_rec_nat = _mean([r["_correct"] for r in rec_natural])

        tok_stop = _mean([float(r.get("tokens_generated", 0) or 0) for r in stop_rows])
        tok_rec_all = _mean([float(r.get("tokens_generated", 0) or 0) for r in rec_all])
        tok_rec_nat = _mean([float(r.get("tokens_generated", 0) or 0) for r in rec_natural])

        acc_clean = clean_map.get((task_id, model), float("nan"))
        has_baseline = (task_id, model) in clean_map
        difficulty = assign_difficulty(acc_clean, bins)

        acc_clean_out = _round(acc_clean, 4)
        acc_stop_out = _round(acc_stop, 4)
        acc_rec_nat_out = _round(acc_rec_nat, 4)
        acc_rec_all_out = _round(acc_rec_all, 4)
        n_stop = len(stop_rows)
        n_rec_all = len(rec_all)
        n_rec_nat_seeds = len(rec_natural)

        # Explicit reasons when acc_* is None — used by significance skip reporting.
        acc_stop_missing_reason = _acc_missing_reason(
            acc_stop_out,
            kind="stop",
            n_stop=n_stop,
            n_recovery_all=n_rec_all,
            n_recovery_natural_seeds=n_rec_nat_seeds,
            has_baseline=has_baseline,
        )
        acc_recovery_natural_missing_reason = _acc_missing_reason(
            acc_rec_nat_out,
            kind="recovery_natural",
            n_stop=n_stop,
            n_recovery_all=n_rec_all,
            n_recovery_natural_seeds=n_rec_nat_seeds,
            has_baseline=has_baseline,
        )
        acc_recovery_all_missing_reason = _acc_missing_reason(
            acc_rec_all_out,
            kind="recovery_all",
            n_stop=n_stop,
            n_recovery_all=n_rec_all,
            n_recovery_natural_seeds=n_rec_nat_seeds,
            has_baseline=has_baseline,
        )
        acc_clean_missing_reason = _acc_missing_reason(
            acc_clean_out,
            kind="clean",
            n_stop=n_stop,
            n_recovery_all=n_rec_all,
            n_recovery_natural_seeds=n_rec_nat_seeds,
            has_baseline=has_baseline,
        )

        # Step 3b: derived per-task metrics (inputs to _AGG_COLS at condition level).
        accuracy_drop = (
            acc_clean - acc_stop
            if not (math.isnan(acc_clean) or math.isnan(acc_stop))
            else None
        )
        rei_natural = compute_rei(acc_rec_nat, acc_stop, tok_rec_nat, tok_stop, min_tokens)
        rei_all = compute_rei(acc_rec_all, acc_stop, tok_rec_all, tok_stop, min_tokens)
        rr_natural = compute_recovery_rate(acc_rec_nat, acc_stop, acc_clean)
        rr_all = compute_recovery_rate(acc_rec_all, acc_stop, acc_clean)

        tokens_baseline = baseline_token_map.get((task_id, model), float("nan"))
        cot_inflation_nat = compute_cot_length_inflation(tok_rec_nat, tokens_baseline)
        cot_inflation_all = compute_cot_length_inflation(tok_rec_all, tokens_baseline)
        recovery_harmed_natural = (
            rei_natural is not None
            and not (isinstance(rei_natural, float) and math.isnan(rei_natural))
            and rei_natural < 0
        )

        # code test counts: read if present (Script 3 does not emit them today)
        code_passed = next((r.get("code_tests_passed") for r in group if r.get("code_tests_passed") is not None), None)
        code_total = next((r.get("code_tests_total") for r in group if r.get("code_tests_total") is not None), None)

        rows.append(
            {
                "model": model,
                "domain": domain,
                "error_type": error_type,
                "k": k,
                "length_level": length_level,
                "task_id": task_id,
                "difficulty_bin": difficulty,
                "acc_clean": acc_clean_out,
                "acc_stop": acc_stop_out,
                "acc_recovery_natural": acc_rec_nat_out,
                "acc_recovery_all": acc_rec_all_out,
                "acc_stop_missing_reason": acc_stop_missing_reason,
                "acc_recovery_natural_missing_reason": acc_recovery_natural_missing_reason,
                "acc_recovery_all_missing_reason": acc_recovery_all_missing_reason,
                "acc_clean_missing_reason": acc_clean_missing_reason,
                "accuracy_drop": _round(accuracy_drop, 4),
                "recovery_rate_natural": _round(rr_natural, 4),
                "recovery_rate_all": _round(rr_all, 4),
                "rei_natural": _round(rei_natural, 6),
                "rei_all": _round(rei_all, 6),
                "tokens_baseline": _round(tokens_baseline, 2),
                "tokens_stop_mean": _round(tok_stop, 2),
                "tokens_recovery_natural_mean": _round(tok_rec_nat, 2),
                "tokens_recovery_all_mean": _round(tok_rec_all, 2),
                "cot_length_inflation_natural": _round(cot_inflation_nat, 4),
                "cot_length_inflation_all": _round(cot_inflation_all, 4),
                "recovery_harmed_natural": recovery_harmed_natural,
                "n_stop": n_stop,
                "n_recovery_all": n_rec_all,
                # Number of recovery runs with recovery_terminated_by == "natural" for this
                # task-cell (one per seed). Compared to min_recovery_seeds at aggregate time.
                "n_recovery_natural_seeds": n_rec_nat_seeds,
                "n_recovery_token_limit": len(rec_token_limit),
                "code_tests_passed": code_passed,
                "code_tests_total": code_total,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Condition-level aggregation  ->  results_per_condition.csv
# ---------------------------------------------------------------------------

# Stage 2 column list: per-task metrics to average across tasks sharing CONDITION_KEYS
# (without task_id). aggregate_condition() loops this list -> mean + CI per condition row.
_AGG_COLS = [
    "acc_clean",
    "acc_stop",
    "acc_recovery_natural",
    "acc_recovery_all",
    "accuracy_drop",
    "recovery_rate_natural",
    "recovery_rate_all",
    "rei_natural",
    "rei_all",
    "tokens_baseline",
    "tokens_stop_mean",
    "tokens_recovery_natural_mean",
    "tokens_recovery_all_mean",
    "cot_length_inflation_natural",
    "cot_length_inflation_all",
]


def aggregate_condition(per_task_rows: list[dict], config: dict) -> list[dict]:
    """
    Stage 2 aggregation -> results_per_condition.csv (one row per design cell).

    Step 1 — group per_task rows by CONDITION_KEYS only (drop task_id).
    Step 2 — for each _AGG_COLS metric, mean across tasks + normal-approx CI
    (spread is task-to-task, not seed-to-seed).
    Step 3 — bootstrap CI for headline metrics; count valid REI / recovery seeds.
    """
    ci_z = float(config["ci_z"])
    min_valid = int(config["min_valid_tasks"])
    min_rec_seeds = int(config["min_recovery_seeds"])

    # Step 1: e.g. 13 tasks under (DeepSeek, math, incorrect_reasoning, 1, short) -> one bucket.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_task_rows:
        groups[tuple(row[col] for col in CONDITION_KEYS)].append(row)

    out: list[dict] = []
    for key, task_rows in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        agg: dict[str, Any] = dict(zip(CONDITION_KEYS, key))
        agg["n_tasks"] = len(task_rows)

        # Step 2: one task value per column -> condition mean + z*SEM half-width.
        for col in _AGG_COLS:
            vals = [r[col] for r in task_rows]
            agg[col] = _round(_mean(vals), 6)
            agg[f"{col}_ci"] = _round(_ci_halfwidth(vals, ci_z), 6)

        # Step 3a: bootstrap lo/hi for primary thesis metrics (small-n robustness).
        for col in ("accuracy_drop", "recovery_rate_natural", "rei_natural"):
            lo, hi = bootstrap_ci([r[col] for r in task_rows])
            agg[f"{col}_ci_lo"] = _round(lo, 6)
            agg[f"{col}_ci_hi"] = _round(hi, 6)

        # Step 3b: descriptive counts (never drop tasks from the mean above).
        n_valid_rei = sum(1 for r in task_rows if r["rei_natural"] is not None)
        agg["rei_natural_n_valid"] = n_valid_rei

        n_ok_seeds = sum(1 for r in task_rows if r["n_recovery_natural_seeds"] >= min_rec_seeds)
        agg["n_tasks_min_recovery_seeds"] = n_ok_seeds

        # Warn only (not a gate): few gradable REI tasks expected on pilot scale.
        if 0 < n_valid_rei < min_valid:
            warnings.warn(
                f"[REI] condition {dict(zip(CONDITION_KEYS, key))}: "
                f"rei_natural_n_valid={n_valid_rei} < min_valid_tasks={min_valid}",
                stacklevel=2,
            )
        out.append(agg)
    return out


# ---------------------------------------------------------------------------
# Sanity checks (descriptive warnings, NOT gates)
# ---------------------------------------------------------------------------


def run_sanity_checks(runs: list[dict], config: dict) -> list[dict]:
    """
    Per error_type corruption_rate on stop rows (extracted != gold). Warns when
    below the configured threshold but never fails the run.

    Denominator is gradable stop runs only (_correct is not None): ungradable
    runs are neither correct nor corrupt, so counting them would dilute the rate.
    Stop runs are prefilled, so n_ungradable is normally ~0; it is reported for
    transparency and to keep the metric correct in edge cases.
    """
    thresholds = config["sanity_thresholds"]
    by_et: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        if r.get("variant") == "stop":
            by_et[r.get("error_type", "unknown")].append(r)

    rows: list[dict] = []
    for et, recs in sorted(by_et.items()):
        n = len(recs)
        gradable = sum(1 for r in recs if r["_correct"] is not None)
        n_ungradable = n - gradable
        corrupted = sum(1 for r in recs if r["_correct"] == 0.0)
        corruption_rate = corrupted / gradable if gradable else float("nan")
        thresh = thresholds.get(et)
        if thresh is not None and not math.isnan(corruption_rate) and corruption_rate < thresh:
            warnings.warn(
                f"[SANITY] error_type={et}: corruption_rate={corruption_rate:.2%} "
                f"< threshold {thresh:.0%} (gradable n={gradable})",
                stacklevel=2,
            )
        rows.append(
            {
                "error_type": et,
                "n_stop_runs": n,
                "n_ungradable": n_ungradable,
                "corruption_rate": _round(corruption_rate, 4),
                "threshold": thresh,
                "below_threshold": (
                    bool(thresh is not None and not math.isnan(corruption_rate) and corruption_rate < thresh)
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Overthinking analysis (recovery termination breakdown)
# ---------------------------------------------------------------------------
# Diagnostic table: how recovery runs end, and how accurate/long each group is.
# Not used for REI (REI uses recovery_terminated_by == "natural" only).
# Overthinking signal: many token_limit rows with low accuracy and high tokens.


def analyze_overthinking(runs: list[dict]) -> list[dict]:
    """
    Group recovery rows by recovery_terminated_by and summarize each bucket.

    Question answered: "After the injection, does the model recover cleanly,
    ramble until the token cap, or produce unparseable output?"

    recovery_terminated_by values (set in score_runs via effective_recovery_termination):
      natural           — model finished; parser extracted an answer.
      token_limit       — hit recovery_max_tokens (e.g. 8192); may still be gradable.
      extraction_failed — no parseable answer in raw_output.

    Output columns (written to results/overthinking.csv):
      n                   — count in this bucket
      n_gradable          — runs with _correct is not None (denominator for accuracy)
      fraction_of_recovery — n / all recovery runs
      accuracy            — mean of _correct (1.0/0.0); None rows excluded by _mean()
      tokens_mean         — mean tokens_generated in this bucket

    Overthinking: token_limit rows with low accuracy and tokens_mean near the cap
    (model spent full budget but rarely got the answer right).

    Example row (pilot-scale):
      natural:           n=176, fraction=0.93, accuracy=0.68, tokens_mean=1763
      token_limit:       n=4,   fraction=0.02, accuracy=0.25, tokens_mean=8192
      extraction_failed: n=9,   fraction=0.05, accuracy=None, tokens_mean=8192
    """
    # Step 1: only recovery variant (stop rows are a different regime).
    recovery = [r for r in runs if r.get("variant") == "recovery"]
    total = len(recovery)

    # Step 2: bucket by how the run ended (_recovery_terminated_by is re-derived
    # in score_runs so parser fixes apply without re-running GPU inference).
    by_term: dict[str, list[dict]] = defaultdict(list)
    for r in recovery:
        term = r.get("_recovery_terminated_by", r.get("recovery_terminated_by", "unknown"))
        by_term[term].append(r)

    # Step 3: one summary row per termination bucket.
    rows: list[dict] = []
    for term, recs in sorted(by_term.items()):
        n = len(recs)
        n_gradable = sum(1 for r in recs if r.get("_correct") is not None)
        rows.append(
            {
                "recovery_terminated_by": term,
                "n": n,
                "n_gradable": n_gradable,
                "fraction_of_recovery": _round(n / total, 4) if total else None,
                # _mean skips _correct is None (e.g. extraction_failed → accuracy blank).
                "accuracy": _round(_mean([r["_correct"] for r in recs]), 4),
                "tokens_mean": _round(_mean([float(r.get("tokens_generated", 0) or 0) for r in recs]), 2),
            }
        )

    # Step 4: ALL row = headline numbers across every recovery run.
    n_gradable_all = sum(1 for r in recovery if r.get("_correct") is not None)
    rows.append(
        {
            "recovery_terminated_by": "ALL",
            "n": total,
            "n_gradable": n_gradable_all,
            "fraction_of_recovery": 1.0 if total else None,
            "accuracy": _round(_mean([r["_correct"] for r in recovery]), 4),
            "tokens_mean": _round(_mean([float(r.get("tokens_generated", 0) or 0) for r in recovery]), 2),
        }
    )
    return rows


def summarize_stop_finish_reason(runs: list[dict]) -> tuple[list[dict], dict[str, float | None]]:
    """
    Group stop rows by finish_reason for rei_summary.txt (no CSV).

    Buckets: stop (EOS), length (token cap — warn signal), unknown (legacy/dry-run).
    """
    stop_runs = [r for r in runs if r.get("variant") == "stop"]
    total = len(stop_runs)

    by_reason: dict[str, list[dict]] = defaultdict(list)
    for r in stop_runs:
        reason = r.get("finish_reason")
        bucket = reason if reason in ("stop", "length") else "unknown"
        by_reason[bucket].append(r)

    rows: list[dict] = []
    for reason in ("stop", "length", "unknown"):
        recs = by_reason.get(reason, [])
        n = len(recs)
        n_gradable = sum(1 for r in recs if r.get("_correct") is not None)
        rows.append(
            {
                "finish_reason": reason,
                "n": n,
                "n_gradable": n_gradable,
                "fraction_of_stop": _round(n / total, 4) if total else None,
                "accuracy": _round(_mean([r["_correct"] for r in recs]), 4) if recs else None,
                "tokens_mean": _round(_mean([float(r.get("tokens_generated", 0) or 0) for r in recs]), 2)
                if recs
                else None,
                "prompt_tokens_mean": _round(
                    _mean([float(r["prompt_tokens"]) for r in recs if r.get("prompt_tokens") is not None]),
                    2,
                )
                if any(r.get("prompt_tokens") is not None for r in recs)
                else None,
            }
        )

    def _variant_prompt_mean(variant: str) -> float | None:
        vals = [
            float(r["prompt_tokens"])
            for r in runs
            if r.get("variant") == variant and r.get("prompt_tokens") is not None
        ]
        return _round(_mean(vals), 2) if vals else None

    prompt_means = {
        "stop": _variant_prompt_mean("stop"),
        "recovery": _variant_prompt_mean("recovery"),
    }
    return rows, prompt_means


def summarize_rei_exclusions(
    per_task_rows: list[dict],
    min_denominator_tokens: int,
) -> list[dict]:
    """
    Count per-task REI exclusions for rei_summary.txt (no CSV).

    Per rei_natural / rei_all: eligible tasks, computed REI, and drop reasons
    (missing variant, non-positive token delta, below min_denominator_tokens).
    """
    specs = (
        ("rei_natural", "tokens_recovery_natural_mean", "n_recovery_natural_seeds"),
        ("rei_all", "tokens_recovery_all_mean", "n_recovery_all"),
    )
    results: list[dict] = []
    total = len(per_task_rows)

    for rei_col, tok_rec_col, n_rec_col in specs:
        stats = {
            "rei_variant": rei_col,
            "total_tasks": total,
            "excluded_missing_variant": 0,
            "eligible": 0,
            "computed": 0,
            "excluded_nonpositive_delta": 0,
            "excluded_below_min": 0,
            "excluded_other": 0,
        }
        for row in per_task_rows:
            n_stop = int(row.get("n_stop") or 0)
            n_rec = int(row.get(n_rec_col) or 0)
            if n_stop == 0 or n_rec == 0:
                stats["excluded_missing_variant"] += 1
                continue
            stats["eligible"] += 1
            tok_stop = float(row.get("tokens_stop_mean") or 0)
            tok_rec = float(row.get(tok_rec_col) or 0)
            delta = tok_rec - tok_stop
            if delta <= 0:
                stats["excluded_nonpositive_delta"] += 1
                continue
            if 0 < delta < min_denominator_tokens:
                stats["excluded_below_min"] += 1
                continue
            rei_val = row.get(rei_col)
            if rei_val is not None and not (isinstance(rei_val, float) and math.isnan(rei_val)):
                stats["computed"] += 1
            else:
                stats["excluded_other"] += 1
        results.append(stats)
    return results


# ---------------------------------------------------------------------------
# Ungradable breakdown (why _correct is None) -> ungradable.csv
# ---------------------------------------------------------------------------


def analyze_ungradable(runs: list[dict]) -> list[dict]:
    """
    Count ungradable runs by (variant, _ungradable_reason) for results/ungradable.csv.

    Lets us (1) rule out parser weakness as a cause and (2) document in the results
    why each ungradable run is ungradable. The 'no_answer_natural' reason is the
    only parser-suspect bucket (model ended on its own, yet nothing parsed); the
    summary flags it explicitly. truncated / empty_output / missing_gold are not
    parser problems.
    """
    ungradable = [r for r in runs if r.get("_correct") is None]
    total = len(ungradable)

    by_key: dict[tuple[str, str], int] = defaultdict(int)
    for r in ungradable:
        variant = r.get("variant", "unknown")
        reason = r.get("_ungradable_reason", "unknown")
        by_key[(variant, reason)] += 1

    rows: list[dict] = [
        {
            "variant": variant,
            "reason": reason,
            "n": n,
            "fraction_of_ungradable": _round(n / total, 4) if total else None,
            "parser_suspect": reason == "no_answer_natural",
        }
        for (variant, reason), n in sorted(by_key.items())
    ]
    rows.append(
        {
            "variant": "ALL",
            "reason": "ALL",
            "n": total,
            "fraction_of_ungradable": 1.0 if total else None,
            "parser_suspect": False,
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Significance tests (paired t-tests on per-task accuracies)
# ---------------------------------------------------------------------------
# Thesis reporting only: "Is the accuracy difference statistically significant?"
# Input is results_per_task.csv rows (one accuracy per task x design cell).
# Does NOT test REI, recovery rate, or tokens — only acc_* columns.
# Output: results/significance_tests.csv (not a pipeline gate).


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """
    Paired (related-samples) t-test: same tasks measured under two conditions.

    Lists a and b must be aligned by position — pair i is the same task under
    condition A (a[i]) vs condition B (b[i]). Uses scipy.stats.ttest_rel.

    Returns (t_stat, p_value). Low p_value → difference unlikely due to chance.
    Returns (nan, nan) when n < 2 valid pairs or all paired differences are
    identical (zero variance → t-statistic undefined).

    Example: a = stop accuracies [0.0, 0.0, 1.0], b = recovery [1.0, 0.0, 1.0]
    tests whether recovery systematically beats stop across those three tasks.
    """
    # Keep only rows where both conditions have a gradable accuracy for this task.
    pairs = [
        (x, y)
        for x, y in zip(a, b)
        if x is not None
        and y is not None
        and not (isinstance(x, float) and math.isnan(x))
        and not (isinstance(y, float) and math.isnan(y))
    ]
    if len(pairs) < 2:
        return float("nan"), float("nan")
    xa = np.array([p[0] for p in pairs], dtype=float)
    xb = np.array([p[1] for p in pairs], dtype=float)
    # Degenerate: zero-variance differences (all equal, incl. all-zero) -> t-test
    # is undefined / infinite. Return NaN rather than emit +/-inf.
    if np.std(xb - xa) == 0.0:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ttest_rel(xb, xa): H0 = mean(xb - xa) == 0; tests if b differs from a.
        result = scipy_stats.ttest_rel(xb, xa)
    return float(result.statistic), float(result.pvalue)


def _format_skip_detail(counts: dict[str, int]) -> str:
    """Semicolon-separated reason counts for significance_tests.csv / summary."""
    return ";".join(f"{reason}={n}" for reason, n in sorted(counts.items()) if n > 0)


def _build_skip_summary(col_a: str, col_b: str, stats: dict[str, Any]) -> str:
    """One-line human-readable skip explanation for rei_summary.txt."""
    if not stats.get("n_skipped"):
        return ""
    parts: list[str] = []
    if stats["n_skip_missing_a"]:
        detail = stats.get("skip_missing_a_detail") or ""
        parts.append(f"missing {col_a}" + (f" ({detail})" if detail else ""))
    if stats["n_skip_missing_b"]:
        detail = stats.get("skip_missing_b_detail") or ""
        parts.append(f"missing {col_b}" + (f" ({detail})" if detail else ""))
    return "; ".join(parts)


def _paired_arrays_and_skips(
    per_task_rows: list[dict],
    col_a: str,
    col_b: str,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """
    Extract aligned accuracy pairs and count why rows were skipped.

    Each row is one task x (model, domain, error_type, k, length_level) cell.
    Rows with either column missing are excluded from the t-test; skip counts
    and per-column missing_reason tallies are returned for reporting.
    """
    a: list[float] = []
    b: list[float] = []
    skip_a_reasons: dict[str, int] = defaultdict(int)
    skip_b_reasons: dict[str, int] = defaultdict(int)
    n_skip_a = 0
    n_skip_b = 0
    # Which per-task column explains why this accuracy is missing (for reporting).
    reason_a_key = _ACC_MISSING_REASON_COL.get(col_a)
    reason_b_key = _ACC_MISSING_REASON_COL.get(col_b)

    for r in per_task_rows:
        a_ok = r.get(col_a) is not None
        b_ok = r.get(col_b) is not None
        # Both accuracies present -> usable paired observation for the t-test.
        if a_ok and b_ok:
            a.append(r[col_a])
            b.append(r[col_b])
            continue
        # Otherwise the task is skipped; tally why (per side, by missing_reason).
        # Note: a row missing BOTH columns increments both counters.
        if not a_ok:
            n_skip_a += 1
            reason = r.get(reason_a_key) if reason_a_key else None
            if reason:
                skip_a_reasons[str(reason)] += 1
        if not b_ok:
            n_skip_b += 1
            reason = r.get(reason_b_key) if reason_b_key else None
            if reason:
                skip_b_reasons[str(reason)] += 1

    n_eligible = len(per_task_rows)
    return a, b, {
        "n_eligible": n_eligible,
        "n_skipped": n_eligible - len(a),
        "n_skip_missing_a": n_skip_a,
        "n_skip_missing_b": n_skip_b,
        "skip_missing_a_detail": _format_skip_detail(skip_a_reasons),
        "skip_missing_b_detail": _format_skip_detail(skip_b_reasons),
    }


def run_significance_tests(per_task_rows: list[dict]) -> list[dict]:
    """
    Run three paired t-tests on per-task accuracies; write significance_tests.csv.

    Each comparison asks a distinct thesis question (accuracy only):

      recovery_natural_vs_stop — Does recovery beat stop? (REI-primary split:
        only recovery rows that ended naturally, not token_limit / failed parse.)
      recovery_all_vs_stop      — Same, but all recovery rows (transparency).
      clean_vs_stop             — Is stop worse than the unperturbed baseline?
        (injection damage: acc_clean from Script 1 vs acc_stop after injection.)

    Run twice per comparison:
      scope=overall              — all per-task rows pooled
      scope=error_type=<name>    — slice by incorrect_reasoning / off_topic / ...

    Output columns:
      n_eligible, n_pairs, n_skipped — how many task-cells entered vs paired
      n_skip_missing_a/b — rows missing col_a or col_b (may overlap if both missing)
      skip_missing_a/b_detail — tallies from acc_*_missing_reason on per-task rows
      skip_summary — one-line explanation for rei_summary.txt
      mean_diff  — average (col_b - col_a); positive => second column higher
      t_stat, p_value — from paired_ttest; small p => significant difference
      note       — set when n_pairs < 2 (pilot-scale cells may be too small)

    Example (pilot overall): recovery_natural_vs_stop mean_diff≈+0.47, p≈0
    → recovery accuracy significantly above stop across tasks.
    """
    # col_a is always the reference (denominator side); col_b is what we test above it.
    comparisons = [
        ("recovery_natural_vs_stop", "acc_stop", "acc_recovery_natural"),
        ("recovery_all_vs_stop", "acc_stop", "acc_recovery_all"),
        ("clean_vs_stop", "acc_stop", "acc_clean"),
    ]

    def _emit(scope: str, rows: list[dict]) -> list[dict]:
        out = []
        for name, col_a, col_b in comparisons:
            a, b, stats = _paired_arrays_and_skips(rows, col_a, col_b)
            t_stat, p_value = paired_ttest(a, b)
            # mean_diff = mean(b - a): e.g. acc_recovery - acc_stop per task.
            diffs = [y - x for x, y in zip(a, b)]
            skip_summary = _build_skip_summary(col_a, col_b, stats)
            # Descriptive warning only (never a gate): flags how many task-cells
            # dropped out of this test so small/biased n is visible in the logs.
            if stats["n_skipped"] > 0:
                warnings.warn(
                    f"[SIGNIFICANCE] {scope} {name}: {len(a)}/{stats['n_eligible']} "
                    f"pairs ({stats['n_skipped']} skipped)"
                    + (f" — {skip_summary}" if skip_summary else ""),
                    stacklevel=2,
                )
            out.append(
                {
                    "scope": scope,
                    "comparison": name,
                    "n_eligible": stats["n_eligible"],
                    "n_pairs": len(a),
                    "n_skipped": stats["n_skipped"],
                    "n_skip_missing_a": stats["n_skip_missing_a"],
                    "n_skip_missing_b": stats["n_skip_missing_b"],
                    "skip_missing_a_detail": stats["skip_missing_a_detail"],
                    "skip_missing_b_detail": stats["skip_missing_b_detail"],
                    "skip_summary": skip_summary,
                    "mean_diff": _round(_mean(diffs), 4) if diffs else None,
                    "t_stat": _round(t_stat, 4),
                    "p_value": _round(p_value, 6),
                    "note": "insufficient pairs (n<2)" if len(a) < 2 else "",
                }
            )
        return out

    # All tasks / conditions together.
    results = _emit("overall", per_task_rows)

    # Same three tests repeated per error_type (injection flavour).
    by_et: dict[str, list[dict]] = defaultdict(list)
    for r in per_task_rows:
        by_et[r["error_type"]].append(r)
    for et, rows in sorted(by_et.items()):
        results.extend(_emit(f"error_type={et}", rows))

    return results


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


def _sort_k_values(values) -> list:
    """Sort injection counts numerically when possible."""
    try:
        return sorted(values, key=lambda x: int(x))
    except (TypeError, ValueError):
        return sorted(values)


def _append_metric_breakdown(
    lines: list[str],
    df: pd.DataFrame,
    factor: str,
    metric: str,
    title: str,
    *,
    fmt: str = ".6f",
    pct: bool = False,
) -> None:
    """Append mean(metric) grouped by a design factor (error_type, k, ...)."""
    if factor not in df.columns or metric not in df.columns:
        return
    lines.append(title)
    values = df[factor].dropna().unique()
    if factor == "k":
        values = _sort_k_values(values)
    else:
        values = sorted(values)
    for val in values:
        sub = df[df[factor] == val][metric].dropna()
        if len(sub):
            mean = sub.mean()
            if pct:
                lines.append(f"  {str(val):<24s}: mean={mean:+.2%}  n={len(sub)}")
            else:
                lines.append(f"  {str(val):<24s}: mean={mean:{fmt}}  n={len(sub)}")
    lines.append("")


def write_summary(
    df_condition: pd.DataFrame,
    df_task: pd.DataFrame,
    sanity_rows: list[dict],
    overthinking_rows: list[dict],
    stop_finish_rows: list[dict],
    stop_prompt_means: dict[str, float | None],
    rei_exclusion_rows: list[dict],
    ungradable_rows: list[dict],
    significance_rows: list[dict],
    config: dict,
) -> str:
    """
    Build human-readable rei_summary.txt from all Script 4 outputs.

    Does not recompute metrics — formats existing CSV/analyzer rows for thesis
    reading. Sections (in order):
      1. Headline means over conditions (REI, accuracy drop, recovery rate)
      2. REI sliced by error_type, domain, k, and length_level
      3. Recovery rate by task difficulty; CoT length inflation; harmful REI count
      4. Overthinking, stop finish_reason, sanity, ungradable, significance (overall scope only)
      5. Pilot warning when rei_natural_n_valid < min_valid_tasks (+ per-condition list)
    """
    # Header: experiment label + grading rules the reader needs to interpret numbers.
    lines: list[str] = [
        "=" * 72,
        "REI SUMMARY — Experiment 1: Perturbed Chain-of-Thought",
        "=" * 72,
        "",
        f"Clean accuracy source: baseline_cots.jsonl (no clean variant in post_injection_runs).",
        f"Primary REI uses recovery_terminated_by == 'natural' (rei_natural).",
        f"min_denominator_tokens={config['min_denominator_tokens']} | "
        f"min_recovery_seeds={config['min_recovery_seeds']} | "
        f"min_valid_tasks={config['min_valid_tasks']}",
        "",
    ]

    if rei_exclusion_rows:
        lines.append("REI exclusions (per task, min_denominator_tokens guard):")
        for row in rei_exclusion_rows:
            eligible = row["eligible"]
            computed = row["computed"]
            drop_rate = _round(computed / eligible, 4) if eligible else None
            lines.append(f"  {row['rei_variant']}:")
            lines.append(
                f"    total={row['total_tasks']}  eligible={eligible}  computed={computed}"
                + (f"  computed/eligible={drop_rate}" if drop_rate is not None else "")
            )
            lines.append(
                f"    missing_variant={row['excluded_missing_variant']}  "
                f"nonpositive_delta={row['excluded_nonpositive_delta']}  "
                f"below_min={row['excluded_below_min']}  "
                f"other={row['excluded_other']}"
            )
        lines.append("")

    if df_condition.empty:
        lines.append("No data to summarise.")
        return "\n".join(lines)

    # Section 1: pool all condition rows -> one mean + min/max per headline metric.
    def _summ(col: str, label: str, fmt: str = ".6f") -> None:
        vals = df_condition[col].dropna()
        if len(vals):
            lines.append(f"{label}: mean={vals.mean():{fmt}}  range=[{vals.min():{fmt}}, {vals.max():{fmt}}]  n={len(vals)}")

    _summ("rei_natural", "REI (natural, PRIMARY)")
    _summ("rei_all", "REI (all recovery)")
    _summ("accuracy_drop", "Accuracy Drop", ".4f")
    _summ("recovery_rate_natural", "Recovery Rate (natural)", ".4f")
    lines.append("")

    # Section 2: same primary metric, broken down by design factor.
    lines.append("REI (natural) by error_type:")
    for et in sorted(df_condition["error_type"].dropna().unique()):
        sub = df_condition[df_condition["error_type"] == et]["rei_natural"].dropna()
        if len(sub):
            lines.append(f"  {et:<24s}: mean={sub.mean():.6f}  n_conditions={len(sub)}")
    lines.append("")

    lines.append("REI (natural) by domain:")
    for d in sorted(df_condition["domain"].dropna().unique()):
        sub = df_condition[df_condition["domain"] == d]["rei_natural"].dropna()
        if len(sub):
            lines.append(f"  {d:<24s}: mean={sub.mean():.6f}  n_conditions={len(sub)}")
    lines.append("")

    # Section 2b: injection-count and trace-length axes (proposal §4.2.2).
    _append_metric_breakdown(
        lines, df_condition, "k", "rei_natural", "REI (natural) by k (injection count):"
    )
    _append_metric_breakdown(
        lines, df_condition, "length_level", "rei_natural", "REI (natural) by length_level:"
    )
    _append_metric_breakdown(
        lines, df_task, "k", "recovery_rate_natural", "Recovery Rate (natural) by k:", fmt=".4f"
    )
    _append_metric_breakdown(
        lines,
        df_task,
        "length_level",
        "recovery_rate_natural",
        "Recovery Rate (natural) by length_level:",
        fmt=".4f",
    )

    # Section 3: per-task difficulty slice (finer than condition-level table).
    if "difficulty_bin" in df_task.columns:
        lines.append("Recovery Rate (natural) by difficulty bin (per task-condition):")
        for label in ["easy", "medium", "hard", "unknown"]:
            sub = df_task[df_task["difficulty_bin"] == label]["recovery_rate_natural"].dropna()
            if len(sub):
                lines.append(f"  {label:<10s}: mean={sub.mean():.4f}  n={len(sub)}")
        lines.append("")

    # Section 3b: CoT length inflation vs clean baseline (proposal §4.2.4, Recum-style).
    if "cot_length_inflation_natural" in df_task.columns:
        inf = df_task["cot_length_inflation_natural"].dropna()
        lines.append(
            "CoT length inflation (natural recovery vs baseline word-count proxy):"
        )
        lines.append(
            "  Formula: (tokens_recovery_natural - tokens_baseline) / tokens_baseline"
        )
        if len(inf):
            lines.append(
                f"  overall: mean={inf.mean():+.2%}  "
                f"range=[{inf.min():+.2%}, {inf.max():+.2%}]  n={len(inf)}"
            )
        _append_metric_breakdown(
            lines,
            df_task,
            "k",
            "cot_length_inflation_natural",
            "  by k:",
            pct=True,
        )
        _append_metric_breakdown(
            lines,
            df_task,
            "domain",
            "cot_length_inflation_natural",
            "  by domain:",
            pct=True,
        )

    # Section 3c: negative REI = recovery hurt accuracy (C3 overthinking signal).
    if "recovery_harmed_natural" in df_task.columns and "rei_natural" in df_task.columns:
        valid_rei = df_task["rei_natural"].notna()
        harmed = df_task.loc[valid_rei, "recovery_harmed_natural"].sum()
        n_valid = int(valid_rei.sum())
        lines.append(
            f"Tasks where recovery harmed accuracy (rei_natural < 0): "
            f"{int(harmed)} of {n_valid}"
        )
        lines.append(
            "  (acc_recovery_natural < acc_stop with token delta >= min_denominator_tokens)"
        )
        if n_valid:
            lines.append(f"  fraction harmed: {harmed / n_valid:.2%}")
        lines.append("")

    # Section 4a: copy analyze_overthinking() rows (natural vs token_limit signal).
    lines.append("Overthinking (recovery termination breakdown):")
    for row in overthinking_rows:
        lines.append(
            f"  {row['recovery_terminated_by']:<18s}: n={row['n']:<4} "
            f"n_gradable={row['n_gradable']:<4} "
            f"frac={row['fraction_of_recovery']}  acc={row['accuracy']}  tokens={row['tokens_mean']}"
        )
    lines.append("")

    # Section 4a2: stop finish_reason — length at stop means cap hit, not natural EOS.
    lines.append("Stop finish_reason breakdown:")
    length_frac: float | None = None
    for row in stop_finish_rows:
        frac = row.get("fraction_of_stop")
        acc = row.get("accuracy")
        acc_str = f"{acc:.4f}" if acc is not None else "n/a"
        lines.append(
            f"  {row['finish_reason']:<18s}: n={row['n']:<4} "
            f"frac={frac}  n_gradable={row['n_gradable']:<4} "
            f"acc={acc_str}  tokens={row.get('tokens_mean')}"
        )
        if row["finish_reason"] == "length" and frac is not None:
            length_frac = float(frac)
    if length_frac is not None and length_frac > STOP_LENGTH_WARN_FRACTION:
        lines.append(
            f"  -> WARNING: {length_frac:.1%} of stop runs hit token cap (finish_reason=length). "
            f"Model may be arguing past prefill; stop regime confound for REI. "
            f"Threshold={STOP_LENGTH_WARN_FRACTION:.0%}."
        )
    elif stop_finish_rows:
        lines.append(
            "  -> length fraction within threshold (stop regime likely healthy)."
        )
    stop_pt = stop_prompt_means.get("stop")
    rec_pt = stop_prompt_means.get("recovery")
    if stop_pt is not None or rec_pt is not None:
        lines.append(
            f"Prompt tokens (mean): stop={stop_pt}  recovery={rec_pt}"
        )
    lines.append("")

    # Section 4b: injection worked? stop accuracy should be mostly wrong per error_type.
    lines.append("Sanity checks (corruption rate on stop rows):")
    for row in sanity_rows:
        flag = "  <-- BELOW THRESHOLD" if row["below_threshold"] else ""
        lines.append(
            f"  {row['error_type']:<24s}: corruption_rate={row['corruption_rate']} "
            f"(thresh={row['threshold']}, n={row['n_stop_runs']}, "
            f"ungradable={row['n_ungradable']}){flag}"
        )
    lines.append("")

    # Section 4c: why runs were ungradable; flag parser-suspect bucket for inspection.
    total_ungradable = next((r["n"] for r in ungradable_rows if r["variant"] == "ALL"), 0)
    parser_suspect = sum(r["n"] for r in ungradable_rows if r.get("parser_suspect"))
    lines.append(f"Ungradable runs (no answer parsed): {total_ungradable} total")
    for row in ungradable_rows:
        if row["variant"] == "ALL":
            continue
        flag = "  <-- parser-suspect (inspect raw_output)" if row.get("parser_suspect") else ""
        lines.append(
            f"  {row['variant']:<10s} {row['reason']:<18s}: n={row['n']} "
            f"({row['fraction_of_ungradable']}){flag}"
        )
    if parser_suspect == 0:
        lines.append("  -> 0 parser-suspect runs: parser weakness excluded as a cause.")
    else:
        lines.append(
            f"  -> {parser_suspect} 'no_answer_natural' run(s): parser miss not excluded; inspect these."
        )
    lines.append("")

    # Section 4d: overall paired t-tests only (per error_type rows stay in CSV).
    lines.append("Significance tests (paired t-test, overall):")
    for row in significance_rows:
        if row["scope"] != "overall":
            continue
        pair_str = f"n_pairs={row['n_pairs']}/{row.get('n_eligible', row['n_pairs'])}"
        skip_str = ""
        if row.get("n_skipped"):
            skip_str = f"  ({row['n_skipped']} skipped"
            if row.get("skip_summary"):
                skip_str += f": {row['skip_summary']}"
            skip_str += ")"
        note = f"  [{row['note']}]" if row.get("note") else ""
        lines.append(
            f"  {row['comparison']:<28s}: {pair_str}{skip_str}  "
            f"mean_diff={row['mean_diff']}  t={row['t_stat']}  p={row['p_value']}{note}"
        )
    lines.append("")

    # Section 5: descriptive pilot warning (not a gate) — few gradable REI tasks/condition.
    low_n = df_condition[
        df_condition["rei_natural_n_valid"].notna()
        & (df_condition["rei_natural_n_valid"] > 0)
        & (df_condition["rei_natural_n_valid"] < int(config["min_valid_tasks"]))
    ]
    if not low_n.empty:
        lines.append(
            f"WARNING: {len(low_n)} condition(s) have rei_natural_n_valid < "
            f"min_valid_tasks={config['min_valid_tasks']} (pilot-scale; interpret with care)."
        )
        lines.append("REI validity by condition:")
        for _, row in low_n.sort_values(list(CONDITION_KEYS)).iterrows():
            cell = (
                f"{row['domain']} / {row['error_type']} / k={row['k']} / {row['length_level']}"
            )
            lines.append(f"  {cell:<52s}: rei_natural_n_valid={int(row['rei_natural_n_valid'])}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _normalize_difficulty_bins(value: Any) -> dict:
    """
    Validate difficulty_bins for assign_difficulty().

    Expects {easy: float, hard: float} from script4 (e.g. easy>=0.70, hard<=0.40).
    Falls back to DEFAULT_CONFIG when YAML is missing keys or wrong type.
    """
    if isinstance(value, dict) and "easy" in value and "hard" in value:
        return {"easy": float(value["easy"]), "hard": float(value["hard"])}
    return dict(DEFAULT_CONFIG["difficulty_bins"])


def resolve_config(args: argparse.Namespace) -> dict:
    """
    Build the final Script 4 config dict (called once at start of main()).

    Merge order (later wins):
      1. DEFAULT_CONFIG — code defaults
      2. experiment.yaml script4: — REI thresholds, CI, sanity, paths
      3. experiment.yaml script3: — token caps shared with run_perturbed.py
      4. CLI flags — --dry-run, --input, --baselines, --output-dir
    """
    config = dict(DEFAULT_CONFIG)
    cfg_file = load_yaml(args.config)

    # Layer 2: Script 4 block (evaluation-only settings).
    script4 = cfg_file.get("script4", {}) if isinstance(cfg_file, dict) else {}
    for key in DEFAULT_CONFIG:
        if key in script4 and script4[key] is not None:
            config[key] = script4[key]

    # Layer 3: Script 3 token caps — needed for ungradable truncation labels
    # and recovery_terminated_by re-derivation, not re-declared in script4.
    script3 = cfg_file.get("script3", {}) if isinstance(cfg_file, dict) else {}
    if script3.get("recovery_max_tokens") is not None:
        config["recovery_max_tokens"] = int(script3["recovery_max_tokens"])
    if script3.get("stop_max_new_tokens") is not None:
        config["stop_max_new_tokens"] = int(script3["stop_max_new_tokens"])

    config["difficulty_bins"] = _normalize_difficulty_bins(config.get("difficulty_bins"))
    if not isinstance(config.get("sanity_thresholds"), dict):
        config["sanity_thresholds"] = dict(DEFAULT_CONFIG["sanity_thresholds"])

    # Layer 4: CLI overrides (highest priority).
    if args.dry_run:
        config["input_path"] = DRY_RUN_INPUT
        config["baseline_path"] = DRY_RUN_BASELINES
    if args.input is not None:
        config["input_path"] = str(args.input)
    if args.baselines is not None:
        config["baseline_path"] = str(args.baselines)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    return config


def _resolve_path(path_str: str) -> Path:
    """Turn config path strings into absolute Paths (relative paths -> project ROOT)."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script 4: Evaluate perturbed CoT runs and compute REI.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic files (no GPU/API): "
        f"{DRY_RUN_INPUT} + {DRY_RUN_BASELINES}.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG, help="experiment.yaml path.")
    parser.add_argument("--input", type=Path, default=None, help="Override input_path (post_injection_runs.jsonl).")
    parser.add_argument("--baselines", type=Path, default=None, help="Override baseline_path (baseline_cots.jsonl).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir (default results/).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    input_path = _resolve_path(config["input_path"])
    baseline_path = _resolve_path(config["baseline_path"])
    output_dir = _resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        hint = (
            "Generate it with scripts/make_synthetic_eval_data.py"
            if args.dry_run
            else "Run scripts/run_perturbed.py first, or use --dry-run."
        )
        sys.exit(f"[FAIL] perturbed runs not found: {input_path}\n       {hint}")

    runs = load_runs(input_path)
    baselines = load_runs(baseline_path)
    if not runs:
        sys.exit(f"[FAIL] No run records in {input_path}")
    if not baselines:
        warnings.warn(f"[WARN] No baseline rows in {baseline_path}; clean accuracy will be NaN.", stacklevel=2)

    print(f"[OK] Loaded {len(runs)} perturbed rows from {input_path}")
    print(f"[OK] Loaded {len(baselines)} baseline rows from {baseline_path}")
    print(f"[INFO] difficulty_bins={config['difficulty_bins']} | output_dir={output_dir}")

    score_runs(runs, config)
    clean_map = clean_accuracy_by_task_model(baselines, config)
    baseline_token_map = baseline_tokens_by_task_model(baselines)

    # Layer 1: task × condition cell  ->  results_per_task.csv
    per_task_rows = build_per_task_condition(runs, clean_map, baseline_token_map, config)
    # Layer 2: mean over tasks per CONDITION_KEYS  ->  results_per_condition.csv
    condition_rows = aggregate_condition(per_task_rows, config)
    sanity_rows = run_sanity_checks(runs, config)
    overthinking_rows = analyze_overthinking(runs)
    stop_finish_rows, stop_prompt_means = summarize_stop_finish_reason(runs)
    rei_exclusion_rows = summarize_rei_exclusions(per_task_rows, int(config["min_denominator_tokens"]))
    ungradable_rows = analyze_ungradable(runs)
    significance_rows = run_significance_tests(per_task_rows)

    df_task = pd.DataFrame(per_task_rows)
    df_condition = pd.DataFrame(condition_rows)

    task_path = output_dir / "results_per_task.csv"
    condition_path = output_dir / "results_per_condition.csv"
    sanity_path = output_dir / "sanity_checks.csv"
    overthinking_path = output_dir / "overthinking.csv"
    ungradable_path = output_dir / "ungradable.csv"
    significance_path = output_dir / "significance_tests.csv"
    summary_path = output_dir / "rei_summary.txt"

    df_task.to_csv(task_path, index=False)
    print(f"[OK] {task_path}  ({len(df_task)} rows)")
    df_condition.to_csv(condition_path, index=False)
    print(f"[OK] {condition_path}  ({len(df_condition)} rows)")
    pd.DataFrame(sanity_rows).to_csv(sanity_path, index=False)
    print(f"[OK] {sanity_path}  ({len(sanity_rows)} rows)")
    pd.DataFrame(overthinking_rows).to_csv(overthinking_path, index=False)
    print(f"[OK] {overthinking_path}  ({len(overthinking_rows)} rows)")
    pd.DataFrame(ungradable_rows).to_csv(ungradable_path, index=False)
    print(f"[OK] {ungradable_path}  ({len(ungradable_rows)} rows)")
    pd.DataFrame(significance_rows).to_csv(significance_path, index=False)
    print(f"[OK] {significance_path}  ({len(significance_rows)} rows)")

    summary_text = write_summary(
        df_condition,
        df_task,
        sanity_rows,
        overthinking_rows,
        stop_finish_rows,
        stop_prompt_means,
        rei_exclusion_rows,
        ungradable_rows,
        significance_rows,
        config,
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"[OK] {summary_path}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()


#code for difficuty bins might be redundant, but it's here for reference

