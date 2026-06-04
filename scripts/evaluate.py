#!/usr/bin/env python3
"""
Script 4: Evaluation + REI (Recovery Efficiency Index) analysis.

Reads perturbed_runs.jsonl (Script 3 output) and baseline_cots.jsonl (Script 1
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
  - perturbed_runs.jsonl has variant in {"stop", "recovery"} only. There is NO
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
  python scripts/evaluate.py --input data/perturbed_runs_synthetic.jsonl \
                             --baselines data/baseline_cots_synthetic.jsonl

Outputs (in output_dir, default results/):
  results_per_task.csv        per task-condition metrics + difficulty bin
  results_per_condition.csv   aggregated metrics + CI per condition
  sanity_checks.csv           corruption rates per error_type (warnings, not gates)
  significance_tests.csv      paired t-tests (scipy), n<2 handled gracefully
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

from evaluator import (  # noqa: E402
    extract_domain_answer,
    infer_math_eval_type,
    is_correct,
)

DEFAULT_EXPERIMENT_CONFIG = ROOT / "configs/experiment.yaml"

# Condition = one cell of the factorial design (task_id is added on top for
# the per-task table). Matches the columns emitted by Script 3.
CONDITION_KEYS = ("model", "domain", "error_type", "k", "length_level")

# Defaults mirror configs/experiment.yaml -> script4. output_dir adapted to
# results/ to stay consistent with the rest of the repo.
DEFAULT_CONFIG: dict[str, Any] = {
    "input_path": "data/perturbed_runs.jsonl",
    "baseline_path": "data/baseline_cots.jsonl",
    "output_dir": "results",
    "ci_z": 1.96,
    "min_recovery_seeds": 2,
    "min_denominator_tokens": 10,
    "min_valid_tasks": 60,
    "difficulty_bins": {"easy": 0.70, "hard": 0.40},
    "sanity_thresholds": {
        "incorrect_reasoning": 0.70,
        "off_topic": 0.60,
        "paraphrased_shift": 0.50,
    },
    "math_tolerance": 1e-4,
}

# Dry-run reads these synthetic files (no GPU, no API). Generate with
# scripts/make_synthetic_eval_data.py.
DRY_RUN_INPUT = "data/perturbed_runs_synthetic.jsonl"
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
# Numeric / answer helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    """Parse a numeric answer (commas/$/% stripped). Returns None on failure."""
    if value is None:
        return None
    text = str(value).strip().lower().replace(",", "").replace("$", "").replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def extract_answer(raw_output: str | None, domain: str) -> str | None:
    """
    Lightweight domain-aware extraction, used only as a fallback when a run
    record lacks a pre-computed extracted_answer. Script 3 already writes
    extracted_answer, so this rarely fires.
    """
    if raw_output is None:
        return None
    import re

    text = str(raw_output).strip()
    if domain == "math":
        boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxed:
            return boxed[-1].strip()
    elif domain == "science":
        m = re.search(r"(?:correct\s+answer\s+is\s+)?option\s+([A-Ea-e]|\d+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    elif domain == "code":
        fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fences:
            return fences[-1].strip()
    m = re.search(r"final\s*answer\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(". ")
    return None


def compute_accuracy(
    extracted: str | None,
    gold: str | None,
    domain: str,
    math_tolerance: float,
    eval_type: str | None = None,
) -> bool:
    """
    Correctness of a single extracted answer against gold.

    math: numeric tolerance for GSM8K-style answers; SymPy equivalence for
    symbolic MATH-500 answers when eval_type is symbolic or inferred;
    science/code/other: normalized string equality.
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


def _run_correct(run: dict, math_tolerance: float) -> bool:
    """
    Authoritative correctness for a run record.

    Prefer the stored boolean `correct` written by Script 3 (computed via
    src.evaluator.is_correct). Fall back to recomputation only when absent /
    null (e.g. NO_PARSE rows or synthetic rows without the field).
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
    return compute_accuracy(
        extracted,
        run.get("gold_answer"),
        run.get("domain", "math"),
        math_tolerance,
        eval_type=run.get("eval_type"),
    )


def score_runs(runs: list[dict], config: dict) -> list[dict]:
    """Annotate each run in place with `_correct` (0/1 as float)."""
    tol = float(config["math_tolerance"])
    for run in runs:
        run["_correct"] = 1.0 if _run_correct(run, tol) else 0.0
    return runs


# ---------------------------------------------------------------------------
# Clean accuracy (from baselines — no clean variant in perturbed_runs)
# ---------------------------------------------------------------------------


def clean_accuracy_by_task_model(baselines: list[dict], config: dict) -> dict[tuple[str, str], float]:
    """
    Clean (un-perturbed) accuracy per (task_id, model), derived from Script 1.

    DESIGN CHOICE: perturbed_runs.jsonl has no clean variant, so clean accuracy
    is the mean correctness of baseline_cots.jsonl rows for that (task_id, model).
    Correctness is parsed_answer vs gold_answer under the domain rule (== the
    `verified` flag for verified rows). In the production pipeline Script 3 only
    perturbs tasks that have a verified baseline, so clean accuracy is ~1.0 for
    every perturbed task (and difficulty binning collapses to "easy"). Including
    all baseline rows (verified + retried/excluded) yields fractional clean
    accuracy when multiple baseline attempts/seeds exist, which is what powers
    the difficulty bins.
    """
    tol = float(config["math_tolerance"])
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for b in baselines:
        tid = b.get("task_id", "")
        model = b.get("model", "")
        domain = b.get("domain", "math")
        if "correct" in b and b["correct"] is not None:
            correct = bool(b["correct"])
        else:
            correct = compute_accuracy(
                b.get("parsed_answer"),
                b.get("gold_answer"),
                domain,
                tol,
                eval_type=b.get("eval_type"),
            )
        by_key[(tid, model)].append(1.0 if correct else 0.0)
    return {key: float(np.mean(vals)) for key, vals in by_key.items() if vals}


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _ci_halfwidth(values: list[float], ci_z: float) -> float:
    """Normal-approx CI half-width: z * SEM. NaN when n < 2."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n < 2:
        return float("nan")
    return float(ci_z * np.std(vals, ddof=1) / math.sqrt(n))


def compute_rei(
    acc_recovery: float,
    acc_stop: float,
    tokens_recovery: float,
    tokens_stop: float,
    min_denominator_tokens: int,
) -> float | None:
    """REI = Δaccuracy / Δtokens. None if denominator < min_denominator_tokens."""
    for v in (acc_recovery, acc_stop, tokens_recovery, tokens_stop):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    delta_tokens = tokens_recovery - tokens_stop
    if delta_tokens < min_denominator_tokens:
        return None
    return (acc_recovery - acc_stop) / delta_tokens


def compute_recovery_rate(acc_recovery: float, acc_stop: float, acc_clean: float) -> float | None:
    """Recovery Rate = (Acc_Rec - Acc_Stop) / (Acc_Clean - Acc_Stop). None if denom <= 0."""
    for v in (acc_recovery, acc_stop, acc_clean):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    denom = acc_clean - acc_stop
    if denom <= 0:
        return None
    return (acc_recovery - acc_stop) / denom


def assign_difficulty(clean_acc: float, bins: dict) -> str:
    """easy if acc >= bins['easy']; hard if acc <= bins['hard']; else medium."""
    if clean_acc is None or (isinstance(clean_acc, float) and math.isnan(clean_acc)):
        return "unknown"
    if clean_acc >= bins.get("easy", 0.70):
        return "easy"
    if clean_acc <= bins.get("hard", 0.40):
        return "hard"
    return "medium"


# ---------------------------------------------------------------------------
# Per task-condition table
# ---------------------------------------------------------------------------


def _round(value: float | None, ndigits: int) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), ndigits)


def build_per_task_condition(
    runs: list[dict],
    clean_map: dict[tuple[str, str], float],
    config: dict,
) -> list[dict]:
    """One row per (model, domain, error_type, k, length_level, task_id)."""
    bins = config["difficulty_bins"]
    min_tokens = int(config["min_denominator_tokens"])

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        key = tuple(r.get(col) for col in CONDITION_KEYS) + (r.get("task_id"),)
        groups[key].append(r)

    rows: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        model, domain, error_type, k, length_level, task_id = key

        stop_rows = [r for r in group if r.get("variant") == "stop"]
        rec_all = [r for r in group if r.get("variant") == "recovery"]
        rec_natural = [r for r in rec_all if r.get("recovery_terminated_by") == "natural"]
        rec_token_limit = [r for r in rec_all if r.get("recovery_terminated_by") == "token_limit"]

        acc_stop = _mean([r["_correct"] for r in stop_rows])
        acc_rec_all = _mean([r["_correct"] for r in rec_all])
        acc_rec_nat = _mean([r["_correct"] for r in rec_natural])

        tok_stop = _mean([float(r.get("tokens_generated", 0) or 0) for r in stop_rows])
        tok_rec_all = _mean([float(r.get("tokens_generated", 0) or 0) for r in rec_all])
        tok_rec_nat = _mean([float(r.get("tokens_generated", 0) or 0) for r in rec_natural])

        acc_clean = clean_map.get((task_id, model), float("nan"))
        difficulty = assign_difficulty(acc_clean, bins)

        accuracy_drop = (
            acc_clean - acc_stop
            if not (math.isnan(acc_clean) or math.isnan(acc_stop))
            else None
        )
        rei_natural = compute_rei(acc_rec_nat, acc_stop, tok_rec_nat, tok_stop, min_tokens)
        rei_all = compute_rei(acc_rec_all, acc_stop, tok_rec_all, tok_stop, min_tokens)
        rr_natural = compute_recovery_rate(acc_rec_nat, acc_stop, acc_clean)
        rr_all = compute_recovery_rate(acc_rec_all, acc_stop, acc_clean)

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
                "acc_clean": _round(acc_clean, 4),
                "acc_stop": _round(acc_stop, 4),
                "acc_recovery_natural": _round(acc_rec_nat, 4),
                "acc_recovery_all": _round(acc_rec_all, 4),
                "accuracy_drop": _round(accuracy_drop, 4),
                "recovery_rate_natural": _round(rr_natural, 4),
                "recovery_rate_all": _round(rr_all, 4),
                "rei_natural": _round(rei_natural, 6),
                "rei_all": _round(rei_all, 6),
                "tokens_stop_mean": _round(tok_stop, 2),
                "tokens_recovery_natural_mean": _round(tok_rec_nat, 2),
                "tokens_recovery_all_mean": _round(tok_rec_all, 2),
                "n_stop": len(stop_rows),
                "n_recovery_all": len(rec_all),
                "n_recovery_natural_seeds": len(rec_natural),
                "n_recovery_token_limit": len(rec_token_limit),
                "code_tests_passed": code_passed,
                "code_tests_total": code_total,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Condition-level aggregation
# ---------------------------------------------------------------------------

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
    "tokens_stop_mean",
    "tokens_recovery_natural_mean",
    "tokens_recovery_all_mean",
]


def aggregate_condition(per_task_rows: list[dict], config: dict) -> list[dict]:
    """Aggregate per-task rows to one row per condition, with CIs and warnings."""
    ci_z = float(config["ci_z"])
    min_valid = int(config["min_valid_tasks"])
    min_rec_seeds = int(config["min_recovery_seeds"])

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_task_rows:
        groups[tuple(row[col] for col in CONDITION_KEYS)].append(row)

    out: list[dict] = []
    for key, task_rows in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        agg: dict[str, Any] = dict(zip(CONDITION_KEYS, key))
        agg["n_tasks"] = len(task_rows)

        for col in _AGG_COLS:
            vals = [r[col] for r in task_rows]
            agg[col] = _round(_mean(vals), 6)
            agg[f"{col}_ci"] = _round(_ci_halfwidth(vals, ci_z), 6)

        # tasks contributing a valid (non-null) primary REI
        n_valid_rei = sum(1 for r in task_rows if r["rei_natural"] is not None)
        agg["rei_natural_n_valid"] = n_valid_rei

        # tasks meeting the min recovery-seeds requirement
        n_ok_seeds = sum(1 for r in task_rows if r["n_recovery_natural_seeds"] >= min_rec_seeds)
        agg["n_tasks_min_recovery_seeds"] = n_ok_seeds

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
    """
    thresholds = config["sanity_thresholds"]
    by_et: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        if r.get("variant") == "stop":
            by_et[r.get("error_type", "unknown")].append(r)

    rows: list[dict] = []
    for et, recs in sorted(by_et.items()):
        n = len(recs)
        corrupted = sum(1 for r in recs if r["_correct"] == 0.0)
        corruption_rate = corrupted / n if n else float("nan")
        thresh = thresholds.get(et)
        if thresh is not None and not math.isnan(corruption_rate) and corruption_rate < thresh:
            warnings.warn(
                f"[SANITY] error_type={et}: corruption_rate={corruption_rate:.2%} "
                f"< threshold {thresh:.0%} (n={n})",
                stacklevel=2,
            )
        rows.append(
            {
                "error_type": et,
                "n_stop_runs": n,
                "corruption_rate": _round(corruption_rate, 4),
                "threshold": thresh,
                "below_threshold": (
                    bool(thresh is not None and not math.isnan(corruption_rate) and corruption_rate < thresh)
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Overthinking analysis (token_limit signal)
# ---------------------------------------------------------------------------


def analyze_overthinking(runs: list[dict]) -> list[dict]:
    """
    Breakdown of recovery rows by recovery_terminated_by. token_limit rows that
    spend many tokens yet stay incorrect are the overthinking signal.
    """
    recovery = [r for r in runs if r.get("variant") == "recovery"]
    total = len(recovery)

    by_term: dict[str, list[dict]] = defaultdict(list)
    for r in recovery:
        by_term[r.get("recovery_terminated_by", "unknown")].append(r)

    rows: list[dict] = []
    for term, recs in sorted(by_term.items()):
        n = len(recs)
        rows.append(
            {
                "recovery_terminated_by": term,
                "n": n,
                "fraction_of_recovery": _round(n / total, 4) if total else None,
                "accuracy": _round(_mean([r["_correct"] for r in recs]), 4),
                "tokens_mean": _round(_mean([float(r.get("tokens_generated", 0) or 0) for r in recs]), 2),
            }
        )

    # overall row for convenience
    rows.append(
        {
            "recovery_terminated_by": "ALL",
            "n": total,
            "fraction_of_recovery": 1.0 if total else None,
            "accuracy": _round(_mean([r["_correct"] for r in recovery]), 4),
            "tokens_mean": _round(_mean([float(r.get("tokens_generated", 0) or 0) for r in recovery]), 2),
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Significance tests (paired t-tests; n < 2 handled gracefully)
# ---------------------------------------------------------------------------


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """
    Paired t-test (scipy ttest_rel). Returns (t_stat, p_value).
    Returns (nan, nan) for n < 2 or when variance is degenerate.
    """
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
        result = scipy_stats.ttest_rel(xb, xa)
    return float(result.statistic), float(result.pvalue)


def _paired_arrays(per_task_rows: list[dict], col_a: str, col_b: str) -> tuple[list[float], list[float]]:
    a: list[float] = []
    b: list[float] = []
    for r in per_task_rows:
        if r[col_a] is not None and r[col_b] is not None:
            a.append(r[col_a])
            b.append(r[col_b])
    return a, b


def run_significance_tests(per_task_rows: list[dict]) -> list[dict]:
    """
    Paired t-tests across task-conditions:
      1. recovery_natural vs stop accuracy (does recovery beat stop?)
      2. recovery_all vs stop accuracy (transparency)
      3. clean vs stop accuracy (accuracy-drop significance)
    Each test is also run overall and per error_type.
    """
    comparisons = [
        ("recovery_natural_vs_stop", "acc_stop", "acc_recovery_natural"),
        ("recovery_all_vs_stop", "acc_stop", "acc_recovery_all"),
        ("clean_vs_stop", "acc_stop", "acc_clean"),
    ]

    def _emit(scope: str, rows: list[dict]) -> list[dict]:
        out = []
        for name, col_a, col_b in comparisons:
            a, b = _paired_arrays(rows, col_a, col_b)
            t_stat, p_value = paired_ttest(a, b)
            diffs = [y - x for x, y in zip(a, b)]
            out.append(
                {
                    "scope": scope,
                    "comparison": name,
                    "n_pairs": len(a),
                    "mean_diff": _round(_mean(diffs), 4) if diffs else None,
                    "t_stat": _round(t_stat, 4),
                    "p_value": _round(p_value, 6),
                    "note": "insufficient pairs (n<2)" if len(a) < 2 else "",
                }
            )
        return out

    results = _emit("overall", per_task_rows)

    by_et: dict[str, list[dict]] = defaultdict(list)
    for r in per_task_rows:
        by_et[r["error_type"]].append(r)
    for et, rows in sorted(by_et.items()):
        results.extend(_emit(f"error_type={et}", rows))

    return results


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


def write_summary(
    df_condition: pd.DataFrame,
    df_task: pd.DataFrame,
    sanity_rows: list[dict],
    overthinking_rows: list[dict],
    significance_rows: list[dict],
    config: dict,
) -> str:
    lines: list[str] = [
        "=" * 72,
        "REI SUMMARY — Experiment 1: Perturbed Chain-of-Thought",
        "=" * 72,
        "",
        f"Clean accuracy source: baseline_cots.jsonl (no clean variant in perturbed_runs).",
        f"Primary REI uses recovery_terminated_by == 'natural' (rei_natural).",
        f"min_denominator_tokens={config['min_denominator_tokens']} | "
        f"min_recovery_seeds={config['min_recovery_seeds']} | "
        f"min_valid_tasks={config['min_valid_tasks']}",
        "",
    ]

    if df_condition.empty:
        lines.append("No data to summarise.")
        return "\n".join(lines)

    def _summ(col: str, label: str, fmt: str = ".6f") -> None:
        vals = df_condition[col].dropna()
        if len(vals):
            lines.append(f"{label}: mean={vals.mean():{fmt}}  range=[{vals.min():{fmt}}, {vals.max():{fmt}}]  n={len(vals)}")

    _summ("rei_natural", "REI (natural, PRIMARY)")
    _summ("rei_all", "REI (all recovery)")
    _summ("accuracy_drop", "Accuracy Drop", ".4f")
    _summ("recovery_rate_natural", "Recovery Rate (natural)", ".4f")
    lines.append("")

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

    if "difficulty_bin" in df_task.columns:
        lines.append("Recovery Rate (natural) by difficulty bin (per task-condition):")
        for label in ["easy", "medium", "hard", "unknown"]:
            sub = df_task[df_task["difficulty_bin"] == label]["recovery_rate_natural"].dropna()
            if len(sub):
                lines.append(f"  {label:<10s}: mean={sub.mean():.4f}  n={len(sub)}")
        lines.append("")

    lines.append("Overthinking (recovery termination breakdown):")
    for row in overthinking_rows:
        lines.append(
            f"  {row['recovery_terminated_by']:<18s}: n={row['n']:<4} "
            f"frac={row['fraction_of_recovery']}  acc={row['accuracy']}  tokens={row['tokens_mean']}"
        )
    lines.append("")

    lines.append("Sanity checks (corruption rate on stop rows):")
    for row in sanity_rows:
        flag = "  <-- BELOW THRESHOLD" if row["below_threshold"] else ""
        lines.append(
            f"  {row['error_type']:<24s}: corruption_rate={row['corruption_rate']} "
            f"(thresh={row['threshold']}, n={row['n_stop_runs']}){flag}"
        )
    lines.append("")

    lines.append("Significance tests (paired t-test, overall):")
    for row in significance_rows:
        if row["scope"] != "overall":
            continue
        note = f"  [{row['note']}]" if row["note"] else ""
        lines.append(
            f"  {row['comparison']:<28s}: n_pairs={row['n_pairs']:<3} "
            f"mean_diff={row['mean_diff']}  t={row['t_stat']}  p={row['p_value']}{note}"
        )
    lines.append("")

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
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _normalize_difficulty_bins(value: Any) -> dict:
    """Accept dict {easy, hard} or fall back to default if a list/None is given."""
    if isinstance(value, dict) and "easy" in value and "hard" in value:
        return {"easy": float(value["easy"]), "hard": float(value["hard"])}
    return dict(DEFAULT_CONFIG["difficulty_bins"])


def resolve_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    cfg_file = load_yaml(args.config)
    script4 = cfg_file.get("script4", {}) if isinstance(cfg_file, dict) else {}
    for key in DEFAULT_CONFIG:
        if key in script4 and script4[key] is not None:
            config[key] = script4[key]

    config["difficulty_bins"] = _normalize_difficulty_bins(config.get("difficulty_bins"))
    if not isinstance(config.get("sanity_thresholds"), dict):
        config["sanity_thresholds"] = dict(DEFAULT_CONFIG["sanity_thresholds"])

    # CLI overrides
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
    parser.add_argument("--input", type=Path, default=None, help="Override input_path (perturbed_runs.jsonl).")
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

    per_task_rows = build_per_task_condition(runs, clean_map, config)
    condition_rows = aggregate_condition(per_task_rows, config)
    sanity_rows = run_sanity_checks(runs, config)
    overthinking_rows = analyze_overthinking(runs)
    significance_rows = run_significance_tests(per_task_rows)

    df_task = pd.DataFrame(per_task_rows)
    df_condition = pd.DataFrame(condition_rows)

    task_path = output_dir / "results_per_task.csv"
    condition_path = output_dir / "results_per_condition.csv"
    sanity_path = output_dir / "sanity_checks.csv"
    overthinking_path = output_dir / "overthinking.csv"
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
    pd.DataFrame(significance_rows).to_csv(significance_path, index=False)
    print(f"[OK] {significance_path}  ({len(significance_rows)} rows)")

    summary_text = write_summary(
        df_condition, df_task, sanity_rows, overthinking_rows, significance_rows, config
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"[OK] {summary_path}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
