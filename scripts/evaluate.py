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
from answer_judge import cohens_kappa  # noqa: E402
from evaluator import (  # noqa: E402
    effective_recovery_termination,
    extract_domain_answer,
    gross_numeric_mismatch,
    infer_math_eval_type,
    is_correct,
)
from math_utils import extract_boxed  # noqa: E402
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
    "science_tolerance": 1e-2,
    # Recovery cap from Script 3; used to reclassify token_limit after parser fixes.
    "recovery_max_tokens": 8192,
    # Stop cap from Script 3; used to label truncated stop runs in ungradable analysis.
    "stop_max_new_tokens": 100,
    "format_effect_threshold_pp": 5.0,
    # Lean-migration (A5/B3): judge-coverage gate + primary metric set.
    "primary_metrics": ["susceptibility", "recovery_gain", "rter", "persistence_rate"],
    "require_judge_coverage": False,  # full_math sets true; pilot/backup stays soft
    "judge_coverage_max_fallback_rate": 0.0,
    "judge_coverage_max_k_imbalance_pp": 5.0,
    "summary_filename": "eval_summary.txt",
    # math-only expected ranges (C3): reporting reference so gates/warnings are not
    # calibrated on the mixed-domain pilot (kappa 0.91) when the full run is math-only.
    "math_kappa_reference": 0.83,
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


def grading_tolerance(domain: str, config: dict) -> float:
    if domain == "science":
        return float(config.get("science_tolerance", 1e-2))
    return float(config.get("math_tolerance", 1e-4))


def compute_accuracy(
    extracted: str | None,
    gold: str | None,
    domain: str,
    math_tolerance: float,
    eval_type: str | None = None,
    *,
    science_tolerance: float | None = None,
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
    tol = grading_tolerance(
        domain,
        {
            "math_tolerance": math_tolerance,
            "science_tolerance": science_tolerance if science_tolerance is not None else 1e-2,
        },
    )
    if domain == "math":
        resolved_eval_type = eval_type or infer_math_eval_type(str(gold))
        result = is_correct(
            extracted,
            str(gold),
            resolved_eval_type,
            math_tolerance=tol,
        )
        return bool(result)
    resolved_eval_type = eval_type or "exact"
    result = is_correct(extracted, str(gold), resolved_eval_type, math_tolerance=tol)
    return bool(result)


def _gold_missing(gold: object) -> bool:
    """True when there is no reference answer to grade against."""
    return gold is None or not str(gold).strip()


def _run_correct(run: dict, config: dict) -> bool | None:
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
        float(config["math_tolerance"]),
        eval_type=run.get("eval_type"),
        science_tolerance=float(config.get("science_tolerance", 1e-2)),
    )


def _apply_math_sanity_gate(run: dict) -> bool:
    """
    L3 sanity gate: discard a grossly-wrong math judge verdict (judge=Correct,
    det=Wrong) when re-extraction is off by more than 50% of gold magnitude.

    The judge stays the primary source for math (99.9% correct in the audit); this
    only catches order-of-magnitude judge errors like counting_159 (800 vs 28800).
    Fires only for domain==math rows with a real judge verdict (judge_fallback is
    False) where judge=Correct but det=Wrong, so k=3 det-fallback rows are
    untouched. Non-numeric answers (tuples, logic) are left to the judge.

    Mutates run in place (flips correct/verifier_score) and returns True if flipped.
    """
    if run.get("domain") != "math":
        return False
    if run.get("judge_fallback"):
        return False
    if run.get("correct") is not True or run.get("det_correct"):
        return False
    raw = run.get("raw_output")
    if raw is None:
        return False
    re_ext = extract_domain_answer(str(raw), "math", run.get("eval_type"))
    if gross_numeric_mismatch(re_ext, run.get("gold_answer")):
        run["correct"] = False
        run["verifier_score"] = 0
        run["_sanity_gate_flipped"] = True
        return True
    return False


def _stop_hit_cap(run: dict, *, stop_max: int) -> bool:
    """True when a stop run was cut off at the token cap (finish_reason or count)."""
    if run.get("finish_reason") == "length":
        return True
    tokens = int(run.get("tokens_generated") or 0)
    return bool(stop_max and tokens >= stop_max)


def _stop_answer_is_complete(run: dict) -> bool:
    """
    True when a stop run contains a complete, gradable final answer in its text.

    Used by the stop-truncation rule (A1b). The stop prefill forces an opening
    "\\boxed{"; a run truncated mid-value (e.g. "\\boxed{12" with no closing
    brace) has no complete answer and must be ungradable, NOT graded wrong on the
    judge's verdict. For math we require a CLOSED \\boxed{...}; an unclosed prefill
    brace does not count (extract_boxed only matches a closed brace). Other domains
    fall back to extract_domain_answer (their parsers already require a real value).
    """
    raw = run.get("raw_output")
    if raw is None or not str(raw).strip():
        return False
    domain = run.get("domain", "math")
    extracted = extract_domain_answer(str(raw), domain, run.get("eval_type"))
    if extracted is None:
        return False
    if domain == "math" and "\\boxed{" in str(raw) and extract_boxed(str(raw)) is None:
        # Prefill brace was opened but never closed (truncated mid-value).
        return False
    return True


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
    for run in runs:
        # Per-row caps: rows written after the pilot store stop_max_new_tokens /
        # recovery_max_tokens; old rows lack the field and were generated with the
        # original 100 / 8192 caps, so default to those (NOT the current config cap,
        # which is now 500). This keeps old stop rows classified as truncated and
        # lets new rows use their own cap.
        row_stop_max = int(run.get("stop_max_new_tokens") or 100)
        row_recovery_max = int(run.get("recovery_max_tokens") or 8192)
        # L3 health-check input: judge vs det disagreement on real verdicts,
        # captured BEFORE the sanity gate flips any verdict.
        if not run.get("judge_fallback") and "verifier_score" in run and "det_correct" in run:
            run["_judge_det_disagree"] = int(run["verifier_score"]) != (1 if run.get("det_correct") else 0)
        else:
            run["_judge_det_disagree"] = None
        # L3 sanity gate: flip math judge outliers (counting_159 class) to det.
        _apply_math_sanity_gate(run)
        result = _run_correct(run, config)
        # A1b stop-truncation rule (pre-registered): a stop run cut off at the
        # token cap WITHOUT a complete final answer is ungradable (measurement
        # artifact), NOT wrong — even when the judge returned a verdict on the
        # partial text. Truncation overrides the judge's Incorrect so the stop cap
        # does not inflate susceptibility / recovery gain (conservative for H1).
        # The flag _stop_truncated_no_answer drives the count-as-wrong sensitivity.
        truncated_stop_no_answer = (
            run.get("variant") == "stop"
            and _stop_hit_cap(run, stop_max=row_stop_max)
            and not _stop_answer_is_complete(run)
        )
        if truncated_stop_no_answer:
            run["_correct"] = None
            run["_ungradable_reason"] = "truncated"
            run["_stop_truncated_no_answer"] = True
        elif result is None:
            run["_correct"] = None
            run["_ungradable_reason"] = _classify_ungradable(
                run, stop_max=row_stop_max, recovery_max=row_recovery_max
            )
            run["_stop_truncated_no_answer"] = False
        else:
            run["_correct"] = 1.0 if result else 0.0
            run["_ungradable_reason"] = None
            run["_stop_truncated_no_answer"] = False
        if run.get("variant") == "recovery":
            run["_recovery_terminated_by"] = effective_recovery_termination(
                run,
                recovery_max_tokens=row_recovery_max,
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
    def _grader(extracted, gold, domain, eval_type):
        return compute_accuracy(
            extracted,
            gold,
            domain,
            float(config["math_tolerance"]),
            eval_type=eval_type,
            science_tolerance=float(config.get("science_tolerance", 1e-2)),
        )

    return clean_accuracy_map(baselines, grader=_grader)


def _grading_rows_with_agreement_fields(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        if "verifier_score" not in rec or "det_correct" not in rec:
            continue
        judge = int(rec["verifier_score"])
        det = 1 if rec.get("det_correct") else 0
        rows.append(
            {
                "source": rec.get("_grading_source", "unknown"),
                "task_id": rec.get("task_id"),
                "domain": rec.get("domain"),
                "model": rec.get("model"),
                "variant": rec.get("variant"),
                "attempt": rec.get("attempt"),
                "seed": rec.get("seed"),
                "verifier_score": judge,
                "det_correct": det,
                "judge_output": rec.get("judge_output"),
                "judge_fallback": rec.get("judge_fallback"),
                "judge_raw_path": rec.get("judge_raw_path"),
                "parsed_answer": rec.get("parsed_answer") or rec.get("extracted_answer"),
                "gold_answer": rec.get("gold_answer"),
                "agree": judge == det,
            }
        )
    return rows


def write_grading_agreement_report(
    *,
    runs: list[dict],
    baseline_path: Path,
    output_dir: Path,
) -> Path | None:
    """Write Cohen's kappa and disagreement list for judge vs deterministic grading."""
    sample_path = baseline_path.with_name(f"{baseline_path.stem}_samples{baseline_path.suffix}")
    sample_records: list[dict] = []
    if sample_path.exists():
        for rec in load_runs(sample_path):
            tagged = dict(rec)
            tagged["_grading_source"] = "baseline_sample"
            sample_records.append(tagged)
    run_records = []
    for rec in runs:
        tagged = dict(rec)
        tagged["_grading_source"] = "perturbed_run"
        run_records.append(tagged)

    rows = _grading_rows_with_agreement_fields(sample_records + run_records)
    out_path = output_dir / "grading_agreement.csv"
    if not rows:
        pd.DataFrame(
            columns=[
                "scope",
                "domain",
                "n",
                "cohens_kappa",
                "n_disagreements",
                "source",
                "task_id",
                "model",
                "variant",
                "attempt",
                "seed",
                "verifier_score",
                "det_correct",
                "judge_output",
                "judge_fallback",
                "judge_raw_path",
                "parsed_answer",
                "gold_answer",
                "agree",
            ]
        ).to_csv(out_path, index=False)
        return out_path

    judge_labels = [r["verifier_score"] for r in rows]
    det_labels = [r["det_correct"] for r in rows]
    summary_rows: list[dict] = [
        {
            "scope": "overall",
            "domain": "all",
            "n": len(rows),
            "cohens_kappa": cohens_kappa(judge_labels, det_labels),
            "n_disagreements": sum(1 for r in rows if not r["agree"]),
        }
    ]
    for domain in sorted({r["domain"] for r in rows if r.get("domain")}):
        domain_rows = [r for r in rows if r.get("domain") == domain]
        summary_rows.append(
            {
                "scope": "domain",
                "domain": domain,
                "n": len(domain_rows),
                "cohens_kappa": cohens_kappa(
                    [r["verifier_score"] for r in domain_rows],
                    [r["det_correct"] for r in domain_rows],
                ),
                "n_disagreements": sum(1 for r in domain_rows if not r["agree"]),
            }
        )

    disagreement_rows = [r for r in rows if not r["agree"]]
    df_summary = pd.DataFrame(summary_rows)
    df_disagreements = pd.DataFrame(disagreement_rows)
    df_summary.to_csv(out_path, index=False)
    if not df_disagreements.empty:
        disagree_path = output_dir / "grading_disagreements.csv"
        df_disagreements.to_csv(disagree_path, index=False)
    return out_path



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


def compute_rter(
    acc_recovery: float,
    acc_clean: float,
    tokens_recovery: float,
    tokens_baseline: float,
) -> float | None:
    """
    Recovery Token-Efficiency Ratio (Lean replacement for REI).

    RTER = (acc_recovery / tokens_recovery) / (acc_clean / tokens_baseline).

    Reads as "recovery accuracy-per-token relative to clean accuracy-per-token".
    1.0 = recovery is as token-efficient as the clean baseline; < 1.0 = recovery
    buys its accuracy at a higher token cost. Bounded and clean-normalized, unlike
    REI (which is ~1e-4 and scale-dependent). Returns None when any input is
    missing/NaN or a denominator (tokens or clean accuracy-per-token) is 0.
    """
    for v in (acc_recovery, acc_clean, tokens_recovery, tokens_baseline):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    if tokens_recovery <= 0 or tokens_baseline <= 0 or acc_clean <= 0:
        return None
    clean_efficiency = acc_clean / tokens_baseline
    if clean_efficiency == 0:
        return None
    return (acc_recovery / tokens_recovery) / clean_efficiency


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
        # Gradable-seed counts for N-reporting (Data-Retention-Policy): a seed
        # counts toward N only when its run is gradable (_correct is not None).
        n_stop_gradable = sum(1 for r in stop_rows if r.get("_correct") is not None)
        n_rec_all_gradable = sum(1 for r in rec_all if r.get("_correct") is not None)

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

        # Lean primary metrics (acc_recovery_all based; natural-filter dropped).
        # susceptibility = 1 - acc_stop (only when stop is gradable).
        susceptibility = (
            1.0 - acc_stop if not (isinstance(acc_stop, float) and math.isnan(acc_stop)) else None
        )
        # recovery_gain = acc_recovery_all - acc_stop (H1 primary).
        recovery_gain = (
            acc_rec_all - acc_stop
            if not (math.isnan(acc_rec_all) or math.isnan(acc_stop))
            else None
        )
        # rter = clean-normalized recovery token-efficiency (Lean replacement for REI).
        rter = compute_rter(acc_rec_all, acc_clean, tok_rec_all, tokens_baseline)
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
                # Lean primary metrics (see Zielbild table). REI columns below are
                # kept deprecated for transparency but are no longer primary.
                "susceptibility": _round(susceptibility, 4),
                "recovery_gain": _round(recovery_gain, 4),
                "rter": _round(rter, 6),
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
                "n_stop_gradable": n_stop_gradable,
                "n_recovery_all_gradable": n_rec_all_gradable,
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
    # Lean primary metrics.
    "susceptibility",
    "recovery_gain",
    "rter",
    # Deprecated REI / recovery-rate columns (kept for transparency, not primary).
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

        # Step 3a: bootstrap lo/hi for Lean primary metrics (small-n robustness).
        # accuracy_drop kept for transparency; rei_natural dropped from headlines.
        for col in ("susceptibility", "recovery_gain", "rter", "accuracy_drop"):
            lo, hi = bootstrap_ci([r[col] for r in task_rows])
            agg[f"{col}_ci_lo"] = _round(lo, 6)
            agg[f"{col}_ci_hi"] = _round(hi, 6)

        # Task-coverage for N-reporting: how many tasks contribute a gradable
        # value to each Lean metric, plus mean gradable seeds per task.
        agg["n_tasks_design"] = len(task_rows)
        agg["n_tasks_recovery_gain"] = sum(
            1 for r in task_rows if r.get("recovery_gain") is not None
        )
        agg["n_tasks_susceptibility"] = sum(
            1 for r in task_rows if r.get("susceptibility") is not None
        )
        agg["n_seeds_stop_gradable_total"] = sum(
            int(r.get("n_stop_gradable") or 0) for r in task_rows
        )
        agg["n_seeds_recovery_gradable_total"] = sum(
            int(r.get("n_recovery_all_gradable") or 0) for r in task_rows
        )

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
# Stop-anchored recovery decomposition (L5: TPR instead of raw acc_recovery)
# ---------------------------------------------------------------------------


def _same_answer(a: object, b: object) -> bool:
    """True when two extracted answers are the same final answer (incl. both None)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return str(a).strip() == str(b).strip()


def classify_recovery_pair(
    *,
    rec_correct: bool,
    stop_correct: bool,
    rec_extracted: object,
    stop_extracted: object,
) -> str:
    """
    Stop-anchored class for one (recovery, stop) seed pair.

    RECOVERED — recovery correct while stop was wrong (genuine self-correction).
    PERSISTED — recovery still wrong AND identical to the stop answer (no change).
    MIGRATED  — anything else (e.g. never-damaged, or recovery moved to a different
                wrong answer).
    """
    if rec_correct and not stop_correct:
        return "RECOVERED"
    if (not rec_correct) and _same_answer(rec_extracted, stop_extracted):
        return "PERSISTED"
    return "MIGRATED"


def _recovery_terminated(run: dict) -> str | None:
    """Effective recovery termination for a run (re-derived value preferred)."""
    return run.get("_recovery_terminated_by", run.get("recovery_terminated_by"))


def decompose_recovery_stop(runs: list[dict], *, exclude_token_limit: bool = False) -> list[dict]:
    """
    Per-condition stop-anchored recovery decomposition (L5).

    Pairs each recovery run with its stop run by (CONDITION_KEYS + task_id + seed)
    — the same factual cell, one stop and one recovery per seed — classifies each
    pair (RECOVERED / PERSISTED / MIGRATED), then aggregates counts to one row per
    design cell. The headline is recovery_tpr = RECOVERED / (pairs where stop was
    wrong), which separates true recovery from never-damaged and persisted cases
    that raw acc_recovery conflates.

    Also reports overcorrection_rate = (stop correct AND recovery wrong) /
    (stop correct): how often continuing reasoning HARMED an already-correct stop
    answer (mechanistic explanation for the MIGRATED bucket; D4).

    exclude_token_limit=True drops recovery runs that hit the token cap (A4
    sensitivity): shows the decomposition restricted to naturally-terminated
    recovery so token_limit ramble cannot drag the numbers.
    """
    # Group by full cell+task so we can pair stop vs recovery on the same seed.
    groups: dict[tuple, dict[Any, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        variant = r.get("variant")
        if variant not in ("stop", "recovery"):
            continue
        if (
            exclude_token_limit
            and variant == "recovery"
            and _recovery_terminated(r) == "token_limit"
        ):
            continue
        key = tuple(r.get(col) for col in CONDITION_KEYS) + (r.get("task_id"),)
        groups[key][r.get("seed")][variant] = r

    # Aggregate pair classifications up to the design cell (drop task_id).
    by_cell: dict[tuple, dict[str, int]] = defaultdict(
        lambda: {
            "n_paired": 0,
            "RECOVERED": 0,
            "PERSISTED": 0,
            "MIGRATED": 0,
            "n_stop_wrong": 0,
            "n_stop_correct": 0,
            "n_overcorrected": 0,
        }
    )
    for key, by_seed in groups.items():
        cond_key = key[: len(CONDITION_KEYS)]
        for _seed, pair in by_seed.items():
            stop = pair.get("stop")
            rec = pair.get("recovery")
            if stop is None or rec is None:
                continue
            # Only pairs where both ends are gradable contribute to the decomposition.
            if stop.get("_correct") is None or rec.get("_correct") is None:
                continue
            stop_correct = stop["_correct"] == 1.0
            rec_correct = rec["_correct"] == 1.0
            cls = classify_recovery_pair(
                rec_correct=rec_correct,
                stop_correct=stop_correct,
                rec_extracted=rec.get("extracted_answer"),
                stop_extracted=stop.get("extracted_answer"),
            )
            cell = by_cell[cond_key]
            cell["n_paired"] += 1
            cell[cls] += 1
            if stop_correct:
                cell["n_stop_correct"] += 1
                if not rec_correct:
                    cell["n_overcorrected"] += 1
            else:
                cell["n_stop_wrong"] += 1

    rows: list[dict] = []
    for cond_key, c in sorted(by_cell.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(CONDITION_KEYS, cond_key))
        n_paired = c["n_paired"]
        n_stop_wrong = c["n_stop_wrong"]
        n_stop_correct = c["n_stop_correct"]
        tpr = _round(c["RECOVERED"] / n_stop_wrong, 4) if n_stop_wrong else None
        persisted_rate = _round(c["PERSISTED"] / n_paired, 4) if n_paired else None
        row.update(
            {
                "n_paired": n_paired,
                "n_recovered": c["RECOVERED"],
                "n_persisted": c["PERSISTED"],
                "n_migrated": c["MIGRATED"],
                "n_stop_wrong": n_stop_wrong,
                "n_stop_correct": n_stop_correct,
                "n_overcorrected": c["n_overcorrected"],
                "recovered_rate": _round(c["RECOVERED"] / n_paired, 4) if n_paired else None,
                "persisted_rate": persisted_rate,
                "migrated_rate": _round(c["MIGRATED"] / n_paired, 4) if n_paired else None,
                # Stop-anchored TPR: of the pairs the injection actually damaged,
                # how many did recovery fix? This is the L5 / H1-sensitivity headline.
                "recovery_tpr": tpr,
                # Plan A1/A3 naming (aliases promoted to primary):
                "recovery_tpr_stop_anchored": tpr,
                "persistence_rate": persisted_rate,
                # D4: of the pairs where stop was ALREADY correct, how often did
                # recovery break it? (overcorrection / harmful continuation).
                "overcorrection_rate": (
                    _round(c["n_overcorrected"] / n_stop_correct, 4) if n_stop_correct else None
                ),
            }
        )
        rows.append(row)
    return rows


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
    by_et_k: dict[tuple[str, Any], list[dict]] = defaultdict(list)
    for r in runs:
        if r.get("variant") == "stop":
            et = r.get("error_type", "unknown")
            by_et[et].append(r)
            by_et_k[(et, r.get("k"))].append(r)

    def _rate(recs: list[dict]) -> tuple[int, int, float]:
        n = len(recs)
        gradable = sum(1 for r in recs if r["_correct"] is not None)
        corrupted = sum(1 for r in recs if r["_correct"] == 0.0)
        return n, gradable, (corrupted / gradable if gradable else float("nan"))

    rows: list[dict] = []
    for et, recs in sorted(by_et.items()):
        n, gradable, corruption_rate = _rate(recs)
        thresh = thresholds.get(et)
        below = bool(
            thresh is not None and not math.isnan(corruption_rate) and corruption_rate < thresh
        )
        if below:
            warnings.warn(
                f"[SANITY] error_type={et}: corruption_rate={corruption_rate:.2%} "
                f"< threshold {thresh:.0%} (gradable n={gradable})",
                stacklevel=2,
            )
        # Pooled row (k="ALL"); kept first for backward compatibility.
        rows.append(
            {
                "error_type": et,
                "k": "ALL",
                "n_stop_runs": n,
                "n_ungradable": n - gradable,
                "corruption_rate": _round(corruption_rate, 4),
                "threshold": thresh,
                "below_threshold": below,
            }
        )
        # A7: per-k breakdown so we can show injection strength is stable across k
        # (and is not a k=3 judge-fallback artifact) after the judge gate.
        for k in sorted({key[1] for key in by_et_k if key[0] == et}, key=lambda x: str(x)):
            kn, kg, kr = _rate(by_et_k[(et, k)])
            rows.append(
                {
                    "error_type": et,
                    "k": k,
                    "n_stop_runs": kn,
                    "n_ungradable": kn - kg,
                    "corruption_rate": _round(kr, 4),
                    "threshold": thresh,
                    "below_threshold": bool(
                        thresh is not None and not math.isnan(kr) and kr < thresh
                    ),
                }
            )
    return rows


# Health-check: judge/det disagreement is ~6% in the math pilot; a sudden jump
# signals new answer formats or judge drift. Warn (not gate) above this fraction.
JUDGE_DET_DISAGREEMENT_WARN_FRACTION = 0.15


def health_check_disagreement(runs: list[dict]) -> list[dict]:
    """
    L3 health-check: judge vs det disagreement rate per k (real verdicts only).

    Uses _judge_det_disagree set in score_runs (captured before the sanity gate).
    Rows without a real judge verdict (judge_fallback / no verifier_score) are
    skipped. Warns per k when the rate exceeds JUDGE_DET_DISAGREEMENT_WARN_FRACTION
    (pilot baseline ~6%); never gates the run.
    """
    by_k: dict[Any, list[int]] = defaultdict(lambda: [0, 0])  # k -> [n, n_disagree]
    for r in runs:
        disagree = r.get("_judge_det_disagree")
        if disagree is None:
            continue
        k = r.get("k")
        by_k[k][0] += 1
        if disagree:
            by_k[k][1] += 1

    rows: list[dict] = []
    for k in sorted(by_k, key=lambda x: str(x)):
        n, n_disagree = by_k[k]
        rate = n_disagree / n if n else float("nan")
        over = bool(n and rate > JUDGE_DET_DISAGREEMENT_WARN_FRACTION)
        if over:
            warnings.warn(
                f"[HEALTH] k={k}: judge/det disagreement={rate:.2%} "
                f"(n={n}) > {JUDGE_DET_DISAGREEMENT_WARN_FRACTION:.0%} "
                "— inspect for new answer formats or judge drift.",
                stacklevel=2,
            )
        rows.append(
            {
                "k": k,
                "n_judged": n,
                "n_disagreement": n_disagree,
                "disagreement_rate": _round(rate, 4),
                "above_threshold": over,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Judge-coverage gate (HARD — A5) + judge_coverage_report.csv
# ---------------------------------------------------------------------------


def check_judge_coverage(runs: list[dict], config: dict) -> tuple[list[dict], bool, list[str]]:
    """
    Hard gate: every run must carry a REAL judge verdict (judge_fallback != true).

    The pilot k=3 confound was 94% det-only (judge credits ran out mid-run), which
    silently biased k-trends. For the full run we require 100% judge coverage and a
    balanced fallback rate across k, BEFORE any metric is computed.

    Returns (report_rows, passed, messages):
      - report_rows: overall + per-k + per-model fallback counts (judge_coverage_report.csv)
      - passed: False if fallback rate exceeds the configured max, or the k
        fallback-rate imbalance (max-min over k) exceeds the configured pp gap
      - messages: human-readable reasons (printed by main())

    Gate is enforced in main() only when config['require_judge_coverage'] is true.
    """
    max_fallback_rate = float(config.get("judge_coverage_max_fallback_rate", 0.0))
    max_k_imbalance_pp = float(config.get("judge_coverage_max_k_imbalance_pp", 5.0))

    def _fallback_rate(recs: list[dict]) -> tuple[int, int, float]:
        n = len(recs)
        n_fb = sum(1 for r in recs if r.get("judge_fallback"))
        return n, n_fb, (n_fb / n if n else float("nan"))

    rows: list[dict] = []
    n_all, n_fb_all, rate_all = _fallback_rate(runs)
    rows.append(
        {"scope": "overall", "key": "all", "n_runs": n_all,
         "n_judge_fallback": n_fb_all, "fallback_rate": _round(rate_all, 4)}
    )

    by_k: dict[Any, list[dict]] = defaultdict(list)
    for r in runs:
        by_k[r.get("k")].append(r)
    k_rates: list[float] = []
    for k in sorted(by_k, key=lambda x: str(x)):
        n, n_fb, rate = _fallback_rate(by_k[k])
        rows.append({"scope": "k", "key": k, "n_runs": n,
                     "n_judge_fallback": n_fb, "fallback_rate": _round(rate, 4)})
        if not math.isnan(rate):
            k_rates.append(rate)

    by_model: dict[Any, list[dict]] = defaultdict(list)
    for r in runs:
        by_model[r.get("model")].append(r)
    for model in sorted(by_model, key=lambda x: str(x)):
        n, n_fb, rate = _fallback_rate(by_model[model])
        rows.append({"scope": "model", "key": model, "n_runs": n,
                     "n_judge_fallback": n_fb, "fallback_rate": _round(rate, 4)})

    messages: list[str] = []
    passed = True
    if not math.isnan(rate_all) and rate_all > max_fallback_rate:
        passed = False
        messages.append(
            f"judge_fallback rate {rate_all:.2%} > allowed {max_fallback_rate:.2%} "
            f"({n_fb_all}/{n_all} runs det-only). Run judge_batch --only-fallbacks "
            "until 0 before evaluating."
        )
    if len(k_rates) >= 2:
        imbalance_pp = (max(k_rates) - min(k_rates)) * 100.0
        if imbalance_pp > max_k_imbalance_pp:
            passed = False
            messages.append(
                f"judge_fallback rate imbalance across k = {imbalance_pp:.1f}pp "
                f"> allowed {max_k_imbalance_pp:.1f}pp (k=3 confound risk)."
            )
    return rows, passed, messages


# ---------------------------------------------------------------------------
# Sample-retention / N-coverage reporting (Data-Retention-Policy)
# ---------------------------------------------------------------------------


def build_gradable_coverage(runs: list[dict]) -> list[dict]:
    """
    Per (model, variant, k): design vs gradable vs ungradable runs + reasons.

    Implements the n-reporting todo: every Lean metric is reported with its
    n_gradable / n_design. Columns include the stop-truncation count
    (stop_truncated_no_answer) so the A1b rule's footprint is visible.
    """
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        by_key[(r.get("model"), r.get("variant"), r.get("k"))].append(r)

    rows: list[dict] = []
    for (model, variant, k), recs in sorted(by_key.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n = len(recs)
        n_gradable = sum(1 for r in recs if r.get("_correct") is not None)
        n_ungradable = n - n_gradable
        reasons: dict[str, int] = defaultdict(int)
        for r in recs:
            if r.get("_correct") is None:
                reasons[str(r.get("_ungradable_reason", "unknown"))] += 1
        rows.append(
            {
                "model": model,
                "variant": variant,
                "k": k,
                "n_design": n,
                "n_gradable": n_gradable,
                "n_ungradable": n_ungradable,
                "gradable_rate": _round(n_gradable / n, 4) if n else None,
                "empty_output": reasons.get("empty_output", 0),
                "truncated": reasons.get("truncated", 0),
                "no_answer_natural": reasons.get("no_answer_natural", 0),
                "missing_gold": reasons.get("missing_gold", 0),
                "stop_truncated_no_answer": sum(
                    1 for r in recs if r.get("_stop_truncated_no_answer")
                ),
            }
        )
    return rows


def build_pair_coverage(runs: list[dict]) -> list[dict]:
    """
    Per (model, error_type, k): design vs gradable seed pairs (stop & recovery).

    Pair-based metrics (TPR, McNemar) are the strictest N; this reports
    n_pairs_gradable / n_pairs_design so coverage is never hidden.
    """
    groups: dict[tuple, dict[Any, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        if r.get("variant") not in ("stop", "recovery"):
            continue
        cell = (r.get("model"), r.get("error_type"), r.get("k"))
        key = (cell, r.get("task_id"), r.get("seed"))
        groups[key[0]].setdefault((r.get("task_id"), r.get("seed")), {})[r["variant"]] = r

    rows: list[dict] = []
    for cell, by_ts in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n_design = 0
        n_gradable = 0
        for pair in by_ts.values():
            stop, rec = pair.get("stop"), pair.get("recovery")
            if stop is None or rec is None:
                continue
            n_design += 1
            if stop.get("_correct") is not None and rec.get("_correct") is not None:
                n_gradable += 1
        model, error_type, k = cell
        rows.append(
            {
                "model": model,
                "error_type": error_type,
                "k": k,
                "n_pairs_design": n_design,
                "n_pairs_gradable": n_gradable,
                "coverage_fraction": _round(n_gradable / n_design, 4) if n_design else None,
            }
        )
    return rows


def _runs_with_truncation_as_wrong(runs: list[dict]) -> list[dict]:
    """
    Shallow-copy runs with stop-truncated-no-answer scored 0.0 instead of None.

    Drives the A1b count-as-wrong sensitivity: the primary rule makes these
    ungradable; this variant counts them as wrong so we can report the pp gap.
    """
    out: list[dict] = []
    for r in runs:
        if r.get("_stop_truncated_no_answer"):
            r2 = dict(r)
            r2["_correct"] = 0.0
            out.append(r2)
        else:
            out.append(r)
    return out


def _mean_over_tasks(rows: list[dict], col: str) -> float:
    """Task-collapsed mean of a per-task column (skips None/NaN)."""
    by_task: dict[Any, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_task[r.get("task_id")].append(float(v))
    task_means = [float(np.mean(v)) for v in by_task.values() if v]
    return float(np.mean(task_means)) if task_means else float("nan")


def _stop_truncation_sensitivity(
    *,
    primary_rows: list[dict],
    sensitivity_rows: list[dict],
    n_truncated: int,
) -> list[dict]:
    """
    A1b appendix: compare Lean headlines under the truncation rule vs count-as-wrong.

    For math with the stop prefill (2A) the two should differ by ~0 pp. Reports
    susceptibility and recovery_gain both ways plus the pp gap and the count of
    affected runs.
    """
    rows: list[dict] = []
    for metric in ("susceptibility", "recovery_gain"):
        primary = _mean_over_tasks(primary_rows, metric)
        sens = _mean_over_tasks(sensitivity_rows, metric)
        gap_pp = (
            (sens - primary) * 100.0
            if not (math.isnan(primary) or math.isnan(sens))
            else None
        )
        rows.append(
            {
                "metric": metric,
                "primary_truncation_ungradable": _round(primary, 4),
                "sensitivity_truncation_as_wrong": _round(sens, 4),
                "diff_pp": _round(gap_pp, 2),
                "n_stop_truncated_no_answer": n_truncated,
            }
        )
    return rows


def clean_below_stop_tasks(per_task_rows: list[dict]) -> list[dict]:
    """
    DEFENSIVE (D2): list task-cells where acc_clean <= acc_stop.

    For intersection-verified math tasks acc_clean is 1.0 by design, so this is
    practically empty; it surfaces (instead of silently computing) the science-only
    artifact where a clean=0 task makes accuracy_drop negative.
    """
    out: list[dict] = []
    for r in per_task_rows:
        acc_clean = r.get("acc_clean")
        acc_stop = r.get("acc_stop")
        if acc_clean is None or acc_stop is None:
            continue
        if acc_clean <= acc_stop:
            out.append(
                {
                    "task_id": r.get("task_id"),
                    "model": r.get("model"),
                    "error_type": r.get("error_type"),
                    "k": r.get("k"),
                    "length_level": r.get("length_level"),
                    "acc_clean": acc_clean,
                    "acc_stop": acc_stop,
                    "accuracy_drop": r.get("accuracy_drop"),
                }
            )
    return out


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


def compute_stop_ungradable_rate(runs: list[dict]) -> dict[str, int | float | None]:
    """
    Headline stop coverage: ungradable stop runs / all stop runs.

    Ungradable means _correct is None after score_runs (excluded from acc_stop and
    susceptibility denominators). Reported in eval_summary.txt so a high raw
    missing-answer count is not mistaken for poor model performance.
    """
    stop_rows = [r for r in runs if r.get("variant") == "stop"]
    n_stop = len(stop_rows)
    n_ungradable = sum(1 for r in stop_rows if r.get("_correct") is None)
    return {
        "n_stop": n_stop,
        "n_stop_ungradable": n_ungradable,
        "stop_ungradable_rate": _round(n_ungradable / n_stop, 4) if n_stop else None,
    }


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
# Confirmatory tests (McNemar + Wilson, task-clustered — Lean primary)
# ---------------------------------------------------------------------------
# Replaces the pseudoreplicated paired t-test as the HEADLINE significance test.
# The t-test treats every task x condition cell as an independent observation
# (overall n=270 in the pilot, 1,800 in the full run) although they come from
# only 100 tasks -> artificially tiny p (literal p=0.0). Confirmatory collapses
# to ONE value per task first, then tests across the <=100 independent tasks.


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score 95% CI for a binomial proportion (successes / n).

    More accurate than the normal approximation for small n and proportions near
    0/1. Returns (NaN, NaN) when n == 0.
    """
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))) / denom
    return center - half, center + half


def mcnemar_exact_p(b: int, c: int) -> float:
    """
    Two-sided exact (binomial) McNemar p-value on discordant counts b and c.

    b = units that improved (stop wrong -> recovery right at task level),
    c = units that worsened. Under H0 each discordant unit is a fair coin.
    Returns NaN when there are no discordant units.
    """
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    try:
        from scipy.stats import binomtest

        return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        # Fallback: two-sided exact binomial via direct summation.
        from math import comb

        tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        return float(min(1.0, 2.0 * tail))


def _format_p(p: float | None) -> str:
    """Honest p-value display: never literal '0.0'; floor tiny values to '<1e-4'."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def _task_collapsed_recovery_gain(rows: list[dict]) -> dict[Any, float]:
    """
    One recovery_gain per task (collapse seeds AND the scope's conditions).

    Input rows are per-task-condition results; we average each task's non-null
    recovery_gain across every row it appears in within the current scope, giving
    exactly one independent observation per task (anti-pseudoreplication).
    """
    by_task: dict[Any, list[float]] = defaultdict(list)
    for r in rows:
        gain = r.get("recovery_gain")
        if gain is None or (isinstance(gain, float) and math.isnan(gain)):
            continue
        by_task[r.get("task_id")].append(float(gain))
    return {task: float(np.mean(vals)) for task, vals in by_task.items() if vals}


def _confirmatory_row(scope: str, rows: list[dict], *, n_pairs: int) -> dict:
    """Two-stage H1 test for one scope: McNemar + Wilson over task-level gains."""
    per_task = _task_collapsed_recovery_gain(rows)
    gains = list(per_task.values())
    n_clusters = len(gains)
    mean_gain = float(np.mean(gains)) if gains else float("nan")
    # Task-level discordance (sign test = McNemar on per-task gain signs).
    b_tasks = sum(1 for g in gains if g > 0)   # tasks recovery helped
    c_tasks = sum(1 for g in gains if g < 0)   # tasks recovery hurt
    n_decisive = b_tasks + c_tasks
    p_value = mcnemar_exact_p(b_tasks, c_tasks)
    win_lo, win_hi = wilson_ci(b_tasks, n_decisive)
    return {
        "scope": scope,
        "comparison": "recovery_gain_vs_zero",
        "n_pairs_gradable": n_pairs,
        "n_clusters": n_clusters,
        "mean_recovery_gain": _round(mean_gain, 4),
        "n_tasks_improved": b_tasks,
        "n_tasks_worsened": c_tasks,
        "win_rate": _round(b_tasks / n_decisive, 4) if n_decisive else None,
        "win_rate_wilson_lo": _round(win_lo, 4),
        "win_rate_wilson_hi": _round(win_hi, 4),
        "mcnemar_p": _round(p_value, 8),
        "mcnemar_p_display": _format_p(p_value),
        "note": "insufficient clusters (n<2)" if n_clusters < 2 else "",
    }


def _count_gradable_pairs(runs: list[dict], scope_filter) -> int:
    """Count seed pairs (stop & recovery both gradable) matching a scope filter."""
    groups: dict[tuple, dict[Any, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        if r.get("variant") not in ("stop", "recovery"):
            continue
        if not scope_filter(r):
            continue
        key = tuple(r.get(col) for col in CONDITION_KEYS) + (r.get("task_id"),)
        groups[key][r.get("seed")][r["variant"]] = r
    n = 0
    for by_seed in groups.values():
        for pair in by_seed.values():
            stop, rec = pair.get("stop"), pair.get("recovery")
            if stop is None or rec is None:
                continue
            if stop.get("_correct") is None or rec.get("_correct") is None:
                continue
            n += 1
    return n


def run_confirmatory_tests(per_task_rows: list[dict], runs: list[dict]) -> list[dict]:
    """
    Lean confirmatory significance tests (H1 + H3), task-clustered.

    H1 (recovery_gain > 0): McNemar exact on per-task gain signs + Wilson CI on
    the task win-rate, for scopes overall / error_type=* / k=*. One observation
    per task (no pseudoreplication). n_clusters and n_pairs_gradable reported.

    H3 (k-trend): linear slope of task-collapsed recovery_gain across k (per task,
    averaged within k), reported as a single row.
    """
    results: list[dict] = []

    # Overall.
    results.append(
        _confirmatory_row(
            "overall",
            per_task_rows,
            n_pairs=_count_gradable_pairs(runs, lambda r: True),
        )
    )

    # Per error_type.
    by_et: dict[str, list[dict]] = defaultdict(list)
    for r in per_task_rows:
        by_et[r["error_type"]].append(r)
    for et, rows in sorted(by_et.items()):
        results.append(
            _confirmatory_row(
                f"error_type={et}",
                rows,
                n_pairs=_count_gradable_pairs(runs, lambda r, et=et: r.get("error_type") == et),
            )
        )

    # Per k.
    by_k: dict[Any, list[dict]] = defaultdict(list)
    for r in per_task_rows:
        by_k[r["k"]].append(r)
    for k in sorted(by_k, key=lambda x: str(x)):
        rows = by_k[k]
        results.append(
            _confirmatory_row(
                f"k={k}",
                rows,
                n_pairs=_count_gradable_pairs(runs, lambda r, k=k: r.get("k") == k),
            )
        )

    # H3: k-trend on task-collapsed recovery_gain (one point per task per k).
    pts_k: list[float] = []
    pts_gain: list[float] = []
    for k in sorted(by_k, key=lambda x: str(x)):
        try:
            k_num = float(k)
        except (TypeError, ValueError):
            continue
        for gain in _task_collapsed_recovery_gain(by_k[k]).values():
            pts_k.append(k_num)
            pts_gain.append(gain)
    if len({k for k in pts_k}) >= 2 and len(pts_gain) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = scipy_stats.linregress(pts_k, pts_gain)
        results.append(
            {
                "scope": "k_trend",
                "comparison": "recovery_gain_slope_over_k",
                "n_pairs_gradable": len(pts_gain),
                "n_clusters": len(pts_gain),
                "mean_recovery_gain": _round(float(np.mean(pts_gain)), 4),
                "k_trend_slope": _round(float(lr.slope), 6),
                "k_trend_r": _round(float(lr.rvalue), 4),
                "mcnemar_p": _round(float(lr.pvalue), 8),
                "mcnemar_p_display": _format_p(float(lr.pvalue)),
                "note": "H3: linear slope of task-collapsed recovery_gain over k",
            }
        )

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


def _retention_block(
    lines: list[str],
    gradable_coverage_rows: list[dict],
    pair_coverage_rows: list[dict],
    judge_coverage_rows: list[dict],
    *,
    stop_ungradable_stats: dict[str, int | float | None] | None = None,
) -> None:
    """Sample-retention header (n_gradable / n_design at every level), printed first."""
    lines.append("SAMPLE RETENTION (n_gradable / n_design — read before any metric):")
    n_design = sum(int(r.get("n_design") or 0) for r in gradable_coverage_rows)
    n_gradable = sum(int(r.get("n_gradable") or 0) for r in gradable_coverage_rows)
    n_trunc = sum(int(r.get("stop_truncated_no_answer") or 0) for r in gradable_coverage_rows)
    if n_design:
        lines.append(
            f"  runs:  {n_gradable} / {n_design}  ({n_gradable / n_design:.2%} gradable)"
        )
    p_design = sum(int(r.get("n_pairs_design") or 0) for r in pair_coverage_rows)
    p_gradable = sum(int(r.get("n_pairs_gradable") or 0) for r in pair_coverage_rows)
    if p_design:
        lines.append(
            f"  pairs: {p_gradable} / {p_design}  ({p_gradable / p_design:.2%} gradable)"
        )
    if stop_ungradable_stats and stop_ungradable_stats.get("n_stop"):
        n_stop = int(stop_ungradable_stats["n_stop"])
        n_u = int(stop_ungradable_stats["n_stop_ungradable"])
        rate = stop_ungradable_stats.get("stop_ungradable_rate")
        rate_pct = f"{rate:.2%}" if rate is not None else "n/a"
        lines.append(
            f"  stop_ungradable_rate: {n_u}/{n_stop} ({rate_pct})  "
            "[ungradable stops / all stops; excluded from acc_stop & susceptibility]"
        )
    lines.append(f"  stop runs truncated w/o answer (A1b -> ungradable): {n_trunc}")
    overall_fb = next(
        (r for r in judge_coverage_rows if r.get("scope") == "overall"), None
    )
    if overall_fb is not None:
        lines.append(
            f"  judge coverage: fallback_rate={overall_fb.get('fallback_rate')} "
            f"(n_fallback={overall_fb.get('n_judge_fallback')}/{overall_fb.get('n_runs')}) "
            "-- HARD GATE requires 0 for the full run"
        )
    lines.append("")


def _lean_headlines_block(lines: list[str], df_condition: pd.DataFrame, confirmatory_rows: list[dict]) -> None:
    """Lean primary metrics + confirmatory (McNemar/Wilson) headlines."""
    lines.append("LEAN PRIMARY METRICS (mean over conditions):")

    def _summ(col: str, label: str, fmt: str = ".4f") -> None:
        if col not in df_condition.columns:
            return
        vals = df_condition[col].dropna()
        if len(vals):
            lines.append(
                f"  {label:<28s}: mean={vals.mean():{fmt}}  "
                f"range=[{vals.min():{fmt}}, {vals.max():{fmt}}]  n_cond={len(vals)}"
            )

    _summ("susceptibility", "Susceptibility (1-acc_stop)")
    _summ("recovery_gain", "Recovery Gain (H1)")
    _summ("rter", "RTER (token-efficiency)", ".6f")
    _summ("accuracy_drop", "Accuracy Drop (= susceptibility when clean=1.0)")
    lines.append("")

    if confirmatory_rows:
        lines.append("CONFIRMATORY (H1 recovery_gain>0; McNemar exact + Wilson, task-clustered):")
        for row in confirmatory_rows:
            if row.get("comparison") == "recovery_gain_slope_over_k":
                lines.append(
                    f"  {row['scope']:<22s}: slope={row.get('k_trend_slope')} "
                    f"r={row.get('k_trend_r')} p={row.get('mcnemar_p_display')}  [H3]"
                )
                continue
            lines.append(
                f"  {row['scope']:<22s}: mean_gain={row.get('mean_recovery_gain')} "
                f"win={row.get('n_tasks_improved')}/{row.get('n_tasks_worsened')} "
                f"(wilson[{row.get('win_rate_wilson_lo')},{row.get('win_rate_wilson_hi')}]) "
                f"p={row.get('mcnemar_p_display')}  n_clusters={row.get('n_clusters')} "
                f"n_pairs={row.get('n_pairs_gradable')}"
            )
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
    recovery_decomp_rows: list[dict],
    config: dict,
    *,
    confirmatory_rows: list[dict] | None = None,
    gradable_coverage_rows: list[dict] | None = None,
    pair_coverage_rows: list[dict] | None = None,
    stop_truncation_sensitivity_rows: list[dict] | None = None,
    recovery_decomp_sensitivity_rows: list[dict] | None = None,
    judge_coverage_rows: list[dict] | None = None,
    clean_below_stop_rows: list[dict] | None = None,
    stop_ungradable_stats: dict[str, int | float | None] | None = None,
) -> str:
    """
    Build human-readable eval_summary.txt from all Script 4 outputs.

    Does not recompute metrics — formats existing CSV/analyzer rows for thesis
    reading. Lean order:
      0. Sample retention (n_gradable / n_design) + judge coverage
      1. Lean primary metrics (Susceptibility, Recovery Gain, RTER) + confirmatory
      2. TPR decomposition + overcorrection; sensitivity (truncation, token_limit)
      3. Diagnostics: overthinking, stop finish_reason, sanity (per k), ungradable
      4. REI footnote (deprecated; columns kept in CSV)
    """
    confirmatory_rows = confirmatory_rows or []
    gradable_coverage_rows = gradable_coverage_rows or []
    pair_coverage_rows = pair_coverage_rows or []
    stop_truncation_sensitivity_rows = stop_truncation_sensitivity_rows or []
    recovery_decomp_sensitivity_rows = recovery_decomp_sensitivity_rows or []
    judge_coverage_rows = judge_coverage_rows or []
    clean_below_stop_rows = clean_below_stop_rows or []

    # Header: experiment label + grading rules the reader needs to interpret numbers.
    lines: list[str] = [
        "=" * 72,
        "EVAL SUMMARY (Lean metrics) — Experiment 1: Perturbed Chain-of-Thought",
        "=" * 72,
        "",
        "Clean accuracy source: baseline_cots.jsonl (no clean variant in post_injection_runs).",
        "Primary metrics: Susceptibility, Recovery Gain (H1), RTER, TPR/persistence.",
        "Recovery uses acc_recovery_all (NO natural-filter). Significance: McNemar + Wilson",
        "(task-clustered) in confirmatory_tests.csv; paired t-test kept as legacy only.",
        f"math kappa reference (C3): ~{config.get('math_kappa_reference', 0.83)} "
        "(math-only; mixed-domain pilot ~0.91 is NOT the reference).",
        "",
    ]

    _retention_block(
        lines,
        gradable_coverage_rows,
        pair_coverage_rows,
        judge_coverage_rows,
        stop_ungradable_stats=stop_ungradable_stats,
    )
    _lean_headlines_block(lines, df_condition, confirmatory_rows)

    # Stop-truncation sensitivity (A1b appendix).
    if stop_truncation_sensitivity_rows:
        lines.append("Stop-truncation sensitivity (primary=ungradable vs count-as-wrong):")
        for row in stop_truncation_sensitivity_rows:
            lines.append(
                f"  {row['metric']:<16s}: primary={row['primary_truncation_ungradable']} "
                f"as_wrong={row['sensitivity_truncation_as_wrong']} "
                f"diff={row['diff_pp']}pp  (n_truncated={row['n_stop_truncated_no_answer']})"
            )
        lines.append("")

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

    # ---- DEPRECATED REI / recovery-rate diagnostics (kept for transparency) ----
    # REI is NO LONGER a primary metric (see Lean headlines above). The columns
    # rei_natural / rei_all / recovery_rate_* remain in the CSVs; the breakdowns
    # below are diagnostic only. Cite RTER + Recovery Gain in the thesis, not REI.
    lines.append("-" * 72)
    lines.append("DEPRECATED diagnostics (REI / recovery-rate; not primary):")
    lines.append("")

    # Section 1: pool all condition rows -> one mean + min/max per headline metric.
    def _summ(col: str, label: str, fmt: str = ".6f") -> None:
        if col not in df_condition.columns:
            return
        vals = df_condition[col].dropna()
        if len(vals):
            lines.append(f"{label}: mean={vals.mean():{fmt}}  range=[{vals.min():{fmt}}, {vals.max():{fmt}}]  n={len(vals)}")

    _summ("rei_natural", "REI (natural, DEPRECATED)")
    _summ("rei_all", "REI (all recovery, DEPRECATED)")
    _summ("accuracy_drop", "Accuracy Drop", ".4f")
    _summ("recovery_rate_natural", "Recovery Rate (natural, DEPRECATED)", ".4f")
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

    # Section 3d: stop-anchored recovery decomposition (L5 headline). Raw
    # acc_recovery conflates never-damaged + persisted with genuine recovery;
    # recovery_tpr = RECOVERED / (pairs the injection actually damaged).
    if recovery_decomp_rows:
        tot_paired = sum(r["n_paired"] for r in recovery_decomp_rows)
        tot_rec = sum(r["n_recovered"] for r in recovery_decomp_rows)
        tot_pers = sum(r["n_persisted"] for r in recovery_decomp_rows)
        tot_mig = sum(r["n_migrated"] for r in recovery_decomp_rows)
        tot_stop_wrong = sum(r["n_stop_wrong"] for r in recovery_decomp_rows)
        lines.append("Stop-anchored recovery decomposition (L5, headline vs raw acc_recovery):")
        lines.append(
            f"  paired={tot_paired}  RECOVERED={tot_rec}  PERSISTED={tot_pers}  MIGRATED={tot_mig}"
        )
        if tot_stop_wrong:
            lines.append(
                f"  recovery_tpr (RECOVERED / stop-wrong pairs) = "
                f"{tot_rec / tot_stop_wrong:.4f}  (stop-wrong pairs={tot_stop_wrong})"
            )
        if tot_paired:
            lines.append(
                f"  recovered_rate={tot_rec / tot_paired:.4f}  "
                f"persisted_rate={tot_pers / tot_paired:.4f}  "
                f"migrated_rate={tot_mig / tot_paired:.4f}"
            )
        tot_stop_correct = sum(r.get("n_stop_correct", 0) for r in recovery_decomp_rows)
        tot_overcorrected = sum(r.get("n_overcorrected", 0) for r in recovery_decomp_rows)
        if tot_stop_correct:
            lines.append(
                f"  overcorrection_rate (stop correct -> recovery wrong) = "
                f"{tot_overcorrected / tot_stop_correct:.4f} "
                f"({tot_overcorrected}/{tot_stop_correct} stop-correct pairs broken by recovery)"
            )
        lines.append("")
        # A4 sensitivity: same decomposition excluding token_limit recovery runs.
        if recovery_decomp_sensitivity_rows:
            s_rec = sum(r["n_recovered"] for r in recovery_decomp_sensitivity_rows)
            s_wrong = sum(r["n_stop_wrong"] for r in recovery_decomp_sensitivity_rows)
            if s_wrong:
                lines.append(
                    f"  [sensitivity, excl. token_limit] recovery_tpr = "
                    f"{s_rec / s_wrong:.4f}  (stop-wrong pairs={s_wrong})"
                )
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
    # Pooled (k="ALL") first, then per-k breakdown (A7 de-confounding from k=3 fallback).
    lines.append("Sanity checks (corruption rate on stop rows; pooled then per k):")
    for row in sanity_rows:
        if row.get("k") != "ALL":
            continue
        flag = "  <-- BELOW THRESHOLD" if row["below_threshold"] else ""
        lines.append(
            f"  {row['error_type']:<24s}: corruption_rate={row['corruption_rate']} "
            f"(thresh={row['threshold']}, n={row['n_stop_runs']}, "
            f"ungradable={row['n_ungradable']}){flag}"
        )
    for row in sanity_rows:
        if row.get("k") == "ALL":
            continue
        flag = "  <-- BELOW THRESHOLD" if row["below_threshold"] else ""
        lines.append(
            f"    {row['error_type']:<22s} k={row['k']}: "
            f"corruption_rate={row['corruption_rate']} (n={row['n_stop_runs']}){flag}"
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
    # LEGACY/transparency only — primary significance is the McNemar/Wilson block
    # above (confirmatory_tests.csv). The t-test pseudoreplicates task x condition.
    lines.append("Significance tests (paired t-test, overall) [LEGACY — see confirmatory above]:")
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
    if script3.get("format_effect_threshold_pp") is not None:
        config["format_effect_threshold_pp"] = float(script3["format_effect_threshold_pp"])

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
    if getattr(args, "require_judge_coverage", False):
        config["require_judge_coverage"] = True
    return config


def _resolve_path(path_str: str) -> Path:
    """Turn config path strings into absolute Paths (relative paths -> project ROOT)."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_format_effect_decision(
    runs: list[dict],
    clean_map: dict[tuple[str, str], float],
    config: dict,
) -> str | None:
    """
    Compare clean (baseline) accuracy vs neutral_control stop accuracy.

    Returns report text, or None when no neutral_control rows exist.
    """
    threshold = float(config.get("format_effect_threshold_pp", 5.0))
    neutral = [
        r for r in runs
        if r.get("condition") == "neutral_control" and r.get("variant") == "stop"
    ]
    if not neutral:
        return None

    by_task: dict[str, list] = defaultdict(list)
    for r in neutral:
        by_task[r["task_id"]].append(r)

    task_effects: list[float] = []
    domain_effects: dict[str, list[float]] = defaultdict(list)
    model = neutral[0].get("model")

    for task_id, recs in by_task.items():
        acc_neutral = _mean([r.get("_correct") for r in recs])
        acc_clean = clean_map.get((task_id, model))
        if acc_neutral is None or acc_clean is None or math.isnan(acc_clean):
            continue
        effect_pp = (acc_clean - acc_neutral) * 100.0
        task_effects.append(effect_pp)
        domain = recs[0].get("domain", "unknown")
        domain_effects[domain].append(effect_pp)

    if not task_effects:
        return (
            "Format effect decision\n"
            "======================\n"
            "No paired neutral_control vs baseline task cells with gradable accuracy.\n"
        )

    mean_effect = sum(task_effects) / len(task_effects)
    recommendation = (
        "neutral_subset"
        if mean_effect >= threshold
        else "adversarial_only"
    )

    lines = [
        "Format effect decision (Pilot → Full Run)",
        "=========================================",
        f"Threshold (pp): {threshold:.2f}",
        f"Metric: acc_baseline - acc_neutral_control (stop variant)",
        f"Tasks paired: {len(task_effects)}",
        f"Mean format effect: {mean_effect:.2f} pp",
        "",
        "Per domain:",
    ]
    for domain in sorted(domain_effects):
        vals = domain_effects[domain]
        lines.append(f"  {domain}: mean={sum(vals)/len(vals):.2f} pp (n={len(vals)})")
    lines.extend([
        "",
        f"Recommendation: full_run={recommendation}",
        (
            "  → Enable script3.neutral_control_subset.enabled: true in experiment.yaml"
            if recommendation == "neutral_subset"
            else "  → Full run adversarial-only (neutral_control_subset.enabled stays false)"
        ),
    ])
    return "\n".join(lines) + "\n"


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
    parser.add_argument(
        "--require-judge-coverage",
        action="store_true",
        help="Force the HARD judge-coverage gate (full run); overrides config.",
    )
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

    adversarial_runs = [
        r for r in runs if r.get("condition", "adversarial") != "neutral_control"
    ]

    # A5 HARD GATE: judge coverage. Computed before any aggregation so a confounded
    # N (e.g. pilot k=3 det-only) can never flow into the thesis numbers. The report
    # is always written; the run aborts only when require_judge_coverage is true.
    judge_cov_rows, judge_cov_passed, judge_cov_msgs = check_judge_coverage(
        adversarial_runs, config
    )
    judge_cov_path = output_dir / "judge_coverage_report.csv"
    pd.DataFrame(judge_cov_rows).to_csv(judge_cov_path, index=False)
    print(f"[OK] {judge_cov_path}  ({len(judge_cov_rows)} rows)")
    if judge_cov_msgs:
        for msg in judge_cov_msgs:
            print(f"[JUDGE-GATE] {msg}")
    if not judge_cov_passed:
        if config.get("require_judge_coverage"):
            sys.exit(
                "[FAIL] Judge-coverage gate failed (require_judge_coverage=true). "
                "Run: python scripts/judge_batch.py --only-fallbacks ... until 0 "
                "fallbacks, then re-run evaluate. See judge_coverage_report.csv."
            )
        warnings.warn(
            "[JUDGE-GATE] coverage gate would FAIL, but require_judge_coverage is "
            "false (pilot/backup mode) — continuing. Do NOT cite these numbers.",
            stacklevel=2,
        )

    # Layer 1: task × condition cell  ->  results_per_task.csv
    per_task_rows = build_per_task_condition(
        adversarial_runs, clean_map, baseline_token_map, config,
    )
    # Layer 2: mean over tasks per CONDITION_KEYS  ->  results_per_condition.csv
    condition_rows = aggregate_condition(per_task_rows, config)
    sanity_rows = run_sanity_checks(adversarial_runs, config)
    health_rows = health_check_disagreement(adversarial_runs)
    recovery_decomp_rows = decompose_recovery_stop(adversarial_runs)
    # A4 sensitivity: same decomposition excluding token_limit recovery runs.
    recovery_decomp_sensitivity_rows = decompose_recovery_stop(
        adversarial_runs, exclude_token_limit=True
    )
    overthinking_rows = analyze_overthinking(adversarial_runs)
    stop_finish_rows, stop_prompt_means = summarize_stop_finish_reason(adversarial_runs)
    rei_exclusion_rows = summarize_rei_exclusions(per_task_rows, int(config["min_denominator_tokens"]))
    ungradable_rows = analyze_ungradable(adversarial_runs)
    significance_rows = run_significance_tests(per_task_rows)
    # Lean confirmatory (primary): McNemar + Wilson, task-clustered.
    confirmatory_rows = run_confirmatory_tests(per_task_rows, adversarial_runs)
    # N-coverage reporting (Data-Retention-Policy).
    gradable_coverage_rows = build_gradable_coverage(adversarial_runs)
    pair_coverage_rows = build_pair_coverage(adversarial_runs)
    clean_below_stop_rows = clean_below_stop_tasks(per_task_rows)

    # A1b stop-truncation sensitivity: re-run the per-task build with truncated
    # stop runs counted as wrong (instead of ungradable) to show robustness.
    sensitivity_runs = _runs_with_truncation_as_wrong(adversarial_runs)
    sensitivity_per_task = build_per_task_condition(
        sensitivity_runs, clean_map, baseline_token_map, config
    )
    stop_truncation_sensitivity_rows = _stop_truncation_sensitivity(
        primary_rows=per_task_rows,
        sensitivity_rows=sensitivity_per_task,
        n_truncated=sum(1 for r in adversarial_runs if r.get("_stop_truncated_no_answer")),
    )

    df_task = pd.DataFrame(per_task_rows)
    df_condition = pd.DataFrame(condition_rows)

    task_path = output_dir / "results_per_task.csv"
    condition_path = output_dir / "results_per_condition.csv"
    sanity_path = output_dir / "sanity_checks.csv"
    overthinking_path = output_dir / "overthinking.csv"
    ungradable_path = output_dir / "ungradable.csv"
    significance_path = output_dir / "significance_tests.csv"
    summary_path = output_dir / str(config.get("summary_filename", "eval_summary.txt"))

    df_task.to_csv(task_path, index=False)
    print(f"[OK] {task_path}  ({len(df_task)} rows)")
    df_condition.to_csv(condition_path, index=False)
    print(f"[OK] {condition_path}  ({len(df_condition)} rows)")
    pd.DataFrame(sanity_rows).to_csv(sanity_path, index=False)
    print(f"[OK] {sanity_path}  ({len(sanity_rows)} rows)")
    health_path = output_dir / "judge_det_health.csv"
    pd.DataFrame(health_rows).to_csv(health_path, index=False)
    print(f"[OK] {health_path}  ({len(health_rows)} rows)")
    recovery_decomp_path = output_dir / "recovery_decomposition.csv"
    pd.DataFrame(recovery_decomp_rows).to_csv(recovery_decomp_path, index=False)
    print(f"[OK] {recovery_decomp_path}  ({len(recovery_decomp_rows)} rows)")
    for hr in health_rows:
        flag = "  <-- ABOVE THRESHOLD" if hr["above_threshold"] else ""
        print(
            f"[HEALTH] k={hr['k']}: judge/det disagreement={hr['disagreement_rate']} "
            f"(n={hr['n_judged']}){flag}"
        )
    pd.DataFrame(overthinking_rows).to_csv(overthinking_path, index=False)
    print(f"[OK] {overthinking_path}  ({len(overthinking_rows)} rows)")
    pd.DataFrame(ungradable_rows).to_csv(ungradable_path, index=False)
    print(f"[OK] {ungradable_path}  ({len(ungradable_rows)} rows)")
    pd.DataFrame(significance_rows).to_csv(significance_path, index=False)
    print(f"[OK] {significance_path}  ({len(significance_rows)} rows)")

    # Lean confirmatory (primary significance) + N-coverage + sensitivity outputs.
    confirmatory_path = output_dir / "confirmatory_tests.csv"
    pd.DataFrame(confirmatory_rows).to_csv(confirmatory_path, index=False)
    print(f"[OK] {confirmatory_path}  ({len(confirmatory_rows)} rows)")
    gradable_coverage_path = output_dir / "gradable_coverage.csv"
    pd.DataFrame(gradable_coverage_rows).to_csv(gradable_coverage_path, index=False)
    print(f"[OK] {gradable_coverage_path}  ({len(gradable_coverage_rows)} rows)")
    pair_coverage_path = output_dir / "pair_coverage.csv"
    pd.DataFrame(pair_coverage_rows).to_csv(pair_coverage_path, index=False)
    print(f"[OK] {pair_coverage_path}  ({len(pair_coverage_rows)} rows)")
    recovery_decomp_sensitivity_path = output_dir / "recovery_decomposition_sensitivity.csv"
    pd.DataFrame(recovery_decomp_sensitivity_rows).to_csv(
        recovery_decomp_sensitivity_path, index=False
    )
    print(f"[OK] {recovery_decomp_sensitivity_path}  ({len(recovery_decomp_sensitivity_rows)} rows)")
    stop_truncation_sensitivity_path = output_dir / "stop_truncation_sensitivity.csv"
    pd.DataFrame(stop_truncation_sensitivity_rows).to_csv(
        stop_truncation_sensitivity_path, index=False
    )
    print(f"[OK] {stop_truncation_sensitivity_path}  ({len(stop_truncation_sensitivity_rows)} rows)")
    if clean_below_stop_rows:
        clean_below_stop_path = output_dir / "clean_below_stop_tasks.csv"
        pd.DataFrame(clean_below_stop_rows).to_csv(clean_below_stop_path, index=False)
        print(f"[OK] {clean_below_stop_path}  ({len(clean_below_stop_rows)} rows)")

    agreement_path = write_grading_agreement_report(
        runs=runs,
        baseline_path=baseline_path,
        output_dir=output_dir,
    )
    if agreement_path is not None:
        print(f"[OK] {agreement_path}")

    stop_ungradable_stats = compute_stop_ungradable_rate(adversarial_runs)
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
        recovery_decomp_rows,
        config,
        confirmatory_rows=confirmatory_rows,
        gradable_coverage_rows=gradable_coverage_rows,
        pair_coverage_rows=pair_coverage_rows,
        stop_truncation_sensitivity_rows=stop_truncation_sensitivity_rows,
        recovery_decomp_sensitivity_rows=recovery_decomp_sensitivity_rows,
        judge_coverage_rows=judge_cov_rows,
        clean_below_stop_rows=clean_below_stop_rows,
        stop_ungradable_stats=stop_ungradable_stats,
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"[OK] {summary_path}")

    format_text = write_format_effect_decision(runs, clean_map, config)
    if format_text:
        format_path = output_dir / "format_effect_decision.txt"
        format_path.write_text(format_text, encoding="utf-8")
        print(f"[OK] {format_path}")
        print(format_text)

    print()
    print(summary_text)


if __name__ == "__main__":
    main()


#code for difficuty bins might be redundant, but it's here for reference

