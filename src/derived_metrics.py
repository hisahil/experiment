"""Derived (secondary) metrics for the continuation experiment — Script 4, Phase 2.

Pure, side-effect-free aggregators consumed by scripts/evaluate.py. Everything here
is computed AFTER the GPU run from stored JSONL:

  * post_injection_runs.jsonl  — stop + recovery rows (already scored: _correct is
    the JUDGE-corrected verdict set by evaluate.score_runs()).
  * complete_dialogues.jsonl   — trace_segments / accumulated_trace per dialogue.
  * baseline_cots_continuation.jsonl — clean continuation_steps (inflation denom).
  * experiment_tasks.jsonl     — decoy_answer / gold (joined by task_id).

Design rules (see plan "Mess-Instrumentierung + abgeleitete Metriken"):
  * Correctness vs GOLD comes from _correct (LLM judge). Decoy matching uses
    is_correct() from evaluator.py (SymPy) — NOT the string match in evaluate.py.
  * Recovery-inflation denominator = HF-tokeniser tokens of the clean baseline
    SECOND half (continuation_steps[midpoint:]); never mixed token/word units.
  * Decoy metrics are scoped to error_type == "false_consensus" only.
  * Length-bin analyses are associative (quartiles per model); never causal.

These are SECONDARY/exploratory metrics. The primary headline metrics
(Susceptibility, Recovery Gain, RTER, Persistence) stay in evaluate.py.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import numpy as np

from evaluator import is_correct
from llm_backend import count_tokens

# Same design cell as evaluate.py (defined locally to avoid an import cycle:
# evaluate imports derived_metrics, not the other way round).
CONDITION_KEYS = ("model", "domain", "error_type", "k", "length_level")

FALSE_CONSENSUS = "false_consensus"

# Minimum cell sizes (plan): decoy aggregates need >=10 stop-decoy cases for a CI;
# inflation needs a non-trivial baseline denominator so ratios stay meaningful.
MIN_DECOY_N = 10
MIN_DENOMINATOR_TOKENS = 50
MIN_DENOMINATOR_WORDS = 20


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), ndigits)


def _is_true(correct: Any) -> bool:
    """True only for a gradable, correct run (1.0 / True). None/0.0 -> False."""
    return correct == 1.0 or correct is True


def _gradable(correct: Any) -> bool:
    return correct is not None


def cell_of(row: dict) -> tuple:
    return tuple(row.get(k) for k in CONDITION_KEYS)


def answer_matches(extracted: str | None, reference: str | None, eval_type: str | None) -> bool:
    """Decoy/gold equality via SymPy-aware is_correct (1e-4 tol). None -> False."""
    if extracted is None or reference is None or str(reference).strip() == "":
        return False
    result = is_correct(extracted, str(reference), eval_type or "exact", math_tolerance=1e-4)
    return result is True


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (sum(vals) / len(vals)) if vals else None


def _percentile(values: list[float], q: float) -> float | None:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def _bootstrap_ci_proportion(
    flags: list[bool], n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for a proportion. (None, None) when n < 2."""
    if len(flags) < 2:
        return None, None
    arr = np.asarray([1.0 if f else 0.0 for f in flags], dtype=float)
    rng = np.random.RandomState(seed)
    means = arr[rng.randint(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (1.0 - ci) / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 + ci) / 2.0))
    return lo, hi


# ---------------------------------------------------------------------------
# Joins / lookups
# ---------------------------------------------------------------------------
def index_complete_dialogues(complete: list[dict]) -> dict[str, dict]:
    """dialogue_id -> complete_dialogues record."""
    return {r.get("dialogue_id"): r for r in complete if r.get("dialogue_id")}


def index_tasks(tasks: list[dict]) -> dict[str, dict]:
    return {t.get("task_id"): t for t in tasks if t.get("task_id")}


def pair_runs_by_dialogue(runs: list[dict]) -> dict[str, dict[str, dict]]:
    """dialogue_id -> {"stop": row, "recovery": row}. Last row wins per variant."""
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in runs:
        did = r.get("dialogue_id")
        variant = r.get("variant")
        if did and variant in ("stop", "recovery"):
            pairs[did][variant] = r
    return pairs


def intermediate_tokens_from_complete(complete_rec: dict | None) -> int:
    """Sum of intermediate model-paragraph tokens (trace_segments type=model).

    These are the tokens the model generated BETWEEN injections (k>1). Excludes
    stop/recovery output and the injected text. 0 for k=1 (no intermediates).
    """
    if not complete_rec:
        return 0
    total = 0
    for seg in complete_rec.get("trace_segments") or []:
        if seg.get("type") == "model" and seg.get("tokens") is not None:
            total += int(seg["tokens"])
    return total


# ---------------------------------------------------------------------------
# Baseline remaining-token maps (recovery-inflation denominators)
# ---------------------------------------------------------------------------
_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(model_name: str) -> Any:
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tok = None
    _TOKENIZER_CACHE[model_name] = tok
    return tok


def baseline_remaining_maps(
    baselines: list[dict], *, load_tokenizer: bool = True
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """(token_map, word_map) keyed by (task_id, model) for continuation baselines.

    The denominator is the CLEAN baseline second half — what the model would still
    have written from the injection midpoint onward WITHOUT any injection. Token
    map uses the model's HF tokeniser (variant A, primary); word map uses
    whitespace counts (variant B, secondary). load_tokenizer=False forces the
    word-count fallback for the token map (dry-run / offline tests).
    """
    token_map: dict[tuple[str, str], int] = {}
    word_map: dict[tuple[str, str], int] = {}
    for rec in baselines:
        if not rec.get("verified", True):
            continue
        steps = [s for s in (rec.get("continuation_steps") or []) if s]
        if not steps:
            continue
        midpoint = len(steps) // 2
        remaining = "\n\n".join(steps[midpoint:])
        if not remaining.strip():
            continue
        model = rec.get("model", "")
        key = (rec.get("task_id", ""), model)
        word_map[key] = len(remaining.split())
        tok = _get_tokenizer(model) if load_tokenizer else None
        token_map[key] = count_tokens(tok, remaining)
    return token_map, word_map


# ---------------------------------------------------------------------------
# 2j.1 Decoy anchoring (false_consensus only)
# ---------------------------------------------------------------------------
def build_decoy_anchoring(
    runs: list[dict], tasks_by_id: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """(per_pair_rows, aggregate_rows) for false_consensus decoy adoption/escape.

    Per pair: did the stop / recovery answer match the DECOY (SymPy)?
    Aggregate per design cell:
      stop_decoy_adoption_rate, recovery_decoy_adoption_rate
      decoy_escape_rate            = P(recovery != decoy | stop == decoy)
      decoy_escape_to_correct_rate = P(recovery correct  | stop == decoy)  (stricter)
      decoy_regression_rate        = P(recovery == decoy | stop != decoy)
    Cells with < MIN_DECOY_N stop-decoy cases are descriptive only (no CI).
    """
    pairs = pair_runs_by_dialogue(runs)
    per_pair: list[dict] = []
    by_cell: dict[tuple, list[dict]] = defaultdict(list)

    for did, pr in pairs.items():
        stop = pr.get("stop")
        rec = pr.get("recovery")
        if stop is None or rec is None:
            continue
        if stop.get("error_type") != FALSE_CONSENSUS:
            continue
        task = tasks_by_id.get(stop.get("task_id"), {})
        decoy = stop.get("decoy_answer", task.get("decoy_answer"))
        if decoy is None:
            continue
        eval_type = stop.get("eval_type") or task.get("eval_type")
        stop_decoy = answer_matches(stop.get("extracted_answer"), decoy, eval_type)
        rec_decoy = answer_matches(rec.get("extracted_answer"), decoy, eval_type)
        rec_correct = _is_true(rec.get("_correct"))
        row = {
            "dialogue_id": did,
            **{k: stop.get(k) for k in CONDITION_KEYS},
            "task_id": stop.get("task_id"),
            "decoy_answer": decoy,
            "stop_decoy_match": stop_decoy,
            "recovery_decoy_match": rec_decoy,
            "recovery_correct": rec_correct if _gradable(rec.get("_correct")) else None,
        }
        per_pair.append(row)
        by_cell[cell_of(stop)].append(row)

    aggregate: list[dict] = []
    for cell, rows in sorted(by_cell.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n = len(rows)
        stop_decoy_flags = [r["stop_decoy_match"] for r in rows]
        rec_decoy_flags = [r["recovery_decoy_match"] for r in rows]
        n_stop_decoy = sum(1 for f in stop_decoy_flags if f)
        n_stop_nondecoy = n - n_stop_decoy

        escaped = [r for r in rows if r["stop_decoy_match"] and not r["recovery_decoy_match"]]
        escaped_to_correct = [
            r for r in rows if r["stop_decoy_match"] and r.get("recovery_correct") is True
        ]
        regressed = [r for r in rows if (not r["stop_decoy_match"]) and r["recovery_decoy_match"]]

        agg = dict(zip(CONDITION_KEYS, cell))
        agg.update({
            "n_pairs": n,
            "n_stop_decoy_match": n_stop_decoy,
            "stop_decoy_adoption_rate": _round(n_stop_decoy / n) if n else None,
            "recovery_decoy_adoption_rate": _round(sum(rec_decoy_flags) / n) if n else None,
            "decoy_escape_rate": _round(len(escaped) / n_stop_decoy) if n_stop_decoy else None,
            "decoy_escape_to_correct_rate": (
                _round(len(escaped_to_correct) / n_stop_decoy) if n_stop_decoy else None
            ),
            "decoy_regression_rate": (
                _round(len(regressed) / n_stop_nondecoy) if n_stop_nondecoy else None
            ),
            # Reliable only above the min-n threshold; flag for the thesis.
            "reliable": n_stop_decoy >= MIN_DECOY_N,
        })
        if n_stop_decoy >= MIN_DECOY_N:
            # Escape flag over the stop-decoy subpopulation only.
            stop_decoy_rows = [r for r in rows if r["stop_decoy_match"]]
            escape_flags = [not r["recovery_decoy_match"] for r in stop_decoy_rows]
            lo, hi = _bootstrap_ci_proportion(escape_flags)
            agg["decoy_escape_rate_ci_lo"] = _round(lo)
            agg["decoy_escape_rate_ci_hi"] = _round(hi)
        aggregate.append(agg)
    return per_pair, aggregate


# ---------------------------------------------------------------------------
# 2j.2 Recovery inflation + token economy
# ---------------------------------------------------------------------------
def build_token_economy(
    runs: list[dict],
    complete_by_id: dict[str, dict],
    baseline_token_map: dict[tuple[str, str], int],
    baseline_word_map: dict[tuple[str, str], int],
) -> tuple[list[dict], list[dict]]:
    """(per_dialogue_rows, aggregate_rows) of tokens + recovery-inflation ratios.

    Per dialogue (recovery path):
      numerator   = sum(intermediate model tokens) + recovery.tokens_generated
      denominator = clean baseline second-half tokens (variant A) / words (B)
      recovery_inflation_ratio_tokens = numerator / denom_tokens   (primary)
      recovery_inflation_ratio_words  = rec_words / denom_words     (secondary)
    Ratios with a too-small denominator are dropped (counted as n_below_min).
    Aggregates: mean / median / p90 / p95 / max (token ratios are right-skewed).
    """
    pairs = pair_runs_by_dialogue(runs)
    per_dialogue: list[dict] = []
    by_cell: dict[tuple, list[dict]] = defaultdict(list)

    for did, pr in pairs.items():
        stop = pr.get("stop")
        rec = pr.get("recovery")
        if rec is None:
            continue
        anchor = stop or rec
        comp = complete_by_id.get(did)
        inter_tokens = intermediate_tokens_from_complete(comp)
        rec_tokens = int(rec.get("tokens_generated") or 0)
        stop_tokens = int(stop.get("tokens_generated") or 0) if stop else None
        key = (rec.get("task_id"), rec.get("model"))
        denom_tokens = baseline_token_map.get(key)
        denom_words = baseline_word_map.get(key)
        rec_words = len((rec.get("raw_output") or "").split())

        numerator_tokens = inter_tokens + rec_tokens
        ratio_tokens = None
        if denom_tokens and denom_tokens >= MIN_DENOMINATOR_TOKENS:
            ratio_tokens = numerator_tokens / denom_tokens
        ratio_words = None
        if denom_words and denom_words >= MIN_DENOMINATOR_WORDS:
            ratio_words = rec_words / denom_words

        row = {
            "dialogue_id": did,
            **{k: anchor.get(k) for k in CONDITION_KEYS},
            "task_id": rec.get("task_id"),
            "tokens_stop": stop_tokens,
            "tokens_recovery": rec_tokens,
            "tokens_intermediate": inter_tokens,
            "tokens_recovery_path": numerator_tokens,
            "baseline_remaining_tokens": denom_tokens,
            "baseline_remaining_words": denom_words,
            "recovery_inflation_ratio_tokens": _round(ratio_tokens, 4),
            "recovery_inflation_ratio_words": _round(ratio_words, 4),
            "log_recovery_inflation_ratio": (
                _round(math.log(ratio_tokens), 4) if ratio_tokens and ratio_tokens > 0 else None
            ),
        }
        per_dialogue.append(row)
        by_cell[cell_of(anchor)].append(row)

    aggregate: list[dict] = []
    for cell, rows in sorted(by_cell.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ratios_tok = [r["recovery_inflation_ratio_tokens"] for r in rows]
        ratios_tok = [v for v in ratios_tok if v is not None]
        ratios_wrd = [r["recovery_inflation_ratio_words"] for r in rows if r["recovery_inflation_ratio_words"] is not None]
        agg = dict(zip(CONDITION_KEYS, cell))
        agg.update({
            "n_dialogues": len(rows),
            "tokens_stop_mean": _round(_mean([r["tokens_stop"] for r in rows if r["tokens_stop"] is not None]), 2),
            "tokens_recovery_mean": _round(_mean([r["tokens_recovery"] for r in rows]), 2),
            "tokens_intermediate_mean": _round(_mean([r["tokens_intermediate"] for r in rows]), 2),
            "n_inflation_tokens": len(ratios_tok),
            "n_below_min_denominator": sum(
                1 for r in rows if r["recovery_inflation_ratio_tokens"] is None
            ),
            "recovery_inflation_ratio_tokens_mean": _round(_mean(ratios_tok)),
            "recovery_inflation_ratio_tokens_median": _round(_percentile(ratios_tok, 50)),
            "recovery_inflation_ratio_tokens_p90": _round(_percentile(ratios_tok, 90)),
            "recovery_inflation_ratio_tokens_p95": _round(_percentile(ratios_tok, 95)),
            "recovery_inflation_ratio_tokens_max": _round(max(ratios_tok) if ratios_tok else None),
            "recovery_inflation_ratio_words_mean": _round(_mean(ratios_wrd)),
            "recovery_inflation_ratio_words_median": _round(_percentile(ratios_wrd, 50)),
        })
        aggregate.append(agg)
    return per_dialogue, aggregate


# ---------------------------------------------------------------------------
# stop_run.csv — per-cell stop vs recovery accuracy + tokens (self-contained)
# ---------------------------------------------------------------------------
def build_stop_run(runs: list[dict]) -> list[dict]:
    """Per-cell variant-split accuracy + token means (figure-friendly subset).

    acc_* use the JUDGE-corrected _correct (gradable rows only). Complements
    results_per_condition.csv with a compact stop/recovery view per design cell.
    """
    by_cell: dict[tuple, dict[str, list]] = defaultdict(lambda: {"stop": [], "recovery": []})
    for r in runs:
        v = r.get("variant")
        if v in ("stop", "recovery"):
            by_cell[cell_of(r)][v].append(r)

    rows: list[dict] = []
    for cell, variants in sorted(by_cell.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        stop_rows = variants["stop"]
        rec_rows = variants["recovery"]
        acc_stop = _mean([1.0 if _is_true(r.get("_correct")) else 0.0 for r in stop_rows if _gradable(r.get("_correct"))])
        acc_rec = _mean([1.0 if _is_true(r.get("_correct")) else 0.0 for r in rec_rows if _gradable(r.get("_correct"))])
        row = dict(zip(CONDITION_KEYS, cell))
        row.update({
            "n_stop": len(stop_rows),
            "n_recovery": len(rec_rows),
            "acc_stop": _round(acc_stop),
            "acc_recovery": _round(acc_rec),
            "recovery_gain": _round((acc_rec - acc_stop) if (acc_rec is not None and acc_stop is not None) else None),
            "tokens_stop_mean": _round(_mean([float(r.get("tokens_generated") or 0) for r in stop_rows]), 2),
            "tokens_recovery_mean": _round(_mean([float(r.get("tokens_generated") or 0) for r in rec_rows]), 2),
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 2j.6 Length-bin analyses (associative)
# ---------------------------------------------------------------------------
_PARAGRAPH_SPLIT = re.compile(r"\n\n+")


def build_length_bins(runs: list[dict], n_bins: int = 4) -> list[dict]:
    """Long-format help/harm/accuracy by recovery-length quartile, PER MODEL.

    help_case = stop wrong AND recovery correct   (~ RECOVERED)
    harm_case = stop correct AND recovery wrong    (~ overcorrection)
    Bins are per-model quartiles of recovery.tokens_generated (token_bins) and of
    the recovery paragraph count (paragraph_bins) — models have different length
    distributions. Strictly ASSOCIATIVE (confounded by task difficulty).
    """
    pairs = pair_runs_by_dialogue(runs)
    # Collect paired, gradable dialogues with a length measure, grouped by model.
    by_model: dict[str, list[dict]] = defaultdict(list)
    for did, pr in pairs.items():
        stop = pr.get("stop")
        rec = pr.get("recovery")
        if stop is None or rec is None:
            continue
        if not (_gradable(stop.get("_correct")) and _gradable(rec.get("_correct"))):
            continue
        stop_correct = _is_true(stop.get("_correct"))
        rec_correct = _is_true(rec.get("_correct"))
        rec_tokens = int(rec.get("tokens_generated") or 0)
        paragraphs = len(_PARAGRAPH_SPLIT.split((rec.get("raw_output") or "").strip())) if rec.get("raw_output") else 0
        by_model[rec.get("model")].append({
            "stop_correct": stop_correct,
            "rec_correct": rec_correct,
            "help_case": (not stop_correct) and rec_correct,
            "harm_case": stop_correct and (not rec_correct),
            "rec_tokens": rec_tokens,
            "paragraphs": paragraphs,
        })

    rows: list[dict] = []
    for model, items in sorted(by_model.items(), key=lambda kv: str(kv[0])):
        for bin_type, field in (("token", "rec_tokens"), ("paragraph", "paragraphs")):
            values = [it[field] for it in items]
            edges = _quantile_edges(values, n_bins)
            if edges is None:
                continue
            for bin_index in range(n_bins):
                lo = edges[bin_index]
                hi = edges[bin_index + 1]
                # Right-closed bins; the last bin includes the max.
                if bin_index < n_bins - 1:
                    members = [it for it in items if lo <= it[field] < hi]
                else:
                    members = [it for it in items if lo <= it[field] <= hi]
                if not members:
                    continue
                n = len(members)
                stop_wrong = [it for it in members if not it["stop_correct"]]
                stop_correct = [it for it in members if it["stop_correct"]]
                rec_acc = _mean([1.0 if it["rec_correct"] else 0.0 for it in members])
                help_rate = _mean([1.0 if it["help_case"] else 0.0 for it in stop_wrong]) if stop_wrong else None
                harm_rate = _mean([1.0 if it["harm_case"] else 0.0 for it in stop_correct]) if stop_correct else None
                for metric_name, value, denom in (
                    ("recovery_accuracy_in_bin", rec_acc, n),
                    ("help_rate_stop_error_in_bin", help_rate, len(stop_wrong)),
                    ("harm_rate_stop_correct_in_bin", harm_rate, len(stop_correct)),
                ):
                    rows.append({
                        "model": model,
                        "bin_type": bin_type,
                        "bin_index": bin_index + 1,
                        "bin_lo": _round(lo, 2),
                        "bin_hi": _round(hi, 2),
                        "metric_name": metric_name,
                        "value": _round(value),
                        "n": denom,
                    })
    return rows


def _quantile_edges(values: list[float], n_bins: int) -> list[float] | None:
    vals = [v for v in values if v is not None]
    if len(vals) < n_bins:
        return None
    qs = [i / n_bins for i in range(n_bins + 1)]
    edges = [float(np.quantile(np.asarray(vals, dtype=float), q)) for q in qs]
    # Degenerate distribution (all edges equal) -> no usable bins.
    if edges[0] == edges[-1]:
        return None
    return edges


# ---------------------------------------------------------------------------
# behavioral_markers.csv (exploratory heuristics on recovery raw_output)
# ---------------------------------------------------------------------------
_MARKER_PATTERNS: dict[str, re.Pattern] = {
    "wait": re.compile(r"\bwait\b|let me reconsider|let me re-?check|hold on", re.IGNORECASE),
    "capitulation": re.compile(r"you('| a)re right|i (was )?wrong|i agree|good point|my mistake", re.IGNORECASE),
    "resistance": re.compile(r"however|but actually|that('| i)s incorrect|i disagree|on the contrary|nevertheless", re.IGNORECASE),
}


def _repetition_score(text: str, n: int = 3) -> float | None:
    """Fraction of repeated word n-grams (0 = none, ->1 = highly repetitive)."""
    words = text.split()
    if len(words) < n + 1:
        return None
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return None
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


def build_behavioral_markers(runs: list[dict]) -> list[dict]:
    """Per-cell mean marker counts on recovery raw_output (EXPLORATORY only)."""
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        if r.get("variant") != "recovery":
            continue
        text = r.get("raw_output") or ""
        counts = {name: len(pat.findall(text)) for name, pat in _MARKER_PATTERNS.items()}
        counts["repetition_score"] = _repetition_score(text)
        by_cell[cell_of(r)].append(counts)

    rows: list[dict] = []
    for cell, items in sorted(by_cell.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(CONDITION_KEYS, cell))
        row["n_recovery"] = len(items)
        for name in _MARKER_PATTERNS:
            row[f"{name}_mean"] = _round(_mean([it[name] for it in items]), 3)
        row["repetition_score_mean"] = _round(_mean([it["repetition_score"] for it in items]), 4)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# confidence.csv (logprobs / perplexity proxy)
# ---------------------------------------------------------------------------
def build_confidence(runs: list[dict], tasks_by_id: dict[str, dict]) -> list[dict]:
    """Per-cell mean_logprob for stop/recovery + stop decoy-vs-gold answer mass.

    answer_top_logprobs (stop only) gives the first-token distribution over the
    boxed answer. For false_consensus we report how much probability mass the top
    candidate carries (p_top) — a calibration signal. Decoy/gold token matching at
    the logprob level is left to figures (token strings vary); here we summarise
    confidence (mean_logprob) and top-candidate mass.
    """
    by_cell: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {"stop_lp": [], "rec_lp": [], "stop_ptop": []}
    )
    for r in runs:
        cell = cell_of(r)
        v = r.get("variant")
        lp = r.get("mean_logprob")
        if v == "stop":
            if lp is not None:
                by_cell[cell]["stop_lp"].append(lp)
            top = r.get("answer_top_logprobs")
            if top:
                # Probability mass of the most likely first answer token.
                p_top = math.exp(top[0]["logprob"]) if top[0].get("logprob") is not None else None
                if p_top is not None:
                    by_cell[cell]["stop_ptop"].append(p_top)
        elif v == "recovery" and lp is not None:
            by_cell[cell]["rec_lp"].append(lp)

    rows: list[dict] = []
    for cell, d in sorted(by_cell.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(CONDITION_KEYS, cell))
        row.update({
            "n_stop_logprob": len(d["stop_lp"]),
            "n_recovery_logprob": len(d["rec_lp"]),
            "stop_mean_logprob": _round(_mean(d["stop_lp"]), 4),
            "recovery_mean_logprob": _round(_mean(d["rec_lp"]), 4),
            "stop_answer_p_top_mean": _round(_mean(d["stop_ptop"]), 4),
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 2h cost_throughput.csv
# ---------------------------------------------------------------------------
def build_cost_throughput(runs: list[dict], complete_by_id: dict[str, dict]) -> list[dict]:
    """Per-model timing/throughput. Per-sample latency is APPROXIMATE (approx_*);
    ETA should use dialogue_wall_time_s (real end-to-end per dialogue)."""
    by_model: dict[str, dict[str, list]] = defaultdict(
        lambda: {"batch_wall": [], "tps_batch": [], "approx": [], "tps_sample": []}
    )
    for r in runs:
        m = r.get("model")
        for field, bucket in (
            ("batch_wall_time_s", "batch_wall"),
            ("tokens_per_second_batch", "tps_batch"),
            ("approx_wall_time_s", "approx"),
            ("tokens_per_second_sample_approx", "tps_sample"),
        ):
            v = r.get(field)
            if v is not None:
                by_model[m][bucket].append(float(v))

    dlg_wall: dict[str, list[float]] = defaultdict(list)
    for rec in complete_by_id.values():
        v = rec.get("dialogue_wall_time_s")
        if v is not None:
            dlg_wall[rec.get("model")].append(float(v))

    rows: list[dict] = []
    models = sorted(set(by_model) | set(dlg_wall), key=lambda x: str(x))
    for m in models:
        d = by_model.get(m, {"batch_wall": [], "tps_batch": [], "approx": [], "tps_sample": []})
        rows.append({
            "model": m,
            "tokens_per_second_batch_mean": _round(_mean(d["tps_batch"]), 2),
            "approx_wall_time_s_mean": _round(_mean(d["approx"]), 5),
            "tokens_per_second_sample_approx_mean": _round(_mean(d["tps_sample"]), 2),
            "dialogue_wall_time_s_mean": _round(_mean(dlg_wall.get(m, [])), 3),
            "dialogue_wall_time_s_median": _round(_percentile(dlg_wall.get(m, []), 50), 3),
            "n_dialogues_timed": len(dlg_wall.get(m, [])),
        })
    return rows


# ---------------------------------------------------------------------------
# Top-level orchestrator (called by evaluate.py)
# ---------------------------------------------------------------------------
def compute_all(
    runs: list[dict],
    complete: list[dict],
    baselines: list[dict],
    tasks: list[dict],
    *,
    load_tokenizer: bool = True,
) -> dict[str, list[dict]]:
    """Build every derived-metric table. Returns {csv_stem: rows}.

    runs MUST already be scored (each row carries _correct from
    evaluate.score_runs). Only continuation-mode adversarial rows should be passed.
    """
    complete_by_id = index_complete_dialogues(complete)
    tasks_by_id = index_tasks(tasks)
    token_map, word_map = baseline_remaining_maps(baselines, load_tokenizer=load_tokenizer)

    decoy_pairs, decoy_agg = build_decoy_anchoring(runs, tasks_by_id)
    econ_dialogue, econ_agg = build_token_economy(runs, complete_by_id, token_map, word_map)

    return {
        "stop_run": build_stop_run(runs),
        "token_economy": econ_agg,
        "token_economy_per_dialogue": econ_dialogue,
        "decoy_anchoring": decoy_agg,
        "decoy_anchoring_per_pair": decoy_pairs,
        "length_bins": build_length_bins(runs),
        "behavioral_markers": build_behavioral_markers(runs),
        "confidence": build_confidence(runs, tasks_by_id),
        "cost_throughput": build_cost_throughput(runs, complete_by_id),
    }
