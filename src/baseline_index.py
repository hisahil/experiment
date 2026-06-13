"""Shared verified-baseline indexing for Scripts 3 and 4.

Single source of truth for selecting the verified baseline CoT per task, so
that clean accuracy in Script 4 uses exactly the same baseline that Script 3
injected into (only verified=True rows). Avoids the earlier inconsistency where
Script 4 averaged failed retries into clean accuracy.

Two consumers, two shapes of the same verified set:
  index_verified_baselines() — Script 3: task_id -> full record (CoT to inject).
  clean_accuracy_map()       — Script 4: (task_id, model) -> clean accuracy.

`verified` is set by Script 1 (generate_baselines.py) via a majority-vote
solvability filter (>= 3 of 5 correct + min_steps). This module only reads that
flag; it does not re-run or re-grade the baseline.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

# Optional fallback grader: (extracted, gold, domain, eval_type) -> bool.
# Only used when a record lacks a usable `correct` field (e.g. synthetic data).
Grader = Callable[[Optional[str], Optional[str], str, Optional[str]], bool]


def index_verified_baselines(
    records: list[dict],
    model_name: str,
) -> dict[str, dict]:
    """task_id -> baseline record (verified only, matching model).

    Used by Script 3 to fetch the CoT it injects into. Only verified=True rows
    for the requested model are kept; everything else is skipped.

    On duplicate task_ids the last record in file order wins (newest run).
    """
    out: dict[str, dict] = {}
    for rec in records:
        # Skip unverified rows (failed majority vote / too few steps).
        if not rec.get("verified"):
            continue
        # Keep only the model we are running the experiment with.
        if rec.get("model") != model_name:
            continue
        out[rec["task_id"]] = rec
    return out


def _verified_row_accuracy(rec: dict, grader: Grader | None) -> float:
    """Clean accuracy contribution of a single verified baseline row.

    Verified is correct by definition, so this is 1.0. The stored `correct`
    flag (if present) is honoured first; `grader` is only a fallback for rows
    that lack a usable `correct` field (e.g. synthetic data).

    Data integrity check: a row that is verified=True yet correct=False is a
    methodologically impossible corruption case. It is still passed through as
    0.0 (behaviour unchanged) but raises a warning so it cannot silently distort
    accuracy-drop downstream.
    """
    # Path A: trust the stored `correct` flag when present.
    if rec.get("correct") is not None:
        # verified=True but correct=False should never happen → flag it loudly.
        if rec.get("verified") and not bool(rec["correct"]):
            warnings.warn(
                f"[DATA INTEGRITY] baseline task_id={rec.get('task_id', '?')} "
                f"model={rec.get('model', '?')} has verified=True but correct=False; "
                "passing through clean accuracy 0.0 (should be impossible).",
                stacklevel=2,
            )
        return 1.0 if bool(rec["correct"]) else 0.0
    # Path B: no `correct` field → re-grade with the supplied fallback grader.
    if grader is not None:
        ok = grader(
            rec.get("parsed_answer"),
            rec.get("gold_answer"),
            rec.get("domain", "math"),
            rec.get("eval_type"),
        )
        return 1.0 if ok else 0.0
    # Path C: no flag and no grader → trust `verified` (correct by definition).
    return 1.0


def clean_accuracy_map(
    baselines: list[dict],
    grader: Grader | None = None,
) -> dict[tuple[str, str], float]:
    """(task_id, model) -> clean accuracy, from verified baselines only.

    Only verified=True rows are considered, matching Script 3's injection set.
    For each (task_id, model) the last verified row in file order is used, so
    there is exactly one value per pair (practically 1.0). Tasks without a
    verified baseline are absent from the map (callers treat that as NaN).
    """
    # Step 1: collapse to one verified record per (task_id, model); last wins.
    verified_by_key: dict[tuple[str, str], dict] = {}
    for rec in baselines:
        if not rec.get("verified"):
            continue
        key = (rec.get("task_id", ""), rec.get("model", ""))
        verified_by_key[key] = rec
    # Step 2: map each pair to its clean accuracy (normally 1.0).
    return {key: _verified_row_accuracy(rec, grader) for key, rec in verified_by_key.items()}
