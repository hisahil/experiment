"""Prefill-anchored First-Commit extraction for Stop outputs.

The Stop prefill opens ``\\boxed{``. The committed answer is the first
brace-balanced value inside that box. Boxes earlier in the reasoning and every
later box or self-correction are ignored. ``raw_output`` is never rewritten.
"""

from __future__ import annotations

from model_family import STOP_PREFILL_MATH_SUFFIX, build_stop_prefill

_BOXED_OPEN = "\\boxed{"


def is_empty_model_output(text: object) -> bool:
    """True when the model produced no visible text."""
    return text is None or not str(text).strip()


def extract_stop_commitment(
    raw_output: object,
    *,
    domain: str = "math",
    eval_type: str | None = None,
    model_name: str | None = None,
    stop_prefill: str | None = None,
) -> dict:
    """Return the anchored Stop commitment and the span a judge may see."""
    raw = "" if raw_output is None else str(raw_output)
    if is_empty_model_output(raw):
        return {
            "answer": None,
            "span": "",
            "anchor_found": False,
            "incomplete": False,
            "empty_output": True,
        }

    prefill = stop_prefill or build_stop_prefill(domain, eval_type)
    anchor_at = raw.find(prefill) if prefill else -1
    if anchor_at >= 0 and prefill.endswith(_BOXED_OPEN):
        open_brace = anchor_at + len(prefill) - 1
    else:
        marker = STOP_PREFILL_MATH_SUFFIX
        marker_at = raw.find(marker)
        if marker_at < 0 or not marker.endswith(_BOXED_OPEN):
            return {
                "answer": None,
                "span": "",
                "anchor_found": False,
                "incomplete": False,
                "empty_output": False,
            }
        anchor_at = marker_at
        open_brace = marker_at + len(marker) - 1

    from math_utils import read_brace_balanced

    inner, end = read_brace_balanced(raw, open_brace)
    if inner is None or end is None:
        return {
            "answer": None,
            "span": raw[anchor_at:],
            "anchor_found": True,
            "incomplete": True,
            "empty_output": False,
        }
    answer = inner.strip()
    return {
        "answer": answer or None,
        "span": raw[anchor_at:end],
        "anchor_found": True,
        "incomplete": False,
        "empty_output": False,
    }


def apply_empty_output(row: dict) -> bool:
    """Mark a blank model output incorrect before any judge call."""
    if not is_empty_model_output(row.get("raw_output")):
        return False
    row["empty_output"] = True
    row["extracted_answer"] = None
    row["correct"] = False
    row["verifier_score"] = 0
    row["judge_output"] = "Incorrect"
    row["judge_fallback"] = False
    row["judge_skipped_reason"] = "empty_output"
    return True


def judge_visible_output(row: dict) -> str:
    """Text the LLM judge may see.

    Stop rows expose only the anchored first box. Recovery rows keep the full
    output because they have no forced prefill.
    """
    raw = str(row.get("raw_output") or "")
    if row.get("variant") != "stop":
        return raw
    commitment = extract_stop_commitment(
        raw,
        domain=str(row.get("domain") or "math"),
        eval_type=row.get("eval_type"),
        model_name=row.get("model"),
    )
    if commitment["anchor_found"]:
        return commitment["span"]
    return raw
