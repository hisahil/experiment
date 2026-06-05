"""Answer parsing and metric helpers."""

from __future__ import annotations

import re
from typing import Any

from math_utils import extract_boxed


FINAL_ANSWER_RE = re.compile(
    r"final\s*answer\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
SCIENCE_OPTION_RE = re.compile(
    r"(?:correct\s+answer\s+is\s+)?option\s+([A-Ea-e]|\d+)",
    re.IGNORECASE,
)
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
FINAL_NUMERIC_RE = re.compile(
    r"final\s*answer\s*(?:is|:)?\s*\$?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
LOGIC_FINAL_RE = re.compile(r"final\s*answer\s*:?\s*(.+)$", re.IGNORECASE | re.DOTALL)


def extract_final_answer(text: str) -> str | None:
    match = FINAL_ANSWER_RE.search(text.strip())
    if not match:
        return None
    answer = match.group(1).strip()
    # Strip trailing punctuation common in model outputs
    return answer.rstrip(". ")


def _extract_numeric_answer(text: str) -> str | None:
    """Numeric answer for open-ended science (SciBench): prefer the value after
    a 'final answer' marker, else the last number in the text."""
    match = FINAL_NUMERIC_RE.search(text)
    if match:
        return match.group(1)
    nums = NUMERIC_TOKEN_RE.findall(text)
    return nums[-1] if nums else None


def _extract_logic_answer(text: str) -> str | None:
    """Return the assignment block for K&K. Prefer text after a 'final answer'
    marker (to the end), else the whole completion; the logic grader scans it
    for every Name=role / 'Name is a knight/knave' mention (later wins)."""
    match = LOGIC_FINAL_RE.search(text)
    if match:
        return match.group(1).strip()
    return text or None


def extract_domain_answer(text: str, domain: str, eval_type: str | None = None) -> str | None:
    """
    Domain-specific answer extraction for Script 3 stop/recovery outputs.

    eval_type refines extraction within a domain (e.g. science numeric SciBench
    vs science MCQ MMLU). Falls back to the 'Final answer:' line when domain
    patterns do not match.
    """
    stripped = text.strip()
    if domain == "math":
        boxed = extract_boxed(stripped)
        if boxed is not None:
            return boxed
    elif domain == "science":
        if eval_type == "numeric":
            num = _extract_numeric_answer(stripped)
            if num is not None:
                return num
        else:
            match = SCIENCE_OPTION_RE.search(stripped)
            if match:
                return match.group(1).strip()
    elif domain == "code":
        fences = list(CODE_FENCE_RE.finditer(stripped))
        if fences:
            return fences[-1].group(1).strip()
    elif domain == "logic":
        return _extract_logic_answer(stripped)
    return extract_final_answer(stripped)


def classify_recovery_termination(
    *,
    finish_reason: str | None,
    extracted_answer: str | None,
    max_new_tokens: int,
    completion_tokens: int,
) -> str:
    """
    recovery_terminated_by for Script 3 recovery variant.

    Script 4 REI should use only records where this is 'natural'.
    """
    if extracted_answer is None:
        return "extraction_failed"
    if finish_reason == "length" or completion_tokens >= max_new_tokens:
        return "token_limit"
    return "natural"


def normalize_answer(answer: str, eval_type: str) -> str:
    text = answer.strip().lower()
    if eval_type in {"numeric", "symbolic"}:
        text = text.replace(",", "").replace("$", "").replace("%", "")
        text = re.sub(r"\s+", "", text)
        if eval_type == "numeric":
            try:
                if "e" in text:
                    return str(float(text))
                if "." in text:
                    value = float(text)
                    return str(int(value)) if value.is_integer() else str(value)
                return str(int(text))
            except ValueError:
                nums = re.findall(r"-?\d+\.?\d*", text)
                return nums[0] if nums else text
    return re.sub(r"\s+", " ", text)


def _to_float(value: str) -> float | None:
    text = value.strip().replace(",", "").replace("$", "").replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def numeric_close(pred: float, gold: float, tolerance: float = 1e-4) -> bool:
    """
    Combined absolute + relative numeric match.

    Returns True when |pred - gold| is within an absolute tolerance OR within a
    relative tolerance (|pred - gold| / max(|gold|, |pred|) <= tolerance). The
    relative branch is what SciBench-style open-ended numeric answers need, where
    answer magnitudes span many orders (e.g. 50.7 atm vs 6.022e23).
    """
    diff = abs(pred - gold)
    if diff <= tolerance:
        return True
    denom = max(abs(gold), abs(pred))
    return denom > 0 and (diff / denom) <= tolerance


def _normalize_latex_for_sympy(text: str) -> str:
    expr = text.strip()
    expr = re.sub(r"\\text\{([^}]*)\}", r"\1", expr)
    expr = expr.replace(r"\pi", "pi")
    expr = expr.replace(r"\cdot", "*")
    expr = expr.replace(r"\times", "*")
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    expr = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", expr)
    expr = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", expr)
    expr = expr.replace("{", "(").replace("}", ")")
    expr = re.sub(r"(?<=\d)i\b", "*I", expr)
    expr = re.sub(r"\bi\b", "I", expr)
    expr = re.sub(r"(?<=\d)I\b", "*I", expr)
    expr = re.sub(r"(?<=\d)([a-zA-Z])", r"*\1", expr)
    expr = re.sub(r"\s+", "", expr)
    return expr


def _parse_sympy_expr(text: str):
    from sympy import Integer, Rational, Symbol, sympify
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    normalized = _normalize_latex_for_sympy(text)
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        return parse_expr(normalized, transformations=transformations, evaluate=True)
    except Exception:
        pass
    try:
        return sympify(normalized, evaluate=True)
    except Exception:
        pass
    frac = re.fullmatch(r"(-?\d+)/(-?\d+)", normalized)
    if frac:
        return Rational(int(frac.group(1)), int(frac.group(2)))
    if re.fullmatch(r"-?\d+", normalized):
        return Integer(int(normalized))
    return None


def _parse_tuple(text: str) -> tuple | None:
    stripped = text.strip()
    if not (stripped.startswith("(") and stripped.endswith(")")):
        return None
    parts = [p.strip() for p in stripped[1:-1].split(",")]
    if len(parts) < 2:
        return None
    return tuple(parts)


def sympy_equivalent(
    predicted: str,
    gold: str,
    *,
    math_tolerance: float = 1e-4,
) -> bool | None:
    """
    Return True/False when symbolic equivalence is resolved, else None.
    """
    pred = predicted.strip()
    gold_text = gold.strip()
    if not pred or not gold_text:
        return None

    pred_float, gold_float = _to_float(pred), _to_float(gold_text)
    if pred_float is not None and gold_float is not None:
        return numeric_close(pred_float, gold_float, math_tolerance)

    pred_tuple, gold_tuple = _parse_tuple(pred), _parse_tuple(gold_text)
    if pred_tuple is not None and gold_tuple is not None:
        if len(pred_tuple) != len(gold_tuple):
            return False
        for p_part, g_part in zip(pred_tuple, gold_tuple):
            part_result = sympy_equivalent(p_part, g_part, math_tolerance=math_tolerance)
            if part_result is False:
                return False
            if part_result is None:
                if normalize_answer(p_part, "symbolic") != normalize_answer(g_part, "symbolic"):
                    return False
        return True

    try:
        from sympy import simplify

        pred_expr = _parse_sympy_expr(pred)
        gold_expr = _parse_sympy_expr(gold_text)
        if pred_expr is not None and gold_expr is not None:
            diff = simplify(pred_expr - gold_expr)
            if diff == 0:
                return True
            if hasattr(diff, "is_zero") and diff.is_zero:
                return True
            try:
                if abs(float(diff)) <= math_tolerance:
                    return True
            except (TypeError, ValueError):
                pass
            return False
    except Exception:
        return None

    return None


# Names are proper nouns (uppercase-anchored, NOT IGNORECASE so words like "and"
# are not captured as names); the role word matches any case via an inline flag.
_LOGIC_ROLE = r"((?i:knight|knave))"
_LOGIC_PAIR_RE = re.compile(rf"([A-Z][\w'-]*)\s*=\s*{_LOGIC_ROLE}")
_LOGIC_FREEFORM_RE = re.compile(
    rf"([A-Z][\w'-]*)\b[^.;\n]*?\bis\s+(?:a\s+|an\s+|the\s+)?{_LOGIC_ROLE}",
)


def parse_logic_assignment(text: str) -> dict[str, str]:
    """
    Parse a knights-and-knaves role assignment into {name(lower): role(lower)}.

    Handles the canonical gold/answer form ("Name=knight, Name=knave") first,
    then falls back to free text ("Penelope is a knave, ... Zoey is a knight").
    Later mentions of the same name win, so a concluding restatement overrides an
    earlier hypothetical ("assume X is a knight ...").
    """
    if not text:
        return {}
    assignment: dict[str, str] = {}
    for match in _LOGIC_PAIR_RE.finditer(text):
        assignment[match.group(1).lower()] = match.group(2).lower()
    if assignment:
        return assignment
    for match in _LOGIC_FREEFORM_RE.finditer(text):
        assignment[match.group(1).lower()] = match.group(2).lower()
    return assignment


def logic_equivalent(predicted: str, gold: str) -> bool | None:
    """
    Exact match on the role tuple. Returns None only when gold is unparseable.

    Every name present in gold must be assigned the same role in the prediction;
    extra names in the prediction are ignored.
    """
    gold_map = parse_logic_assignment(gold)
    if not gold_map:
        return None
    pred_map = parse_logic_assignment(predicted)
    if not pred_map:
        return False
    return all(pred_map.get(name) == role for name, role in gold_map.items())


def is_correct(
    parsed_answer: str | None,
    gold_answer: str,
    eval_type: str,
    *,
    math_tolerance: float = 1e-4,
) -> bool | None:
    if parsed_answer is None:
        return None
    if eval_type == "logic":
        equiv = logic_equivalent(parsed_answer, gold_answer)
        if equiv is not None:
            return equiv
        return normalize_answer(parsed_answer, "exact") == normalize_answer(gold_answer, "exact")
    if eval_type == "symbolic":
        equiv = sympy_equivalent(parsed_answer, gold_answer, math_tolerance=math_tolerance)
        if equiv is not None:
            return equiv
        pred = normalize_answer(parsed_answer, "symbolic")
        gold = normalize_answer(gold_answer, "symbolic")
        if pred == gold:
            return True
        pred_norm = re.sub(r"\s+", "", pred)
        gold_norm = re.sub(r"\s+", "", gold)
        return pred_norm == gold_norm
    pred = normalize_answer(parsed_answer, eval_type)
    gold = normalize_answer(gold_answer, eval_type)
    if eval_type == "exact":
        return pred == gold or pred in gold or gold in pred
    if eval_type == "numeric":
        pred_float, gold_float = _to_float(pred), _to_float(gold)
        if pred_float is not None and gold_float is not None:
            return numeric_close(pred_float, gold_float, math_tolerance)
    return pred == gold


def infer_math_eval_type(gold_answer: str) -> str:
    """Infer numeric vs symbolic grading for math-domain answers."""
    text = gold_answer.strip()
    if re.fullmatch(r"-?\d+\.?\d*", text.replace(",", "")):
        return "numeric"
    return "symbolic"


def compute_rei(
    accuracy_recovery: float,
    accuracy_stop: float,
    tokens_recovery: int,
    tokens_stop: int,
    min_token_delta: int = 1,
) -> float | None:
    delta_acc = accuracy_recovery - accuracy_stop
    delta_tokens = tokens_recovery - tokens_stop
    if delta_tokens < min_token_delta:
        return None
    return delta_acc / delta_tokens


def aggregate_rei(records: list[dict[str, Any]], min_token_delta: int = 1) -> float | None:
    """Mean REI over records where token delta is large enough."""
    values: list[float] = []
    for record in records:
        if record.get("variant") not in {"stop", "recovery"}:
            continue
        stop = record.get("_paired_stop")
        if stop is None:
            continue
        rei = compute_rei(
            accuracy_recovery=1.0 if record.get("correct") else 0.0,
            accuracy_stop=1.0 if stop.get("correct") else 0.0,
            tokens_recovery=int(record.get("tokens_generated") or 0),
            tokens_stop=int(stop.get("tokens_generated") or 0),
            min_token_delta=min_token_delta,
        )
        if rei is not None:
            values.append(rei)
    if not values:
        return None
    return sum(values) / len(values)
