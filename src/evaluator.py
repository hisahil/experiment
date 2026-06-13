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


_MATH_TAIL_WINDOW = 4000

# Colon may sit inside (**Answer:**) or outside (**Answer**:) the bold markers.
_MATH_LINE_PATTERNS = (
    re.compile(r"^\*\*Answer:?\*\*:?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^Answer:\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\*\*Final\s+Answer:?\*\*:?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^Final\s+Answer:?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\*\*Conclusion:?\*\*:?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\*\*Final\s+Equation:?\*\*:?\s*(.+)$", re.IGNORECASE),
)

# Bold-wrapped final phrases. These are strong final-answer signals and run
# before loose per-line heuristics; last match in the slice wins.
_MATH_BOLD_FINAL_PATTERNS = (
    re.compile(r"(?:confirmed to be|equals?|is|are)\s+\*\*([^*]+)\*\*", re.IGNORECASE),
    re.compile(r"there\s+are\s+\*\*(\d+[^*]*)\*\*", re.IGNORECASE),
    re.compile(r"\*\*(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\*\*\s*\.?\s*$", re.MULTILINE),
)

_MATH_TAIL_PATTERNS = (
    re.compile(r"(?:confirmed to be|is)\s+\*\*([^*]+)\*\*", re.IGNORECASE),
    re.compile(r"(?:Thus|Therefore),?\s+[^.\n]{0,150}\s+is\s+\*\*([^*]+)\*\*", re.IGNORECASE),
    re.compile(r"there\s+are\s+\*\*(\d+[^*]*)\*\*", re.IGNORECASE),
    re.compile(
        r"(?:gives?|yields?|simplif(?:y|ies)(?:\s+\d+/\d+)?\s+to)\s+(\d+\s*/\s*\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:area|median|probability|sum|value)\s+(?:of[^.\n]{0,60}\s+)?is\s+(\d+(?:\s*/\s*\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"there\s+are\s+(\d+)\s+possible", re.IGNORECASE),
    re.compile(r"\\\[\s*([^[\]]+=\s*0)\s*\\\]\s*\.?\s*$"),
    re.compile(r"\\\[\s*([^[\]]+)\s*\\\]\s*\.?\s*$"),
    re.compile(r"is\s+\\\(\s*([^)]+?)\s*\\\)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"as\s+(\d+(?:\.\d+)?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"is\s+(\d+\s*/\s*\d+)\s*\.?\s*$", re.IGNORECASE),
)

_FINAL_ANSWER_MARKER_RE = re.compile(
    r"^\*\*Final\s+Answer:?\*\*:?\s*$|^Final\s+Answer:?\s*$",
    re.IGNORECASE,
)
_ANSWER_MARKER_RE = re.compile(r"^\*\*Answer:?\*\*:?\s*$", re.IGNORECASE)
_CONCLUSION_MARKER_RE = re.compile(r"^\*\*Conclusion:?\*\*:?\s*$", re.IGNORECASE)
_FINAL_EQUATION_MARKER_RE = re.compile(r"^\*\*Final\s+Equation:?\*\*:?\s*$", re.IGNORECASE)
# Positional markers for Phase-4 final-region slicing (last marker wins).
# Colon may sit inside or outside the bold (**Final Answer:** vs **Final Answer**:).
_FINAL_REGION_MARKERS = (
    re.compile(r"\\boxed\{"),
    re.compile(r"^\s*\*\*Final\s+Answer:?\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*\*Answer:?\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*\*Conclusion:?\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*\*Final\s+Equation:?\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Final\s+Answer\s*:", re.IGNORECASE | re.MULTILINE),
)
_FUNCTION_RHS_RE = re.compile(r"^[A-Za-z]+\([^)]*\)\s*=\s*(.+)$")
_COUNT_ANSWER_RE = re.compile(
    r"(?:^|there\s+are|is)\s*(\d+)\s+possible",
    re.IGNORECASE,
)
# Definitional formulas (area = 1/2 * base * height) are not final answers.
_DEFINITIONAL_EXPR_RE = re.compile(r"\\text|\\times|\\frac|\\cdot|base|height", re.IGNORECASE)
_SYMBOLIC_EQUATION_RE = re.compile(
    r"("
    r"(?:[-+]?\s*\d*\s*[xyz]\b)"
    r"(?:\s*[-+]\s*\d*\s*[xyz]\b)*"
    r"(?:\s*[-+]\s*\d+)?"
    r"\s*=\s*"
    r"-?\d+"
    r")",
    re.IGNORECASE,
)


def _strip_thinking_suffix(text: str) -> str:
    """Prefer model text after a closing thinking tag when present."""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx == -1:
        return text
    after = text[idx + len(marker) :].strip()
    return after or text


def _plausible_math_answer(text: str) -> bool:
    if not text or len(text) > 200:
        return False
    if _normalize_symbolic_equation(text) is not None:
        return True
    if re.search(r"\d", text) or "=" in text or "\\" in text or "/" in text:
        return True
    return len(text) <= 30


def _normalize_symbolic_equation(text: str) -> str | None:
    """
    Extract a linear equation answer (e.g. 5x - 7y + 11z + 4 = 0) from wrapped text.

    MATH-500 plane problems use this form instead of a scalar \\boxed{} value. Must run
    before the trailing-digit heuristic, which would otherwise return only the final 0.
    """
    if not text or "=" not in text:
        return None
    cleaned = text.strip().rstrip(".")
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-•]\s+(?=\d)", "", cleaned)
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = re.sub(r"\\\(|\\\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not re.search(r"(?<![a-zA-Z])[xyz](?![a-zA-Z])", cleaned, re.IGNORECASE):
        return None
    matches = list(_SYMBOLIC_EQUATION_RE.finditer(cleaned))
    if not matches:
        return None
    equation = matches[-1].group(1).strip()
    if not _is_multi_term_equation(equation):
        return None
    return equation


def _is_multi_term_equation(equation: str) -> bool:
    """Reject single assignments like y = 1 from intermediate scratch work."""
    if "=" not in equation:
        return False
    lhs = equation.split("=", 1)[0]
    var_terms = re.findall(r"[-+]?\s*\d*\s*[xyz]\b", lhs, re.IGNORECASE)
    if len(var_terms) >= 2:
        return True
    variables = set(re.findall(r"(?<![a-zA-Z])[xyz](?![a-zA-Z])", lhs, re.IGNORECASE))
    return len(variables) >= 2


def _locate_final_answer_region(body: str) -> str | None:
    """
    Return text from the last explicit final-answer marker through the end.

    When present, downstream parsing uses only this slice so flow-text heuristics
    do not pick intermediate values (Phase 4).
    """
    last_start: int | None = None
    for pattern in _FINAL_REGION_MARKERS:
        for match in pattern.finditer(body):
            if last_start is None or match.start() > last_start:
                last_start = match.start()
    if last_start is None:
        return None
    return body[last_start:]


def _coerce_math_answer_candidate(candidate: str) -> str | None:
    """Pull a gradable math answer out of a marker line or short prose."""
    if not candidate:
        return None
    text = candidate.strip().rstrip(".")
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text).strip()
    text = text.replace(r"\dfrac", r"\frac")
    if not text:
        return None

    count_match = _COUNT_ANSWER_RE.search(text)
    if count_match:
        return count_match.group(1)

    func_match = _FUNCTION_RHS_RE.match(text)
    if func_match:
        rhs = func_match.group(1).strip()
        inner = _coerce_math_answer_candidate(rhs)
        return inner if inner is not None else rhs

    equation = _normalize_symbolic_equation(text)
    if equation is not None:
        return equation

    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    inline = re.search(r"\\\((.+?)\\\)\s*$", text)
    if inline:
        inner = inline.group(1).strip()
        equation = _normalize_symbolic_equation(inner)
        if equation is not None:
            return equation
        return inner

    for pat in (
        r"(?:is|are|as|gives?)\s+([-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*$",
        r"(\d+\s*/\s*\d+)\s*$",
        r"([-+]?\d+(?:\.\d+)?)\s*$",
    ):
        match = re.search(pat, text, re.IGNORECASE)
        if match and len(text) <= 120:
            return match.group(1).replace(" ", "")

    if "=" in text and len(text) <= 160:
        return re.sub(r"\\\(|\\\)", "", text).strip()

    if len(text) <= 80 and _plausible_math_answer(text):
        return text
    return None


def _extract_math_answer_from_slice(slice_text: str, *, allow_flow_heuristics: bool) -> str | None:
    """
    Parse order inside a text slice (most explicit first, loosest last):

    1. Last \\boxed{} in slice
    2. Explicit marker blocks (**Final Answer:** / **Answer:** / **Conclusion:**)
    3. Marker-line captures (**Answer:** value on same line)
    4. [flow] bold-wrapped final phrases (...is **32**, there are **4** ...): last match wins
    5. Per-line loose heuristics (inline is \\(X\\), multi-term equations)
    6. [flow] remaining tail regex heuristics

    Bold-wrapped final phrases (step 4) run before the loose per-line inline
    heuristic so a definitional mid-text "is \\(formula\\)" cannot beat the real
    final "**value**" near the end.
    """
    boxed = extract_boxed(slice_text)
    if boxed is not None:
        return boxed

    lines = [ln.strip() for ln in slice_text.splitlines() if ln.strip()]
    marker_res = (
        _FINAL_ANSWER_MARKER_RE,
        _FINAL_EQUATION_MARKER_RE,
        _ANSWER_MARKER_RE,
        _CONCLUSION_MARKER_RE,
    )
    for i in range(len(lines) - 1, -1, -1):
        if any(marker.match(lines[i]) for marker in marker_res):
            for follow in lines[i + 1 : i + 4]:
                answer = _coerce_math_answer_candidate(follow)
                if answer and _plausible_math_answer(answer):
                    return answer

    for line in reversed(lines):
        for pat in _MATH_LINE_PATTERNS:
            match = pat.match(line)
            if match:
                answer = _coerce_math_answer_candidate(match.group(1))
                if answer and _plausible_math_answer(answer):
                    return answer

    if allow_flow_heuristics:
        for pat in _MATH_BOLD_FINAL_PATTERNS:
            matches = list(pat.finditer(slice_text))
            if not matches:
                continue
            answer = _coerce_math_answer_candidate(matches[-1].group(1))
            if answer and _plausible_math_answer(answer):
                return answer

    for line in reversed(lines):
        inline = re.search(r"is\s+\\\(\s*([^)]+?)\s*\\\)\s*\.?\s*$", line, re.IGNORECASE)
        if inline and not _DEFINITIONAL_EXPR_RE.search(inline.group(1)):
            answer = _coerce_math_answer_candidate(inline.group(1))
            if answer:
                return answer
        equation = _normalize_symbolic_equation(line)
        if equation is not None:
            return equation

    if allow_flow_heuristics:
        for pat in _MATH_TAIL_PATTERNS:
            matches = list(pat.finditer(slice_text))
            if not matches:
                continue
            answer = _coerce_math_answer_candidate(matches[-1].group(1))
            if answer and _plausible_math_answer(answer):
                return answer

    return None


def _extract_math_answer(text: str) -> str | None:
    """
    Math answer extraction for recovery runs.

    Stop variant uses a \\boxed{ prefill, so answers are almost always parseable.
    Recovery has no prefill; the model often emits **Answer:** / **Final Answer:**
    blocks instead of \\boxed{}.

    Phase-4 rule: when an explicit final marker exists, parse only from that marker
    to EOF (no flow-text heuristics earlier in the completion).
    """
    stripped = text.strip()
    body = _strip_thinking_suffix(stripped)

    final_region = _locate_final_answer_region(body)
    if final_region is not None:
        answer = _extract_math_answer_from_slice(final_region, allow_flow_heuristics=False)
        if answer is not None:
            return _normalize_math_extracted(answer)

    # No explicit final marker: fall back to tail window with flow heuristics.
    tail = body[-_MATH_TAIL_WINDOW:] if len(body) > _MATH_TAIL_WINDOW else body
    answer = _extract_math_answer_from_slice(tail, allow_flow_heuristics=True)
    return _normalize_math_extracted(answer) if answer is not None else None


def _normalize_math_extracted(answer: str) -> str:
    """Normalize LaTeX variants so symbolic grading matches MATH-500 gold."""
    return answer.replace(r"\dfrac", r"\frac").strip()


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

    Inputs are per-run raw completions (not aggregated accuracies).

    Math: \\boxed{} first (stop prefill), then recovery-style markers
    (**Answer:**, **Final Answer:**, trailing display math).
    Science uses open-ended numeric extraction (SciBench). Falls back to the
    'Final answer:' line when domain patterns do not match.
    """
    stripped = text.strip()
    if domain == "math":
        answer = _extract_math_answer(stripped)
        if answer is not None:
            return answer
    elif domain == "science":
        num = _extract_numeric_answer(stripped)
        if num is not None:
            return num
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
    extraction_failed means no gradable answer could be parsed; token_limit
    means the run hit the recovery cap (may still contain a parseable answer).
    """
    if extracted_answer is None:
        return "extraction_failed"
    if finish_reason == "length" or completion_tokens >= max_new_tokens:
        return "token_limit"
    return "natural"


def effective_recovery_termination(
    run: dict[str, Any],
    *,
    recovery_max_tokens: int,
) -> str | None:
    """
    Re-derive recovery_terminated_by from raw_output (Script 4 / regrade path).

    Uses the current extract_domain_answer() implementation so parser fixes
    apply to historical JSONL without re-running GPU inference. When extraction
    succeeds, token_limit is inferred from tokens_generated >= recovery_max_tokens.
    """
    if run.get("variant") != "recovery":
        return None
    raw = run.get("raw_output")
    if raw is None:
        return run.get("recovery_terminated_by", "extraction_failed")
    extracted = extract_domain_answer(
        str(raw),
        run.get("domain", "math"),
        run.get("eval_type"),
    )
    if extracted is None:
        return "extraction_failed"
    tokens = int(run.get("tokens_generated") or 0)
    if tokens >= recovery_max_tokens:
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


# ---------------------------------------------------------------------------
# UNUSED / DEAD CODE -- nicht in der Pipeline aufgerufen (Stand: Juni 2026).
# Die aktive REI-Berechnung lebt in scripts/evaluate.py (compute_rei, L451),
# die ueber _mean() None/NaN korrekt ueberspringt und nur variant=="recovery"
# aggregiert. Die beiden Funktionen unten bleiben nur als Referenz erhalten.
# Bekannte Schwaechen (NICHT als Vorlage uebernehmen):
#   - aggregate_rei: `1.0 if record.get("correct") else 0.0` wertet correct=None
#     als falsch statt es auszuschliessen.
#   - aggregate_rei: Variant-Filter laesst "stop" durch und verlaesst sich auf
#     fehlendes _paired_stop.
# Vor erneuter Nutzung pruefen oder loeschen.
# ---------------------------------------------------------------------------
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
