"""
Math answer extraction and decoy helpers for MATH-500 task prep and evaluation.

Used by:
  prepare_tasks.py  — gold_answer, task_id, decoy_answer for math-domain rows
  evaluator.py      — extract_boxed() when parsing model outputs

Decoys are deterministic: callers pass random.Random(seed) so the same task always
gets the same wrong answer across re-runs (Script 2 injection target).
"""

from __future__ import annotations

import random
import re

# Nested braces allowed (e.g. \boxed{\frac{1}{2}}).
BOXED_ANSWER_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def read_brace_balanced(text: str, open_brace_index: int) -> tuple[str | None, int | None]:
    """Read the brace group whose opening ``{`` is at *open_brace_index*.

    Returns ``(inner, end_exclusive)``. ``inner`` is None when the group is
    never closed, which is a truncated commit rather than a nested-LaTeX miss.
    """
    if open_brace_index < 0 or open_brace_index >= len(text) or text[open_brace_index] != "{":
        return None, None
    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index + 1 : index], index + 1
    return None, None


def extract_boxed(text: str) -> str | None:
    """Return the inner content of the last \\boxed{...} in *text*, if any."""
    matches = BOXED_ANSWER_RE.findall(text)
    if not matches:
        return None
    # Last \boxed{} is the final answer; earlier ones are often intermediate steps.
    return matches[-1].strip()


def extract_boxed_first(text: str) -> str | None:
    """Return the inner content of the FIRST \\boxed{...} in *text*, if any.

    Used when *text* is already restricted to the final-answer region (after an
    explicit "Final Answer"/"Answer" marker): there the first boxed value is the
    committed answer, while a trailing boxed is often a re-check sub-result
    (pilot bug: "\\boxed{10} ... \\boxed{2}" graded 2 instead of 10).
    """
    matches = BOXED_ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[0].strip()


def parse_math500_gold(answer: str, solution: str | None = None) -> str:
    """
    Reference answer for a MATH-500 row.

    Prefer the short dataset ``answer`` field; if empty, take the last \\boxed{}
    from the long ``solution`` write-up. Returns "" when neither yields a value
    (prepare_tasks.py skips such rows).
    """
    gold = (answer or "").strip()
    if gold:
        return gold
    if solution:
        boxed = extract_boxed(solution)
        if boxed:
            return boxed
    return gold


def _sanitize_unique_id(unique_id: str) -> str:
    """Make a dataset path safe for task_id (only alnum, dot, underscore, hyphen)."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", unique_id.strip()).strip("_")


def math500_task_id(unique_id: str) -> str:
    """Internal task key, e.g. test/algebra/529.json -> math500_test_algebra_529.json."""
    return f"math500_{_sanitize_unique_id(unique_id)}"


def numeric_decoy(gold: str, rng: random.Random) -> str:
    """
    Simple wrong answer for pure numeric gold (SciBench; also used inside math_decoy).

    Integers: add a small offset (3, 7, -2, 10, -5). Decimals: multiply by 1.15.
    Non-numeric input falls back to "0".
    """
    try:
        v = float(gold.replace(",", ""))
        if v == int(v):
            return str(int(v) + rng.choice([3, 7, -2, 10, -5]))
        return str(round(v * 1.15, 4))
    except ValueError:
        return "0"


def math_decoy(gold: str, rng: random.Random) -> str:
    """
    Plausible wrong answer for MATH-500 (symbolic gold shapes).

    Rule-based perturbation per answer type — not random garbage, so Script 2
    (incorrect_reasoning / off_topic) can steer toward a believable mistake.
    Branches are tried in order; first match wins. Fallback appends +1/+2/+3.
    """
    text = gold.strip()
    if not text:
        return "0"

    # Plain scalar: delegate to numeric_decoy (e.g. gold "42" -> "49").
    if re.fullmatch(r"-?\d+\.?\d*", text.replace(",", "")):
        return numeric_decoy(text, rng)

    # Tuple (a, b, ...): perturb one component only.
    if text.startswith("(") and text.endswith(")"):
        parts = [p.strip() for p in text[1:-1].split(",")]
        if len(parts) >= 2:
            idx = rng.randrange(len(parts))
            part = parts[idx]
            try:
                val = float(part.replace(",", ""))
                if val == int(val):
                    parts[idx] = str(int(val) + rng.choice([1, 2, -1, 3]))
                else:
                    parts[idx] = str(round(val + 0.5, 4))
            except ValueError:
                parts[idx] = f"{part}_wrong"
            return "(" + ", ".join(parts) + ")"

    # Simple a/b fraction (not LaTeX): bump numerator or denominator.
    frac = re.fullmatch(r"(\d+)/(\d+)", text)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if rng.random() < 0.5:
            return f"{num + rng.choice([1, 2, 3])}/{den}"
        return f"{num}/{den + rng.choice([1, 2, 3])}"

    # Pi expressions: flip sign or add a small offset.
    if r"\pi" in text or re.search(r"(?<![a-zA-Z])pi(?![a-zA-Z])", text, re.IGNORECASE):
        offset = rng.choice([1, 2, 3, 4])
        if text.startswith("-"):
            return text[1:] or str(offset)
        return f"-({text})" if rng.random() < 0.5 else f"{text}+{offset}"

    # Complex numbers (a+bi): tweak sign or append +1.
    if re.search(r"(?<=\d)i\b", text) or re.search(r"\+\s*\d*i\b", text):
        return text.replace("i", "i").replace("+", "-+", 1) if "+" in text else f"{text}+1"

    # LaTeX text answers: generic wrong label.
    text_match = re.fullmatch(r"\\text\{([^}]+)\}", text)
    if text_match:
        return r"\text{Other}"

    # Second numeric check (defensive; unreachable for well-formed gold today).
    if re.fullmatch(r"-?\d+\.?\d*", text.replace(",", "")):
        return numeric_decoy(text, rng)

    # Equations, intervals, \frac{...}, etc.: append a small term.
    return f"{text}+{rng.choice([1, 2, 3])}"
