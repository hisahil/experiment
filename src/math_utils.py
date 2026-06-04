"""Math answer extraction and decoy helpers for MATH-500 / GSM8K task prep."""

from __future__ import annotations

import random
import re

BOXED_ANSWER_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> str | None:
    """Return the inner content of the last \\boxed{...} in *text*, if any."""
    matches = BOXED_ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def parse_math500_gold(answer: str, solution: str | None = None) -> str:
    """Prefer the dataset answer; fall back to the last \\boxed{} in the solution."""
    gold = (answer or "").strip()
    if gold:
        return gold
    if solution:
        boxed = extract_boxed(solution)
        if boxed:
            return boxed
    return gold


def _sanitize_unique_id(unique_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", unique_id.strip()).strip("_")


def math500_task_id(unique_id: str) -> str:
    return f"math500_{_sanitize_unique_id(unique_id)}"


def numeric_decoy(gold: str, rng: random.Random) -> str:
    try:
        v = float(gold.replace(",", ""))
        if v == int(v):
            return str(int(v) + rng.choice([3, 7, -2, 10, -5]))
        return str(round(v * 1.15, 4))
    except ValueError:
        return "0"


def math_decoy(gold: str, rng: random.Random) -> str:
    """
    Build a plausible wrong answer for math tasks.

    Handles plain numbers, tuples, simple fractions, pi expressions, and text.
    """
    text = gold.strip()
    if not text:
        return "0"

    if re.fullmatch(r"-?\d+\.?\d*", text.replace(",", "")):
        return numeric_decoy(text, rng)

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

    frac = re.fullmatch(r"(\d+)/(\d+)", text)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if rng.random() < 0.5:
            return f"{num + rng.choice([1, 2, 3])}/{den}"
        return f"{num}/{den + rng.choice([1, 2, 3])}"

    if r"\pi" in text or re.search(r"(?<![a-zA-Z])pi(?![a-zA-Z])", text, re.IGNORECASE):
        offset = rng.choice([1, 2, 3, 4])
        if text.startswith("-"):
            return text[1:] or str(offset)
        return f"-({text})" if rng.random() < 0.5 else f"{text}+{offset}"

    if re.search(r"(?<=\d)i\b", text) or re.search(r"\+\s*\d*i\b", text):
        return text.replace("i", "i").replace("+", "-+", 1) if "+" in text else f"{text}+1"

    text_match = re.fullmatch(r"\\text\{([^}]+)\}", text)
    if text_match:
        return r"\text{Other}"

    if re.fullmatch(r"-?\d+\.?\d*", text.replace(",", "")):
        return numeric_decoy(text, rng)

    return f"{text}+{rng.choice([1, 2, 3])}"
