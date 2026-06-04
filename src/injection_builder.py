"""Domain-aware injection text from templates."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


def load_injection_templates(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _decoy_answer(task: dict, rng: random.Random) -> str:
    if task.get("decoy_answer"):
        return str(task["decoy_answer"])
    gold = str(task.get("gold_answer", ""))
    eval_type = task.get("eval_type", "exact")
    if eval_type == "numeric":
        try:
            value = float(gold.replace(",", ""))
            if value == int(value):
                offsets = [1, 2, -1, 7, 10, -5]
                return str(int(value) + rng.choice(offsets))
            return str(round(value * 1.1, 4))
        except ValueError:
            pass
    return f"not {gold}"


def _shift_hint(task: dict, rng: random.Random) -> str:
    if task.get("shift_hint"):
        return str(task["shift_hint"])
    prompt = task.get("prompt", "")
    nums = re.findall(r"\d+", prompt)
    if nums:
        n = int(rng.choice(nums))
        return str(n + rng.choice([3, 5, 10, -2]))
    return "an alternative value"


def resolve_injection_text(
    templates: dict,
    error_type: str,
    task: dict,
    rng: random.Random,
) -> str:
    domain = task.get("domain", "default")
    bucket = templates.get(error_type, {})
    template = bucket.get(domain) or bucket.get("default", "Step 3: {decoy_answer}.")
    decoy = _decoy_answer(task, rng)
    return template.format(decoy_answer=decoy, shift_hint=_shift_hint(task, rng))
