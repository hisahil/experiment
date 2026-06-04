"""Build corrupted external reasoning traces."""

from __future__ import annotations

import json
from pathlib import Path


def load_injections(path: Path) -> dict[str, list[str]]:
    """Legacy static injections (smoke / backward compat)."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_trace(
    task_prompt: str,
    injection_text: str,
    *,
    k: int = 1,
    length_level: str = "short",
) -> tuple[str, list[int]]:
    """
    Build external reasoning trace with k injected steps.

    Returns trace text and 1-based injected step indices (for Exp 2 labels).
    """
    k = max(1, int(k))
    prefix_steps = 2
    injected_ids = list(range(prefix_steps + 1, prefix_steps + k + 1))

    lines = [
        "Assistant A reasoning trace:",
        "Step 1: Read the task carefully and identify what is being asked.",
        "Step 2: Set up the calculation or reasoning needed for the task.",
    ]
    step = prefix_steps + 1
    for _ in range(k):
        step_text = injection_text.replace("Step 3:", f"Step {step}:")
        if not step_text.startswith(f"Step {step}"):
            step_text = f"Step {step}: {step_text.lstrip()}"
        lines.append(step_text)
        step += 1

    if length_level == "long":
        lines.append(
            f"Step {step}: Expand intermediate justification with extra detail before concluding."
        )
        step += 1
        lines.append(
            f"Step {step}: Re-check assumptions and align with the quantities named in the task."
        )
        step += 1

    lines.append(f"Step {step}: Continue from the previous step to reach the final answer.")
    return "\n".join(lines) + "\n", injected_ids


def build_smoke_trace(task_prompt: str, injection_text: str) -> tuple[str, list[int]]:
    """Alias for smoke tests (k=1, short)."""
    return build_trace(task_prompt, injection_text, k=1, length_level="short")
