"""CoT step extraction and fallback segmentation for baseline generation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from evaluator import FINAL_ANSWER_RE

if TYPE_CHECKING:
    from llm_backend import BaseBackend

STEP_HEADER_RE = re.compile(r"^Step\s+(\d+)\s*:", re.IGNORECASE | re.MULTILINE)

RESEGMENT_PROMPT = """Reformat the reasoning below into numbered steps only.

Rules:
- Use exactly this format, one step per line: Step 1: ... Step 2: ... etc.
- Produce at least {min_steps} steps.
- Do NOT include a "Final answer" line.
- Preserve the reasoning content; only restructure into steps.

Reasoning:
{reasoning}
"""


def strip_final_answer_line(text: str) -> str:
    match = FINAL_ANSWER_RE.search(text.strip())
    if match:
        return text[: match.start()].strip()
    return text.strip()


def parse_numbered_steps(text: str) -> list[str] | None:
    """Extract 'Step N: ...' blocks from model output (prompt-format primary path)."""
    body = strip_final_answer_line(text)
    if not body:
        return None

    matches = list(STEP_HEADER_RE.finditer(body))
    if not matches:
        return None

    steps: list[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end].strip()
        if chunk:
            steps.append(chunk)

    return steps if steps else None


def _renumber_steps(chunks: list[str]) -> list[str]:
    renumbered: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = re.sub(r"^Step\s+\d+\s*:\s*", "", chunk, flags=re.IGNORECASE).strip()
        renumbered.append(f"Step {i}: {text}" if text else f"Step {i}:")
    return renumbered


def segment_spacy(text: str, min_steps: int) -> list[str] | None:
    """Sentence-boundary segmentation via spaCy (optional dependency)."""
    try:
        import spacy
    except ImportError:
        return None

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        return None

    body = strip_final_answer_line(text)
    if not body:
        return None

    doc = nlp(body)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if len(sentences) < min_steps:
        return None

    return _renumber_steps(sentences)


def segment_llm_resegment(
    text: str,
    backend: BaseBackend,
    model_cfg: dict,
    min_steps: int,
) -> list[str] | None:
    """Short same-model call to restructure reasoning into numbered steps."""
    body = strip_final_answer_line(text) or text.strip()
    prompt = RESEGMENT_PROMPT.format(min_steps=min_steps, reasoning=body)
    result = backend.generate(
        prompt=prompt,
        max_new_tokens=min(int(model_cfg.get("max_new_tokens", 2048)), 1024),
        temperature=float(model_cfg.get("temperature", 0.6)),
        top_p=float(model_cfg.get("top_p", 0.95)),
    )
    steps = parse_numbered_steps(result.text)
    if steps and len(steps) >= min_steps:
        return _renumber_steps(steps)
    return None


def segment_cot(
    raw_text: str,
    *,
    min_steps: int,
    backend: BaseBackend | None = None,
    model_cfg: dict | None = None,
) -> tuple[list[str], str]:
    """
    Segment CoT into steps. Returns (steps, method).

    Order: prompt-format parse -> spaCy -> LLM re-segment.
    """
    steps = parse_numbered_steps(raw_text)
    if steps and len(steps) >= min_steps:
        return _renumber_steps(steps), "prompt"

    spacy_steps = segment_spacy(raw_text, min_steps)
    if spacy_steps:
        return spacy_steps, "spacy"

    if backend is not None and model_cfg is not None:
        llm_steps = segment_llm_resegment(raw_text, backend, model_cfg, min_steps)
        if llm_steps:
            return llm_steps, "llm_resegment"

    if steps:
        return _renumber_steps(steps), "prompt"

    if spacy_steps := segment_spacy(raw_text, 1):
        return spacy_steps, "spacy"

    return [raw_text.strip()], "prompt"


def steps_to_baseline_cot(steps: list[str], final_answer: str | None = None) -> str:
    """Join steps and optional final answer line."""
    lines = list(steps)
    if final_answer is not None:
        lines.append(f"Final answer: {final_answer}")
    return "\n".join(lines)
