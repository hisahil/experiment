"""
CoT step extraction and fallback segmentation for Script 1 (baseline generation).

Used by generate_baselines.py to turn raw model output into a list of numbered steps
(["Step 1: ...", "Step 2: ..."]) that Script 3 later injects into at the midpoint.

Primary path: model already wrote "Step N:" lines (prompt-format parse).
Fallbacks: spaCy sentence split, then a short LLM reformat call.
Hard quality gate (min_steps, verified) lives in generate_baselines.py, not here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from evaluator import FINAL_ANSWER_RE

if TYPE_CHECKING:
    from llm_backend import BaseBackend

STEP_HEADER_RE = re.compile(r"^Step\s+(\d+)\s*:", re.IGNORECASE | re.MULTILINE)

# Fallback prompt when prose has no "Step N:" lines; same model, capped at 1024 tokens.
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
    """
    Remove the trailing "Final answer: ..." line from model output.

    Segmentation should cover reasoning only; the answer is re-appended later by
    steps_to_baseline_cot() after grading.
    """
    match = FINAL_ANSWER_RE.search(text.strip())
    if match:
        return text[: match.start()].strip()
    return text.strip()


def parse_numbered_steps(text: str) -> list[str] | None:
    """
    Primary parser: extract 'Step N: ...' blocks from model output.

    Returns None when no Step headers are found (triggers fallbacks in segment_cot).
    Uses the last region before any Final answer line; each block runs until the
    next Step header.
    """
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
    """
    Format already-split chunks as Step 1:, Step 2:, ... (not a segmentation method).

    Strips any existing Step N: prefix and renumbers from 1. Used by the prompt
    parser, spaCy, and LLM-resegment paths so all outputs share one shape.
    """
    renumbered: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = re.sub(r"^Step\s+\d+\s*:\s*", "", chunk, flags=re.IGNORECASE).strip()
        renumbered.append(f"Step {i}: {text}" if text else f"Step {i}:")
    return renumbered


# Loaded once per process and reused; spacy.load() costs ~1-2s, and segment_cot
# may call segment_spacy twice per sample (min_steps then min_steps=1).
_SPACY_NLP = None
_SPACY_UNAVAILABLE = False


def _get_spacy_nlp():
    """Lazy, cached spaCy pipeline; None if spaCy or en_core_web_sm is unavailable."""
    global _SPACY_NLP, _SPACY_UNAVAILABLE
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if _SPACY_UNAVAILABLE:
        return None
    try:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        # Cache the failure so we do not retry the expensive load on every call.
        _SPACY_UNAVAILABLE = True
        return None
    return _SPACY_NLP


def segment_spacy(text: str, min_steps: int) -> list[str] | None:
    """
    Fallback 1: split prose into sentences via spaCy, then _renumber_steps.

    Returns None when spaCy is missing, the body is empty, or fewer than min_steps
    sentences are found. Optional dependency — ImportError is handled gracefully.
    """
    nlp = _get_spacy_nlp()
    if nlp is None:
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
    """
    Fallback 2: ask the same model to reformat reasoning into numbered steps.

    strip_final_answer_line runs first so the model does not echo a Final answer
    line into the step list. Output is parsed with parse_numbered_steps; None if
    the reformat still yields too few steps.
    """
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
    Best-effort CoT segmentation. Always returns a list; method tags how it was built.

    Order (first success wins):
      1. parse_numbered_steps — model wrote Step N: (method "prompt")
      2. segment_spacy — sentence split (method "spacy")
      3. segment_llm_resegment — second model call (method "llm_resegment")
      4. partial Step list if any steps were found but < min_steps
      5. segment_spacy with min_steps=1 — relaxed sentence split
      6. single-element list = whole raw text (method "prompt"; often fails min_steps
         in generate_baselines and becomes exclude_reason: insufficient_steps)

    Returns (steps, method). method is logged in baseline_cots.jsonl as
    segmentation_method for post-run audits.
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

    # Fewer than min_steps but some Step headers — return anyway; Script 1 may exclude.
    if steps:
        return _renumber_steps(steps), "prompt"

    # Last spaCy attempt: accept any sentence count >= 1.
    if spacy_steps := segment_spacy(raw_text, 1):
        return spacy_steps, "spacy"

    # Guaranteed fallback: one blob; step_count=1 usually fails verified gate.
    return [raw_text.strip()], "prompt"


def steps_to_baseline_cot(steps: list[str], final_answer: str | None = None) -> str:
    """
    Rejoin segmented steps into the baseline_cot string stored in baseline_cots.jsonl.

    Appends "Final answer: ..." when parsed_answer is known so Script 1 can grade
    and Script 3 can inject the prefix up to the midpoint.
    """
    lines = list(steps)
    if final_answer is not None:
        lines.append(f"Final answer: {final_answer}")
    return "\n".join(lines)
