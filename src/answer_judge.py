"""LLM-as-a-Judge answer grading (Claude Sonnet via Anthropic API)."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from evaluator import is_correct

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


class JudgeCreditError(RuntimeError):
    """Raised when the judge API fails for a non-transient credit/billing/auth reason.

    Unlike generic judge failures (which silently fall back to the deterministic
    cross-check), this is propagated by grade_sample() so the pipeline aborts
    instead of silently labelling thousands of rows with det_correct.
    """


# Non-transient client errors that mean "stop, do not retry": credit exhaustion,
# billing/quota, and auth/permission. Rate limits (429) are intentionally NOT
# matched here — they are transient and handled by the normal retry/backoff.
_JUDGE_CREDIT_ERROR_RE = re.compile(
    r"credit balance|insufficient.{0,20}credit|billing|quota|payment required|"
    r"\b40[0-3]\b|authentication_error|permission_error|invalid_api_key|insufficient_quota",
    re.IGNORECASE,
)


def is_credit_error(message: str) -> bool:
    """True when an exception message indicates non-transient credit/billing/auth failure."""
    return bool(_JUDGE_CREDIT_ERROR_RE.search(message or ""))

JUDGE_SYSTEM_PROMPT = """You are an expert answer grader for math, science, and logic problems.
Determine whether the student's answer is correct relative to the reference answer.

Treat as CORRECT when:
- The numerical value matches within reasonable rounding or precision
- Equivalent units or scales express the same quantity (e.g. 0.205 K vs 205 mK)
- Mathematically or logically equivalent forms match (fractions, expressions, assignments)
- The reasoning supports the reference answer even if the extracted final line differs slightly

Treat as INCORRECT when:
- The reasoning or final answer is fundamentally wrong
- The student is uncertain when a definitive answer is required
- Assignments or truth values contradict the reference

Respond with ONLY one word: Correct or Incorrect."""


def resolve_det_tolerance(
    domain: str,
    eval_type: str | None,
    *,
    math_tolerance: float = 1e-4,
    science_tolerance: float = 1e-2,
) -> float:
    """Tolerance for the deterministic cross-check (not the judge)."""
    if domain == "science":
        return science_tolerance
    return math_tolerance


def resolve_eval_type(domain: str, eval_type: str | None, gold_answer: str) -> str:
    if eval_type:
        return str(eval_type)
    if domain == "science":
        return "numeric"
    if domain == "logic":
        return "logic"
    return "exact"


def build_judge_user_prompt(
    *,
    question: str,
    gold_answer: str,
    answer_content: str,
    extracted_answer: str | None,
    domain: str,
    eval_type: str | None,
) -> str:
    extracted_line = extracted_answer if extracted_answer is not None else "(none parsed)"
    return (
        f"Domain: {domain}\n"
        f"Eval type: {eval_type or 'unspecified'}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Reference answer:\n{gold_answer.strip()}\n\n"
        f"Extracted final answer:\n{extracted_line}\n\n"
        f"Student reasoning and answer content:\n{answer_content.strip()}"
    )


def parse_judge_response(text: str) -> bool | None:
    """Parse a Correct/Incorrect verdict from the judge.

    Checks the FIRST and LAST line only (an exact one-word verdict, then a
    word-boundary regex). Restricting the regex to first/last line avoids false
    positives on mid-text reasoning like "I need to check if this is correct".
    """
    if not text.strip():
        return None
    lines = text.strip().splitlines()
    # Exact one-word verdict on the first or last line (original behaviour).
    for line in (lines[0], lines[-1]):
        lowered = line.strip().lower().rstrip(".")
        if lowered == "correct":
            return True
        if lowered == "incorrect":
            return False
    # Regex on first and last line only (avoid false positives on mid-text).
    for line in (lines[0], lines[-1]):
        lowered = line.strip().lower()
        if re.search(r"\bincorrect\b", lowered):
            return False
        if re.search(r"\bcorrect\b", lowered):
            return True
    return None


def judge_raw_log_path(judge_cfg: dict[str, Any], phase: str, task_id: str) -> Path | None:
    if not judge_cfg.get("raw_log_enabled", True):
        return None
    base = Path(str(judge_cfg.get("raw_log_dir", "data/judge_raw")))
    safe_task_id = task_id.replace("/", "_")
    return base / phase / f"{safe_task_id}.jsonl"


def append_judge_raw_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_judge_raw_log(
    *,
    judge_cfg: dict[str, Any],
    phase: str,
    task_id: str,
    grading: dict[str, Any],
    dry_run: bool = False,
    **extra: Any,
) -> str | None:
    """Append one judge call to data/judge_raw/{phase}/{task_id}.jsonl; return path."""
    if dry_run or not judge_cfg.get("enabled", True):
        return None
    path = judge_raw_log_path(judge_cfg, phase, task_id)
    if path is None:
        return None
    record: dict[str, Any] = {
        "task_id": task_id,
        "phase": phase,
        "judge_model": judge_cfg.get("model", DEFAULT_JUDGE_MODEL),
        "judge_raw_output": grading.get("judge_raw_output", ""),
        "judge_output": grading.get("judge_output", ""),
        # judge_verdict stores the graded verdict (grading["correct"]), NOT whether
        # the judge text parsed. Renamed from the misleading "judge_parsed".
        "judge_verdict": grading.get("correct"),
        "judge_fallback": grading.get("judge_fallback"),
        "verifier_score": grading.get("verifier_score"),
        "det_correct": grading.get("det_correct"),
    }
    if grading.get("judge_error"):
        record["judge_error"] = grading["judge_error"]
    record.update(extra)
    append_judge_raw_log(path, record)
    return str(path)


# Prompt caching: the JUDGE_SYSTEM_PROMPT is identical for every judge call, so
# marking it as a cache_control block makes all calls after the first read it at
# 0.1x input price. Opt-in via ANTHROPIC_PROMPT_CACHE=1.
JUDGE_PROMPT_CACHE_ENABLED = os.environ.get("ANTHROPIC_PROMPT_CACHE", "0") == "1"


def call_anthropic_judge(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    system: str,
    max_retries: int = 3,
) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Install anthropic: pip install anthropic") from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    if JUDGE_PROMPT_CACHE_ENABLED and system:
        system_param: Any = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    else:
        system_param = system
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_param,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as exc:
            # Credit/billing/auth errors are non-transient: abort immediately
            # (circuit breaker) instead of burning retries and silently falling
            # back to the deterministic cross-check.
            if is_credit_error(str(exc)):
                raise JudgeCreditError(
                    f"Judge API credit/billing/auth failure: {exc}"
                ) from exc
            last_exc = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Judge API failed after {max_retries} attempts: {last_exc}")


def judge_answer(
    *,
    question: str,
    gold_answer: str,
    answer_content: str,
    extracted_answer: str | None,
    domain: str,
    eval_type: str | None,
    judge_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Return judge verdict dict with keys correct, judge_output, judge_raw_output, judge_fallback."""
    if not judge_cfg.get("enabled", True):
        return {
            "correct": None,
            "judge_output": "",
            "judge_raw_output": "",
            "judge_fallback": True,
        }

    prompt = build_judge_user_prompt(
        question=question,
        gold_answer=gold_answer,
        answer_content=answer_content,
        extracted_answer=extracted_answer,
        domain=domain,
        eval_type=eval_type,
    )
    raw = call_anthropic_judge(
        prompt=prompt,
        model=str(judge_cfg.get("model", DEFAULT_JUDGE_MODEL)),
        max_tokens=int(judge_cfg.get("max_tokens", 200)),
        temperature=float(judge_cfg.get("temperature", 0)),
        system=str(judge_cfg.get("system_prompt", JUDGE_SYSTEM_PROMPT)),
        max_retries=int(judge_cfg.get("max_retries", 3)),
    )
    raw_stripped = raw.strip()
    parsed = parse_judge_response(raw)
    if parsed is None:
        return {
            "correct": None,
            "judge_output": raw_stripped,
            "judge_raw_output": raw_stripped,
            "judge_fallback": True,
        }
    return {
        "correct": parsed,
        "judge_output": "Correct" if parsed else "Incorrect",
        "judge_raw_output": raw_stripped,
        "judge_fallback": False,
    }


def grade_sample(
    *,
    question: str,
    gold_answer: str,
    answer_content: str,
    extracted_answer: str | None,
    domain: str,
    eval_type: str | None,
    judge_cfg: dict[str, Any],
    math_tolerance: float = 1e-4,
    science_tolerance: float = 1e-2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Primary grading via LLM judge; deterministic is_correct() runs in parallel
    for agreement reporting (det_correct).
    """
    resolved_eval_type = resolve_eval_type(domain, eval_type, gold_answer)
    det_tol = resolve_det_tolerance(
        domain,
        resolved_eval_type,
        math_tolerance=math_tolerance,
        science_tolerance=science_tolerance,
    )
    det_result = is_correct(
        extracted_answer,
        str(gold_answer),
        resolved_eval_type,
        math_tolerance=det_tol,
    )
    det_correct = bool(det_result) if det_result is not None else False

    use_judge = bool(judge_cfg.get("enabled", True)) and not dry_run
    if not use_judge:
        return {
            "correct": det_correct,
            "verifier_score": int(det_correct),
            "judge_output": "Correct" if det_correct else "Incorrect",
            "judge_raw_output": "",
            "det_correct": det_correct,
            "judge_fallback": True,
        }

    try:
        judge_result = judge_answer(
            question=question,
            gold_answer=str(gold_answer),
            answer_content=answer_content,
            extracted_answer=extracted_answer,
            domain=domain,
            eval_type=resolved_eval_type,
            judge_cfg=judge_cfg,
        )
    except JudgeCreditError:
        # Do NOT silently fall back on credit exhaustion — propagate so the
        # pipeline (or rejudge script) aborts and the operator tops up credits.
        raise
    except Exception as exc:
        return {
            "correct": det_correct,
            "verifier_score": int(det_correct),
            "judge_output": f"fallback:{exc}",
            "judge_raw_output": "",
            "judge_error": str(exc),
            "det_correct": det_correct,
            "judge_fallback": True,
        }

    if judge_result.get("judge_fallback") or judge_result.get("correct") is None:
        return {
            "correct": det_correct,
            "verifier_score": int(det_correct),
            "judge_output": judge_result.get("judge_output", ""),
            "judge_raw_output": judge_result.get("judge_raw_output", ""),
            "det_correct": det_correct,
            "judge_fallback": True,
        }

    correct = bool(judge_result["correct"])
    return {
        "correct": correct,
        "verifier_score": int(correct),
        "judge_output": judge_result.get("judge_output", ""),
        "judge_raw_output": judge_result.get("judge_raw_output", ""),
        "det_correct": det_correct,
        "judge_fallback": False,
    }


def cohens_kappa(y1: list[int], y2: list[int]) -> float | None:
    """Cohen's kappa for two binary label lists of equal length."""
    if not y1 or len(y1) != len(y2):
        return None
    n = len(y1)
    agree = sum(1 for a, b in zip(y1, y2) if a == b)
    p_o = agree / n
    p_yes_1 = sum(y1) / n
    p_yes_2 = sum(y2) / n
    p_e = p_yes_1 * p_yes_2 + (1 - p_yes_1) * (1 - p_yes_2)
    if abs(1 - p_e) < 1e-12:
        return None
    return (p_o - p_e) / (1 - p_e)
