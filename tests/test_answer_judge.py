"""Tests for LLM-as-a-Judge grading helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer_judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    JudgeCreditError,
    cohens_kappa,
    grade_sample,
    is_credit_error,
    judge_answer,
    parse_judge_response,
    resolve_det_tolerance,
    write_judge_raw_log,
)
from evaluator import (  # noqa: E402
    _extract_logic_answer,
    extract_domain_answer,
    extract_final_answer,
    is_correct,
)


def test_parse_judge_response():
    assert parse_judge_response("Correct") is True
    assert parse_judge_response("Incorrect") is False
    assert parse_judge_response("Incorrect.\nExtra") is False
    assert parse_judge_response("maybe") is None


def test_parse_judge_response_verbose_first_line():
    assert parse_judge_response("Incorrect. The student's value 0 does not match.") is False


def test_parse_judge_response_verdict_on_last_line():
    text = "Let me trace through the steps.\nThe values match the reference.\nCorrect"
    assert parse_judge_response(text) is True


def test_parse_judge_response_midtext_verdict_ignored():
    # 'correct' only appears in a middle line; first/last lines have no verdict.
    text = "Let me think about this.\nThis looks correct to me so far.\nHmm, not sure."
    assert parse_judge_response(text) is None


def test_parse_judge_response_incorrect_wins_on_last_line():
    text = "First the reasoning.\nThe final value is incorrect."
    assert parse_judge_response(text) is False


def test_is_credit_error_matches_credit_and_auth_not_transient():
    assert is_credit_error("Your credit balance is too low to access the Anthropic API")
    assert is_credit_error("Error code: 401 - authentication_error")
    assert is_credit_error("400 - insufficient_quota")
    assert not is_credit_error("Connection timeout")
    # 429 rate limit is transient: must NOT be treated as a hard credit error.
    assert not is_credit_error("429 rate limit exceeded")


def test_grade_sample_propagates_credit_error():
    with patch(
        "answer_judge.judge_answer",
        side_effect=JudgeCreditError("Your credit balance is too low"),
    ):
        with pytest.raises(JudgeCreditError):
            grade_sample(
                question="Q",
                gold_answer="1",
                answer_content="Final answer: 1",
                extracted_answer="1",
                domain="math",
                eval_type="exact",
                judge_cfg={"enabled": True},
                dry_run=False,
            )


def test_extract_final_answer_markdown_variants():
    assert extract_final_answer("**Final answer:** 42") == "42"
    assert extract_final_answer("**Final Answer**: 42") == "42"
    assert extract_final_answer("**Final answer:**\n\n42") == "42"
    assert extract_final_answer("Final answer: 42") == "42"


def test_extract_final_answer_prose_without_colon_is_none():
    # No colon label and no "Final answer:" line -> nothing extracted (preserved).
    assert extract_final_answer("the final answer is unclear here") is None


def test_config_judge_model():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    assert cfg["judge"]["model"] == "claude-sonnet-4-6"
    assert cfg["judge"]["max_tokens"] == 200


def test_cohens_kappa_perfect_agreement():
    labels = [1, 0, 1, 1, 0]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_cohens_kappa_empty():
    assert cohens_kappa([], []) is None


def test_grade_sample_dry_run_uses_deterministic():
    result = grade_sample(
        question="What is 2+2?",
        gold_answer="4",
        answer_content="Step 1: 2+2=4\nFinal answer: 4",
        extracted_answer="4",
        domain="math",
        eval_type="exact",
        judge_cfg={"enabled": True},
        dry_run=True,
    )
    assert result["judge_fallback"] is True
    assert result["correct"] is True
    assert result["verifier_score"] == 1
    assert result["det_correct"] is True


def test_grade_sample_judge_primary_when_enabled():
    with patch(
        "answer_judge.judge_answer",
        return_value={
            "correct": True,
            "judge_output": "Correct",
            "judge_raw_output": "Correct",
            "judge_fallback": False,
        },
    ):
        result = grade_sample(
            question="temp rise?",
            gold_answer="205",
            answer_content="Final answer: 0.205 K",
            extracted_answer="0.205",
            domain="science",
            eval_type="numeric",
            judge_cfg={"enabled": True},
            science_tolerance=1e-2,
            dry_run=False,
        )
    assert result["correct"] is True
    assert result["verifier_score"] == 1
    assert result["det_correct"] is False
    assert result["judge_fallback"] is False
    assert result["judge_raw_output"] == "Correct"


def test_grade_sample_fallback_on_judge_failure():
    with patch("answer_judge.judge_answer", side_effect=RuntimeError("api down")):
        result = grade_sample(
            question="Q",
            gold_answer="22.2",
            answer_content="Final answer: 22.24",
            extracted_answer="22.24",
            domain="science",
            eval_type="numeric",
            judge_cfg={"enabled": True},
            science_tolerance=1e-2,
            dry_run=False,
        )
    assert result["judge_fallback"] is True
    assert result["correct"] is True
    assert result["det_correct"] is True


def test_write_judge_raw_log(tmp_path):
    judge_cfg = {
        "enabled": True,
        "raw_log_enabled": True,
        "raw_log_dir": str(tmp_path / "judge_raw"),
        "model": DEFAULT_JUDGE_MODEL,
    }
    grading = {
        "correct": True,
        "verifier_score": 1,
        "judge_output": "Correct",
        "judge_raw_output": "Correct",
        "det_correct": False,
        "judge_fallback": False,
    }
    path = write_judge_raw_log(
        judge_cfg=judge_cfg,
        phase="baseline",
        task_id="test_task",
        grading=grading,
        sample_index=1,
        seed=42,
    )
    assert path is not None
    log_file = tmp_path / "judge_raw" / "baseline" / "test_task.jsonl"
    assert log_file.exists()
    row = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert row["judge_raw_output"] == "Correct"
    assert row["sample_index"] == 1


def test_judge_answer_returns_raw_output():
    with patch("answer_judge.call_anthropic_judge", return_value="Incorrect.\nExtra"):
        result = judge_answer(
            question="Q",
            gold_answer="1",
            answer_content="Final answer: 0",
            extracted_answer="0",
            domain="math",
            eval_type="exact",
            judge_cfg={"enabled": True, "model": DEFAULT_JUDGE_MODEL},
        )
    assert result["judge_fallback"] is False
    assert result["judge_output"] == "Incorrect"
    assert result["judge_raw_output"] == "Incorrect.\nExtra"


def test_regression_atkins_100_unit_mismatch_det_wrong():
    """Deterministic cross-check fails; judge path is expected to rescue."""
    det = is_correct("0.205", "205", "numeric", math_tolerance=1e-2)
    assert det is False


def test_regression_fund_412_near_miss_det_correct_at_1e2():
    det = is_correct("22.24", "22.2", "numeric", math_tolerance=1e-2)
    assert det is True


def test_regression_atkins_8_genuine_error():
    det = is_correct("0", "-167", "numeric", math_tolerance=1e-2)
    assert det is False


def test_unicode_pi_symbolic_match():
    extracted = extract_domain_answer(
        "Step 1: diff = 7π\nFinal answer: 7π",
        "math",
        "symbolic",
    )
    assert is_correct(extracted, r"7\pi", "symbolic") is True


def test_logic_truncation_fallback_returns_none():
    long_cot = "Step 1: analyze\n" + ("x" * 600)
    assert _extract_logic_answer(long_cot) is None


def test_resolve_det_tolerance():
    assert resolve_det_tolerance("science", "numeric", science_tolerance=1e-2) == 1e-2
    assert resolve_det_tolerance("math", "symbolic", math_tolerance=1e-4) == 1e-4
