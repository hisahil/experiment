"""Tests for the A1b stop-truncation grading rule in Script 4."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import _stop_answer_is_complete, score_runs  # noqa: E402

CONFIG = {"math_tolerance": 1e-4, "science_tolerance": 1e-2}


def _stop(**kw):
    base = {
        "variant": "stop",
        "domain": "math",
        "eval_type": "numeric",
        "gold_answer": "1234",
        "stop_max_new_tokens": 500,
    }
    base.update(kw)
    return base


class StopAnswerCompleteTests(unittest.TestCase):
    def test_closed_boxed_is_complete(self):
        run = _stop(raw_output="</think>\n\nTherefore, the final answer is \\boxed{1234}")
        self.assertTrue(_stop_answer_is_complete(run))

    def test_open_boxed_prefill_truncated_is_incomplete(self):
        # Prefill trap: parser would salvage nothing AND brace never closed.
        run = _stop(raw_output="</think>\n\nTherefore, the final answer is \\boxed{12")
        self.assertFalse(_stop_answer_is_complete(run))

    def test_empty_output_is_incomplete(self):
        self.assertFalse(_stop_answer_is_complete(_stop(raw_output="")))


class ScoreRunsTruncationTests(unittest.TestCase):
    def test_truncated_open_boxed_overrides_judge_incorrect(self):
        # Judge said wrong on the partial text, but truncation -> ungradable.
        run = _stop(
            raw_output="</think>\n\nTherefore, the final answer is \\boxed{12",
            finish_reason="length",
            correct=False,  # judge verdict on partial text
            tokens_generated=500,
        )
        score_runs([run], CONFIG)
        self.assertIsNone(run["_correct"])
        self.assertEqual(run["_ungradable_reason"], "truncated")
        self.assertTrue(run["_stop_truncated_no_answer"])

    def test_truncated_but_closed_boxed_is_graded(self):
        # Complete answer reached before the cap -> grade normally (judge verdict).
        run = _stop(
            raw_output="</think>\n\nTherefore, the final answer is \\boxed{1234} and then ...",
            finish_reason="length",
            correct=True,
            tokens_generated=500,
        )
        score_runs([run], CONFIG)
        self.assertEqual(run["_correct"], 1.0)
        self.assertFalse(run["_stop_truncated_no_answer"])

    def test_natural_stop_no_truncation_unaffected(self):
        run = _stop(
            raw_output="</think>\n\nTherefore, the final answer is \\boxed{9}",
            finish_reason="stop",
            correct=False,
            tokens_generated=40,
        )
        score_runs([run], CONFIG)
        self.assertEqual(run["_correct"], 0.0)
        self.assertFalse(run["_stop_truncated_no_answer"])

    def test_recovery_token_limit_not_affected_by_rule(self):
        # Recovery may be long; truncation rule must not touch it.
        run = {
            "variant": "recovery",
            "domain": "math",
            "eval_type": "numeric",
            "gold_answer": "1234",
            "raw_output": "lots of reasoning ... \\boxed{12",
            "finish_reason": "length",
            "correct": False,
            "recovery_max_tokens": 8192,
            "tokens_generated": 8192,
        }
        score_runs([run], CONFIG)
        # recovery keeps the judge verdict (0.0), not forced ungradable by A1b
        self.assertEqual(run["_correct"], 0.0)


if __name__ == "__main__":
    unittest.main()
