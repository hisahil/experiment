"""Tests for Lean primary metrics (susceptibility, recovery_gain, rter) in Script 4."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import build_per_task_condition, compute_rter  # noqa: E402


class ComputeRterTests(unittest.TestCase):
    def test_rter_equals_one_when_efficiency_matches_clean(self):
        # recovery acc/token == clean acc/token -> RTER 1.0
        self.assertAlmostEqual(compute_rter(1.0, 1.0, 200.0, 200.0), 1.0)

    def test_rter_below_one_when_recovery_costs_more_tokens(self):
        # same accuracy but 2x the tokens -> half as efficient
        self.assertAlmostEqual(compute_rter(1.0, 1.0, 400.0, 200.0), 0.5)

    def test_rter_none_on_zero_tokens(self):
        self.assertIsNone(compute_rter(1.0, 1.0, 0.0, 200.0))
        self.assertIsNone(compute_rter(1.0, 1.0, 200.0, 0.0))

    def test_rter_none_on_zero_clean_accuracy(self):
        self.assertIsNone(compute_rter(0.5, 0.0, 200.0, 200.0))


def _run(**kw):
    base = {
        "model": "m",
        "domain": "math",
        "error_type": "off_topic",
        "k": 1,
        "length_level": "short",
        "task_id": "t1",
        "seed": 42,
        "gold_answer": "5",
        "eval_type": "numeric",
        "tokens_generated": 100,
    }
    base.update(kw)
    return base


class LeanColumnTests(unittest.TestCase):
    def _config(self):
        return {
            "difficulty_bins": {"easy": 0.7, "hard": 0.4},
            "min_denominator_tokens": 1,
            "math_tolerance": 1e-4,
            "science_tolerance": 1e-2,
        }

    def test_susceptibility_and_recovery_gain(self):
        # stop wrong (acc_stop=0), recovery correct (acc_recovery_all=1)
        runs = [
            _run(variant="stop", _correct=0.0, extracted_answer="9", tokens_generated=100),
            _run(variant="recovery", _correct=1.0, extracted_answer="5", tokens_generated=300,
                 _recovery_terminated_by="natural"),
        ]
        clean_map = {("t1", "m"): 1.0}
        tok_map = {("t1", "m"): 50.0}
        rows = build_per_task_condition(runs, clean_map, tok_map, self._config())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["susceptibility"], 1.0)   # 1 - acc_stop(0)
        self.assertAlmostEqual(row["recovery_gain"], 1.0)    # acc_rec_all(1) - acc_stop(0)
        self.assertIsNotNone(row["rter"])

    def test_recovery_gain_negative_when_overcorrected(self):
        runs = [
            _run(variant="stop", _correct=1.0, extracted_answer="5"),
            _run(variant="recovery", _correct=0.0, extracted_answer="9",
                 _recovery_terminated_by="natural", tokens_generated=300),
        ]
        clean_map = {("t1", "m"): 1.0}
        rows = build_per_task_condition(runs, clean_map, {("t1", "m"): 50.0}, self._config())
        self.assertAlmostEqual(rows[0]["recovery_gain"], -1.0)


if __name__ == "__main__":
    unittest.main()
