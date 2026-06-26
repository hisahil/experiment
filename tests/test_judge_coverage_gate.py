"""Tests for the hard judge-coverage gate (A5)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import check_judge_coverage  # noqa: E402

CONFIG = {
    "judge_coverage_max_fallback_rate": 0.0,
    "judge_coverage_max_k_imbalance_pp": 5.0,
}


def _row(k, fallback):
    return {"model": "m", "k": k, "judge_fallback": fallback}


class JudgeCoverageGateTests(unittest.TestCase):
    def test_pass_when_no_fallbacks(self):
        runs = [_row(1, False), _row(3, False), _row(5, False)]
        rows, passed, msgs = check_judge_coverage(runs, CONFIG)
        self.assertTrue(passed)
        self.assertEqual(msgs, [])

    def test_fail_on_any_fallback(self):
        runs = [_row(1, False), _row(3, True)]
        rows, passed, msgs = check_judge_coverage(runs, CONFIG)
        self.assertFalse(passed)
        self.assertTrue(any("fallback rate" in m for m in msgs))

    def test_fail_on_k_imbalance(self):
        # allow some fallback but flag the k=3 imbalance specifically
        cfg = {"judge_coverage_max_fallback_rate": 1.0, "judge_coverage_max_k_imbalance_pp": 5.0}
        runs = [_row(1, False), _row(1, False)] + [_row(3, True), _row(3, True)]
        rows, passed, msgs = check_judge_coverage(runs, cfg)
        self.assertFalse(passed)
        self.assertTrue(any("imbalance" in m for m in msgs))

    def test_report_has_overall_and_per_k(self):
        runs = [_row(1, False), _row(3, False)]
        rows, _, _ = check_judge_coverage(runs, CONFIG)
        scopes = {r["scope"] for r in rows}
        self.assertIn("overall", scopes)
        self.assertIn("k", scopes)
        self.assertIn("model", scopes)


if __name__ == "__main__":
    unittest.main()
