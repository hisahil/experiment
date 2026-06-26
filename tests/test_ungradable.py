"""Tests for ungradable reason classification in Script 4."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import _classify_ungradable, compute_stop_ungradable_rate  # noqa: E402


class UngradableClassificationTests(unittest.TestCase):
    def test_finish_reason_length_before_token_cap_heuristic(self):
        run = {
            "variant": "stop",
            "gold_answer": "42",
            "raw_output": "partial answer without parseable final",
            "finish_reason": "length",
            "tokens_generated": 50,
        }
        self.assertEqual(
            _classify_ungradable(run, stop_max=100, recovery_max=8192),
            "truncated",
        )

    def test_token_cap_fallback_without_finish_reason(self):
        run = {
            "variant": "stop",
            "gold_answer": "42",
            "raw_output": "truncated at cap",
            "tokens_generated": 100,
        }
        self.assertEqual(
            _classify_ungradable(run, stop_max=100, recovery_max=8192),
            "truncated",
        )


class StopUngradableRateTests(unittest.TestCase):
    def test_rate_counts_only_stop_variant(self):
        runs = [
            {"variant": "stop", "_correct": 1.0},
            {"variant": "stop", "_correct": None},
            {"variant": "stop", "_correct": 0.0},
            {"variant": "recovery", "_correct": None},
        ]
        stats = compute_stop_ungradable_rate(runs)
        self.assertEqual(stats["n_stop"], 3)
        self.assertEqual(stats["n_stop_ungradable"], 1)
        self.assertAlmostEqual(stats["stop_ungradable_rate"], 0.3333, places=4)


if __name__ == "__main__":
    unittest.main()
