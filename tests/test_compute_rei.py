"""Tests for REI computation guard in Script 4."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import classify_recovery_pair, compute_rei  # noqa: E402


class ComputeReiTests(unittest.TestCase):
    def test_small_positive_delta_with_min_one(self):
        # delta=5: efficient recovery near stop cap should compute REI
        result = compute_rei(1.0, 0.0, 105.0, 100.0, min_denominator_tokens=1)
        self.assertAlmostEqual(result, 0.2)

    def test_delta_nine_with_min_one(self):
        result = compute_rei(1.0, 0.0, 109.0, 100.0, min_denominator_tokens=1)
        self.assertAlmostEqual(result, 1.0 / 9.0)

    def test_zero_delta_excluded_with_min_one(self):
        self.assertIsNone(compute_rei(1.0, 0.0, 100.0, 100.0, min_denominator_tokens=1))

    def test_negative_delta_excluded_with_min_one(self):
        self.assertIsNone(compute_rei(1.0, 0.0, 97.0, 100.0, min_denominator_tokens=1))

    def test_delta_nine_excluded_with_min_ten_regression(self):
        self.assertIsNone(compute_rei(1.0, 0.0, 109.0, 100.0, min_denominator_tokens=10))


class ClassifyRecoveryPairTests(unittest.TestCase):
    def test_recovered_when_rec_correct_stop_wrong(self):
        self.assertEqual(
            classify_recovery_pair(
                rec_correct=True, stop_correct=False, rec_extracted="10", stop_extracted="5"
            ),
            "RECOVERED",
        )

    def test_persisted_when_rec_wrong_and_same_as_stop(self):
        self.assertEqual(
            classify_recovery_pair(
                rec_correct=False, stop_correct=False, rec_extracted="5", stop_extracted="5"
            ),
            "PERSISTED",
        )

    def test_persisted_when_both_none(self):
        self.assertEqual(
            classify_recovery_pair(
                rec_correct=False, stop_correct=False, rec_extracted=None, stop_extracted=None
            ),
            "PERSISTED",
        )

    def test_migrated_when_rec_wrong_but_different_answer(self):
        self.assertEqual(
            classify_recovery_pair(
                rec_correct=False, stop_correct=False, rec_extracted="7", stop_extracted="5"
            ),
            "MIGRATED",
        )

    def test_migrated_when_never_damaged(self):
        # Both correct: not RECOVERED (stop not wrong), not PERSISTED (rec correct).
        self.assertEqual(
            classify_recovery_pair(
                rec_correct=True, stop_correct=True, rec_extracted="10", stop_extracted="10"
            ),
            "MIGRATED",
        )


if __name__ == "__main__":
    unittest.main()
