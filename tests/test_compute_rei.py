"""Tests for REI computation guard in Script 4."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import compute_rei  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
