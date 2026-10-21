"""Evaluator regression: x=<val> equivalence and prose minimum markers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluator import _extract_math_answer, is_correct  # noqa: E402


class VariableAssignmentEquivalenceTests(unittest.TestCase):
    def test_bare_value_matches_x_equals_gold(self) -> None:
        self.assertTrue(is_correct("5", "x=5", "symbolic"))

    def test_x_equals_pred_matches_x_equals_gold(self) -> None:
        self.assertTrue(is_correct("x=5", "x=5", "symbolic"))


class ProseMinimumExtractionTests(unittest.TestCase):
    def test_minimum_rotation_degrees(self) -> None:
        text = "Therefore, the minimum rotation is 72 degrees."
        self.assertEqual(_extract_math_answer(text), "72")


if __name__ == "__main__":
    unittest.main()
