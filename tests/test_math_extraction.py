"""Regression tests for math recovery answer extraction (evaluator.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluator import _extract_math_answer, is_correct  # noqa: E402


class MathExtractionTests(unittest.TestCase):
    def test_boxed_scalar(self):
        self.assertEqual(_extract_math_answer(r"work \boxed{10}"), "10")

    def test_p1_function_rhs(self):
        text = "**Answer:** \\( P(5) = 15 \\)"
        self.assertEqual(_extract_math_answer(text), "15")
        self.assertTrue(is_correct(_extract_math_answer(text), "15", "symbolic"))

    def test_p2_ignore_intermediate_y_equals(self):
        text = (
            "- x = 54: y = 1 (invalid)\n\n"
            "**Conclusion:**\nThus, there are 4 possible values of \\( x \\)."
        )
        self.assertEqual(_extract_math_answer(text), "4")

    def test_plane_equation_display_math(self):
        text = "**Final Answer:**\n\\[5x - 7y + 11z + 4 = 0\\]"
        self.assertEqual(_extract_math_answer(text), "5x - 7y + 11z + 4 = 0")

    def test_final_marker_region_beats_early_number(self):
        text = (
            "The median might be 13 from an earlier attempt.\n"
            "**Conclusion:** The median of the 27 consecutive positive integers is **81**."
        )
        self.assertEqual(_extract_math_answer(text), "81")

    def test_answer_line_prose_with_number(self):
        text = "**Answer:** The value of \\( x + y \\) is \\( 17 \\)."
        self.assertEqual(_extract_math_answer(text), "17")

    def test_fraction_scalar(self):
        text = "**Answer:** The probability is 1/2."
        self.assertEqual(_extract_math_answer(text), "1/2")
        self.assertTrue(is_correct(_extract_math_answer(text), r"\frac{1}{2}", "symbolic"))

    def test_colon_inside_bold_final_answer(self):
        # **Final Answer:** with value on same line (colon inside the bold).
        text = "7. Calculate x + y:\n   - 8 + 9 = 17\n\n**Final Answer:** 17"
        self.assertEqual(_extract_math_answer(text), "17")

    def test_bold_final_phrase_beats_intermediate(self):
        text = (
            "Sum = \\( 13.5 \\times 162 = 2187 \\).\n\n"
            "Thus, the median is confirmed to be **81**."
        )
        self.assertEqual(_extract_math_answer(text), "81")

    def test_definitional_formula_not_picked(self):
        text = (
            "- The area of a right triangle is "
            "\\(\\frac{1}{2} \\times \\text{base} \\times \\text{height}\\).\n"
            "Both methods confirm that the area of the triangle is **32**."
        )
        self.assertEqual(_extract_math_answer(text), "32")

    def test_count_values_bold(self):
        text = "Thus, there are **4 possible values of \\( x \\)**."
        self.assertEqual(_extract_math_answer(text), "4")

    def test_truncated_no_final_returns_nonmatching(self):
        # Truncated mid-calculation: gold (12) is only scratch, must not be returned.
        text = "Cross-multiplying:\n\\[\n\\sqrt{3} (3t - t^3) = 1 - 3t^2\n\\]\nLet me try"
        extracted = _extract_math_answer(text)
        self.assertNotEqual(extracted, "12")

    def test_l1_boxed_inside_think_block_truncated_after(self):
        # Stop prefill landed inside <think>; post-</think> presentation truncated.
        text = (
            "Step 1 ... Step 2 ...\n"
            "Therefore, the final answer is \\boxed{28800}.\n\n"
            "</think>\n\n"
            "Let me re-derive the result more carefully but"
        )
        self.assertEqual(_extract_math_answer(text), "28800")

    def test_l1_no_boxed_anywhere_stays_none(self):
        text = "</think>\n\nLet me try again from the start without finishing"
        self.assertIsNone(_extract_math_answer(text))

    def test_l2_prose_coordinate_tuple(self):
        text = "After substituting back, the coordinates of point P are (8, -2)."
        self.assertEqual(_extract_math_answer(text), "(8,-2)")
        self.assertTrue(is_correct(_extract_math_answer(text), "(8,-2)", "symbolic"))

    def test_l2_prose_trailing_percent(self):
        text = "Hence the minimal compound interest rate required is 10%"
        self.assertEqual(_extract_math_answer(text), "10")

    def test_l2_prose_ignores_intermediate_lines(self):
        # A number mid-sentence (not after a conclusion cue at line end) must not
        # be picked by the prose fallback; the conclusion itself has no answer.
        text = (
            "We had 50 candidates at the very beginning of this long search\n"
            "Now let us reconsider the whole approach again and"
        )
        self.assertIsNone(_extract_math_answer(text))


if __name__ == "__main__":
    unittest.main()
