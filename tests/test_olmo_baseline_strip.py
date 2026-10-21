#!/usr/bin/env python3
"""Regression tests for OLMo fair-copy strip + LaTeX grading fixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluator import is_correct  # noqa: E402
from llm_backend import prepare_baseline_raw_text  # noqa: E402
from model_family import ThinkTags  # noqa: E402


class OlmoThinkStripTests(unittest.TestCase):
    TAGS = ThinkTags(open="<think>", close="</think>")

    def test_proper_close_keeps_fair_copy(self) -> None:
        raw = "draft junk\n</think>\n\nStep 1: fair\nFinal answer: 4"
        visible = prepare_baseline_raw_text(raw, think_tags=self.TAGS)
        self.assertEqual(visible, "Step 1: fair\nFinal answer: 4")

    def test_malformed_close_hash(self) -> None:
        raw = "draft\n</think#\n\nStep 1: ok\nFinal answer: 1"
        visible = prepare_baseline_raw_text(raw, think_tags=self.TAGS)
        self.assertTrue(visible.startswith("Step 1:"))
        self.assertNotIn("</think", visible)

    def test_last_of_multiple_closes(self) -> None:
        raw = "a</think>\ndraft</think>\n\nStep 1: fair\nFinal answer: 7"
        visible = prepare_baseline_raw_text(raw, think_tags=self.TAGS)
        self.assertEqual(visible, "Step 1: fair\nFinal answer: 7")
        self.assertNotIn(">", visible[:3])


class OlmoLatexGradingTests(unittest.TestCase):
    def test_shorthand_frac_gold(self) -> None:
        self.assertTrue(is_correct("1/4", r"\frac14", "symbolic"))
        self.assertTrue(is_correct(r"\frac{1}{4}", r"\frac14", "symbolic"))
        self.assertTrue(is_correct("3/4", r"\frac 34", "symbolic"))
        self.assertTrue(is_correct(r"\frac{9}{19}", r"\frac9{19}", "symbolic"))

    def test_currency_symbolic_gold(self) -> None:
        self.assertTrue(is_correct("18.90", r"\$18.90", "symbolic"))
        self.assertTrue(is_correct("36", r"\$36", "symbolic"))
        self.assertTrue(is_correct("$18.90$", r"\$18.90", "symbolic"))


if __name__ == "__main__":
    unittest.main()
