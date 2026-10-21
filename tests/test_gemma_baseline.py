"""Regression tests for Gemma baseline prompt + degenerate guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cot_segmentation import detect_degenerate_trace  # noqa: E402
from evaluator import _extract_math_answer  # noqa: E402
from llm_backend import build_baseline_prompt, prepare_baseline_raw_text  # noqa: E402
from model_family import resolve_model_family, set_active_model_family  # noqa: E402


class FakeTokenizerGemma4:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    ):
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        rendered = "\n".join(parts)
        if add_generation_prompt:
            if enable_thinking:
                rendered += "\n<|assistant|>\n<|channel|>thought\n"
            else:
                rendered += "\n<|assistant|>\n"
        return rendered


class GemmaBaselinePromptTests(unittest.TestCase):
    MODEL = "google/gemma-4-12B-it"

    def setUp(self) -> None:
        set_active_model_family(self.MODEL)
        self.family = resolve_model_family(self.MODEL)
        self.template = (
            ROOT / "prompts/baseline_generation_gemma.txt"
        ).read_text(encoding="utf-8")

    def test_chat_template_opens_thinking_channel(self) -> None:
        prompt = build_baseline_prompt(
            FakeTokenizerGemma4(),
            self.template,
            "What is 2+2?",
            use_chat_template=True,
        )
        self.assertIn("<|think|>", prompt)
        self.assertIn("<|channel|>thought", prompt)
        self.assertIn("What is 2+2?", prompt)
        self.assertNotIn("<answer>", prompt)

    def test_prepare_baseline_raw_text_strips_thinking(self) -> None:
        raw = (
            "<|channel|>thought\nStep 1: compute\n"
            "<|channel|>Step 1: compute\nFinal answer: \\boxed{4}"
        )
        visible = prepare_baseline_raw_text(raw)
        self.assertIn("Final answer:", visible)
        self.assertNotIn("<|channel|>thought", visible)


class GemmaDegenerateGuardTests(unittest.TestCase):
    def test_prompt_echo_flagged(self) -> None:
        raw = "Final answer: <answer>\n" * 6
        degenerate, reason = detect_degenerate_trace(["one blob"], raw_text=raw)
        self.assertTrue(degenerate)
        self.assertEqual(reason, "prompt_echo")

    def test_intra_text_repeat_flagged(self) -> None:
        chunk = "a" * 250
        degenerate, reason = detect_degenerate_trace(["loop"], raw_text=chunk)
        self.assertTrue(degenerate)
        self.assertEqual(reason, "intra_text_repeat")


class GemmaExtractionTests(unittest.TestCase):
    def test_minimum_rotation_prose(self) -> None:
        text = "Therefore, the minimum rotation is 72 degrees."
        self.assertEqual(_extract_math_answer(text), "72")


if __name__ == "__main__":
    unittest.main()
