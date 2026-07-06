"""Continuation-mode prompt construction and trace hygiene (Phase 1)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_backend import (  # noqa: E402
    THINK_CLOSE_TAG,
    DryRunBackend,
    build_continuation_prompt,
    strip_think_closure,
)
from model_family import set_active_model_family  # noqa: E402


def _reset_active_family_to_deepseek() -> None:
    """Cross-model tests may switch the active family; restore DeepSeek here.

    Legacy tests here assume `<think>` / `</think>` — the DeepSeek defaults.
    Without this reset, running them AFTER `tests/test_cross_model_family.py`
    leaves the process active family set to Ministral/Gemma and every
    strip_think_closure assertion breaks.
    """
    set_active_model_family("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")


class FakeTokenizer:
    """Minimal DeepSeek-like chat template that opens a think block."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        rendered = "\n".join(parts)
        if add_generation_prompt:
            rendered += "\n<|assistant|>\n<think>\n"
        return rendered


class BuildContinuationPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_active_family_to_deepseek()
        self.tok = FakeTokenizer()

    def test_prompt_ends_in_open_think_block(self) -> None:
        prompt = build_continuation_prompt(self.tok, "What is 2+2?", "")
        self.assertTrue(prompt.endswith("<think>\n"))
        self.assertNotIn(THINK_CLOSE_TAG, prompt)

    def test_trace_appended_raw_inside_block(self) -> None:
        trace = "First, I add the numbers.\n\nThen I check the sum."
        prompt = build_continuation_prompt(self.tok, "Q", trace)
        self.assertIn(trace, prompt)
        # Trace comes after the opening think tag.
        self.assertLess(prompt.index("<think>"), prompt.index(trace))
        # Still no closing tag — block stays open for the model to continue.
        self.assertNotIn(THINK_CLOSE_TAG, prompt)

    def test_system_prompt_included(self) -> None:
        prompt = build_continuation_prompt(self.tok, "Q", "trace", system="be terse")
        self.assertIn("be terse", prompt)

    def test_stop_prefill_closes_block_once(self) -> None:
        prefill = "</think>\n\nTherefore, the final answer is \\boxed{"
        prompt = build_continuation_prompt(self.tok, "Q", "some trace", stop_prefill=prefill)
        # Exactly one closing tag (from the prefill) — no double </think>.
        self.assertEqual(prompt.count(THINK_CLOSE_TAG), 1)
        self.assertTrue(prompt.endswith("\\boxed{"))

    def test_manual_fallback_without_tokenizer(self) -> None:
        prompt = build_continuation_prompt(None, "Q", "trace")
        self.assertIn("<think>", prompt)
        self.assertIn("trace", prompt)
        self.assertNotIn(THINK_CLOSE_TAG, prompt)


class StripThinkClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_active_family_to_deepseek()

    def test_cuts_at_closing_tag(self) -> None:
        text = "I keep thinking.\n\nDone.</think>\n\nThe answer is 5."
        self.assertEqual(strip_think_closure(text), "I keep thinking.\n\nDone.")

    def test_no_tag_returns_text(self) -> None:
        text = "Just reasoning, still open."
        self.assertEqual(strip_think_closure(text), text)

    def test_removes_stray_open_tag(self) -> None:
        text = "<think>extra open"
        self.assertEqual(strip_think_closure(text), "extra open")


class GenerateFromTraceBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_active_family_to_deepseek()

    def test_stop_prefill_prepended_to_output(self) -> None:
        backend = DryRunBackend()
        backend.gold_hint = "42"
        prefill = "</think>\n\nTherefore, the final answer is \\boxed{"
        results = backend.generate_from_trace_batch(
            [("Q", "trace", prefill)],
            max_new_tokens=64,
            temperature=0.0,
            top_p=1.0,
            seeds=[42],
        )
        self.assertEqual(len(results), 1)
        # Prefill (with \boxed{) is prepended; DryRun emits gold_hint + "}".
        self.assertIn("\\boxed{42}", results[0].text)

    def test_recovery_has_no_prefill(self) -> None:
        backend = DryRunBackend()
        results = backend.generate_from_trace_batch(
            [("Q", "trace", None)],
            max_new_tokens=64,
            temperature=0.0,
            top_p=1.0,
            seeds=[42],
        )
        self.assertNotIn("\\boxed{", results[0].text)


if __name__ == "__main__":
    unittest.main()
