"""Cross-model portage tests (Phase 0).

Covers the four families that Phase 1 will smoke-test on GPU:
  - DeepSeek-R1-Distill  (`<think>` / `</think>`)             — legacy default
  - GLM-Z1-9B            (`<think>` / `</think>`)             — same tags, top_k=40
  - Gemma-4-12B          (`<|channel|>thought` / `<|channel|>` + `<|think|>`)
  - Ministral-3-14B      (`[THINK]` / `[/THINK]`)

Each FakeTokenizer emulates the family-specific chat_template contract that
`build_continuation_prompt` relies on (open think block after
`add_generation_prompt=True`, trigger injection for Gemma).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llm_backend import (  # noqa: E402
    build_continuation_prompt,
    strip_think_closure,
)
from model_family import (  # noqa: E402
    apply_baseline_sampling,
    make_stop_prefill_selector,
    resolve_model_family,
    set_active_model_family,
    model_slug,
)


# ── Family-specific FakeTokenizers ─────────────────────────────────────────────


class FakeTokenizerDeepSeek:
    """Minimal DeepSeek-R1-Distill chat template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        rendered = "\n".join(parts)
        if add_generation_prompt:
            rendered += "\n<|assistant|>\n<think>\n"
        return rendered


class FakeTokenizerGLMZ1:
    """GLM-Z1 chat template — template injects the think block directly, no
    explicit `<think>` prefix; the model's output goes into an open block."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        rendered = "\n".join(parts)
        if add_generation_prompt:
            # GLM-Z1 opens with a raw newline + think marker
            rendered += "\n<|assistant|>\n<think>\n"
        return rendered


class FakeTokenizerGemma4:
    """Gemma-4 chat template: `enable_thinking=True` opens `<|channel|>thought`."""

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


class FakeTokenizerMinistral:
    """Ministral chat template: opens `[THINK]` on generation prompt."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        rendered = "\n".join(parts)
        if add_generation_prompt:
            rendered += "\n<|assistant|>\n[THINK]"
        return rendered


# ── Tests ──────────────────────────────────────────────────────────────────────


class DeepSeekPortageTests(unittest.TestCase):
    def setUp(self) -> None:
        set_active_model_family("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    def test_prompt_ends_open_think(self) -> None:
        prompt = build_continuation_prompt(FakeTokenizerDeepSeek(), "Q", "trace")
        self.assertIn("<think>", prompt)
        self.assertNotIn("</think>", prompt)
        self.assertIn("trace", prompt)

    def test_stop_prefill_closes_once(self) -> None:
        sel = make_stop_prefill_selector("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        prefill = sel("math", "symbolic")
        self.assertTrue(prefill.startswith("</think>"))
        prompt = build_continuation_prompt(
            FakeTokenizerDeepSeek(), "Q", "trace", stop_prefill=prefill,
        )
        self.assertEqual(prompt.count("</think>"), 1)
        self.assertTrue(prompt.endswith("\\boxed{"))

    def test_strip_think_closure(self) -> None:
        text = "reasoning</think>\n\nanswer"
        self.assertEqual(strip_think_closure(text), "reasoning")


class GLMZ1PortageTests(unittest.TestCase):
    MODEL = "zai-org/GLM-Z1-9B-0414"

    def setUp(self) -> None:
        set_active_model_family(self.MODEL)

    def test_slug(self) -> None:
        self.assertEqual(model_slug(self.MODEL), "glm-z1-9b")

    def test_baseline_sampling_has_top_k(self) -> None:
        merged = apply_baseline_sampling({"name": self.MODEL}, self.MODEL)
        self.assertEqual(merged["temperature"], 0.6)
        self.assertEqual(merged["top_p"], 0.95)
        self.assertEqual(merged["top_k"], 40)

    def test_prompt_ends_open_think(self) -> None:
        prompt = build_continuation_prompt(FakeTokenizerGLMZ1(), "Q", "trace")
        self.assertIn("<think>", prompt)
        self.assertNotIn("</think>", prompt)
        self.assertIn("trace", prompt)

    def test_stop_prefill_uses_think_close(self) -> None:
        prefill = make_stop_prefill_selector(self.MODEL)("math", "symbolic")
        self.assertTrue(prefill.startswith("</think>"))


class Gemma4PortageTests(unittest.TestCase):
    MODEL = "google/gemma-4-12B-it"

    def setUp(self) -> None:
        set_active_model_family(self.MODEL)

    def test_slug(self) -> None:
        self.assertEqual(model_slug(self.MODEL), "gemma-4-12b")

    def test_baseline_sampling_has_top_k(self) -> None:
        merged = apply_baseline_sampling({"name": self.MODEL}, self.MODEL)
        self.assertEqual(merged["temperature"], 1.0)
        self.assertEqual(merged["top_p"], 0.95)
        self.assertEqual(merged["top_k"], 64)

    def test_trigger_in_system_prompt(self) -> None:
        prompt = build_continuation_prompt(FakeTokenizerGemma4(), "Q", "trace")
        self.assertIn("<|think|>", prompt)  # trigger injected into system role
        self.assertIn("<|channel|>thought", prompt)  # opened via enable_thinking
        self.assertNotIn("<|channel|>\n\n", prompt)  # close tag not present yet
        self.assertIn("trace", prompt)

    def test_stop_prefill_uses_channel_close(self) -> None:
        prefill = make_stop_prefill_selector(self.MODEL)("math", "symbolic")
        self.assertTrue(prefill.startswith("<|channel|>"))
        self.assertTrue(prefill.endswith("\\boxed{"))

    def test_strip_think_closure_channel(self) -> None:
        # After switching to Gemma, strip_think_closure must respect <|channel|>
        text = "reasoning<|channel|>final answer"
        self.assertEqual(strip_think_closure(text), "reasoning")


class MinistralPortageTests(unittest.TestCase):
    MODEL = "mistralai/Ministral-3-14B-Reasoning-2512"

    def setUp(self) -> None:
        set_active_model_family(self.MODEL)

    def test_slug_no_collision_with_deepseek_14b(self) -> None:
        self.assertEqual(model_slug(self.MODEL), "ministral-3-14b")
        # Legacy DeepSeek 14B keeps its per-checkpoint slug (data/baselines/14b/).
        self.assertEqual(
            model_slug("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"), "14b",
        )

    def test_keep_in_history_flag(self) -> None:
        family = resolve_model_family(self.MODEL)
        self.assertTrue(family.think_tags.keep_in_history)

    def test_prompt_ends_open_think(self) -> None:
        prompt = build_continuation_prompt(FakeTokenizerMinistral(), "Q", "trace")
        self.assertIn("[THINK]", prompt)
        self.assertNotIn("[/THINK]", prompt)
        self.assertIn("trace", prompt)

    def test_stop_prefill_uses_bracket_close(self) -> None:
        prefill = make_stop_prefill_selector(self.MODEL)("math", "symbolic")
        self.assertTrue(prefill.startswith("[/THINK]"))

    def test_strip_think_closure_bracket(self) -> None:
        text = "reasoning[/THINK]final"
        self.assertEqual(strip_think_closure(text), "reasoning")

    def test_vllm_flags_mistral_mode(self) -> None:
        family = resolve_model_family(self.MODEL)
        flags = list(family.vllm.flags)
        self.assertIn("--tokenizer_mode", flags)
        self.assertIn("mistral", flags)
        self.assertIn("--reasoning-parser", flags)


class CrossFamilySlugTests(unittest.TestCase):
    """Guards against the historical Ministral-14B ↔ DeepSeek-14B collision."""

    def test_all_slugs_unique(self) -> None:
        models = [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "zai-org/GLM-Z1-9B-0414",
            "google/gemma-4-12B-it",
            "mistralai/Ministral-3-14B-Reasoning-2512",
            "microsoft/Phi-4-reasoning-plus",
        ]
        slugs = [model_slug(m) for m in models]
        self.assertEqual(len(slugs), len(set(slugs)), f"slug collision: {slugs}")


if __name__ == "__main__":
    unittest.main()
