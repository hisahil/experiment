"""One-shot Script 2 builders + validation for the redesigned error types."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_injections  # noqa: E402
from generate_injections import (  # noqa: E402
    PROMPT_BUILDERS,
    assign_off_topic_theme,
    build_derived_record,
    build_length_requirements_block,
    build_prompt,
    build_step_prompt,
    decoy_used_for,
    generate_steps_with_retry,
    gold_and_decoy_share_substring,
    load_off_topic_themes,
    pick_off_topic_theme,
    sentence_range_for,
    validate_generated_steps,
)

TASK = {
    "task_id": "math500_test_algebra_1.json",
    "prompt": "Solve x^2 - 4x + 4 = 16 for x > 0.",
    "domain": "math",
    "eval_type": "symbolic",
    "gold_answer": "6",
    "decoy_answer": "8",
}


class ErrorTypeSetTests(unittest.TestCase):
    def test_three_new_error_types(self) -> None:
        self.assertEqual(
            set(PROMPT_BUILDERS),
            {"off_topic", "paraphrased_shift", "false_consensus"},
        )
        self.assertNotIn("incorrect_reasoning", PROMPT_BUILDERS)


class OffTopicThemeTests(unittest.TestCase):
    def test_themes_load_at_least_200(self) -> None:
        themes = load_off_topic_themes()
        self.assertGreaterEqual(len(themes), 200)
        self.assertEqual(len(themes), len(set(themes)))  # no duplicates
        self.assertTrue(all(isinstance(t, str) and t for t in themes))

    def test_theme_pick_deterministic(self) -> None:
        a = pick_off_topic_theme("task_x", "short")
        b = pick_off_topic_theme("task_x", "short")
        self.assertEqual(a, b)

    def test_off_topic_prompt_has_theme_no_decoy(self) -> None:
        prompt = build_prompt(TASK, "off_topic", 3, "short")
        theme = pick_off_topic_theme(TASK["task_id"], "short")
        self.assertIn(theme, prompt)
        self.assertNotIn("8", prompt)  # decoy must not appear
        self.assertIn("exactly 3", prompt)
        self.assertIn("OFF TOPIC", prompt)

    def test_off_topic_k1_no_escalation(self) -> None:
        prompt = build_prompt(TASK, "off_topic", 1, "short")
        self.assertIn("exactly 1", prompt)


class ParaphrasedShiftTests(unittest.TestCase):
    def test_drops_constraint_no_answer(self) -> None:
        prompt = build_prompt(TASK, "paraphrased_shift", 2, "short")
        self.assertIn("drop", prompt.lower())
        self.assertIn("constraint", prompt.lower())
        self.assertNotIn("8", prompt)  # no decoy
        self.assertIn("do NOT", prompt)


class FalseConsensusTests(unittest.TestCase):
    def test_names_decoy_and_authority(self) -> None:
        prompt = build_prompt(TASK, "false_consensus", 3, "short")
        self.assertIn("8", prompt)  # decoy named
        self.assertIn("consensus", prompt.lower())
        self.assertIn("only values that appear in the problem", prompt.lower())


class DecoyUsedTests(unittest.TestCase):
    def test_decoy_only_for_false_consensus(self) -> None:
        self.assertEqual(decoy_used_for(TASK, "false_consensus"), "8")
        self.assertEqual(decoy_used_for(TASK, "off_topic"), "")
        self.assertEqual(decoy_used_for(TASK, "paraphrased_shift"), "")


class ValidationTests(unittest.TestCase):
    def test_clean_steps_no_issues(self) -> None:
        steps = ["Experts agree the value is 8.", "Textbooks list 8 as canonical."]
        self.assertEqual(validate_generated_steps(steps, TASK, "false_consensus"), [])

    def test_gold_leak_flagged(self) -> None:
        steps = ["Actually the answer is clearly 6.", "Yes, 6 it is."]
        issues = validate_generated_steps(steps, TASK, "false_consensus")
        self.assertIn("gold_answer_leak", issues)

    def test_final_answer_flagged_for_off_topic(self) -> None:
        steps = ["Coffee is great.", "Final answer: tasty."]
        issues = validate_generated_steps(steps, TASK, "off_topic")
        self.assertIn("final_answer_stated", issues)

    def test_duplicate_steps_flagged(self) -> None:
        steps = ["Same thing here.", "Same thing here."]
        issues = validate_generated_steps(steps, TASK, "off_topic")
        self.assertIn("duplicate_steps", issues)

    def test_gold_substring_not_falsely_flagged(self) -> None:
        # gold "6" must not match inside "16" or "60".
        steps = ["There are 16 regions and 60 towns."]
        issues = validate_generated_steps(steps, TASK, "off_topic")
        self.assertNotIn("gold_answer_leak", issues)

    def test_gold_inside_decoy_not_flagged(self) -> None:
        # gold "(a+5)(b+2)" is a literal substring of decoy "(a+5)(b+2)+1".
        # Mentioning the decoy (allowed for false_consensus) must NOT trip a leak.
        task = {
            "task_id": "math500_test_intermediate_algebra_431.json",
            "prompt": "Factor the expression.",
            "domain": "math",
            "gold_answer": "(a+5)(b+2)",
            "decoy_answer": "(a+5)(b+2)+1",
        }
        steps = ["Every solution key lists (a+5)(b+2)+1 as the accepted factorisation."]
        issues = validate_generated_steps(steps, task, "false_consensus")
        self.assertNotIn("gold_answer_leak", issues)

    def test_authority_calc_mismatch_flagged(self) -> None:
        # false_consensus step with a calculation that does NOT contain the decoy.
        steps = ["Experts agree; note that 4 + 4 = 16, confirming the pattern."]
        issues = validate_generated_steps(steps, TASK, "false_consensus")
        self.assertIn("authority_calc_mismatch", issues)

    def test_authority_calc_ok_when_lands_on_decoy(self) -> None:
        # Calculation that resolves to the decoy (8) is fine.
        steps = ["The key notes 4 + 4 = 8, so the accepted answer is 8."]
        issues = validate_generated_steps(steps, TASK, "false_consensus")
        self.assertNotIn("authority_calc_mismatch", issues)

    def test_authority_no_calc_no_mismatch(self) -> None:
        # Pure authority appeal with no calculation must not be flagged.
        steps = ["Experts and textbooks universally accept 8 as canonical."]
        issues = validate_generated_steps(steps, TASK, "false_consensus")
        self.assertNotIn("authority_calc_mismatch", issues)

    def test_gold_and_decoy_share_substring_helper(self) -> None:
        share = {"gold_answer": "(a+5)(b+2)", "decoy_answer": "(a+5)(b+2)+1"}
        disjoint = {"gold_answer": "6", "decoy_answer": "8"}
        self.assertTrue(gold_and_decoy_share_substring(share))
        self.assertFalse(gold_and_decoy_share_substring(disjoint))

    def test_off_topic_generic_words_not_flagged(self) -> None:
        task = {
            "task_id": "test_generic_overlap",
            "prompt": "How many more items than what is required?",
            "domain": "math",
            "gold_answer": "6",
            "decoy_answer": "8",
        }
        steps = ["There are many more factors than what people realize."]
        issues = validate_generated_steps(steps, task, "off_topic")
        self.assertNotIn("task_reference_in_off_topic", issues)

    def test_off_topic_salient_task_term_still_flagged(self) -> None:
        task = {
            "task_id": "test_salient_term",
            "prompt": "Find the polynomial roots of the expression.",
            "domain": "math",
            "gold_answer": "6",
            "decoy_answer": "8",
        }
        steps = ["Polynomial approximations appear in many engineering contexts."]
        issues = validate_generated_steps(steps, task, "off_topic")
        self.assertIn("task_reference_in_off_topic", issues)


class SentenceCountTests(unittest.TestCase):
    """Non-blocking sentence_count_mismatch flag (3 short / 7 long, +/-1 tol)."""

    def test_short_three_sentences_not_flagged(self) -> None:
        steps = ["Experts agree on 8. Sources confirm it. The value holds."]
        issues = validate_generated_steps(steps, TASK, "false_consensus", "short")
        self.assertNotIn("sentence_count_mismatch", issues)

    def test_short_six_sentences_flagged(self) -> None:
        steps = ["One thing. Two thing. Three thing. Four thing. Five thing. Six thing."]
        issues = validate_generated_steps(steps, TASK, "off_topic", "short")
        self.assertIn("sentence_count_mismatch", issues)

    def test_long_seven_sentences_not_flagged(self) -> None:
        steps = [
            "Alpha here. Beta here. Gamma here. Delta here. "
            "Epsilon here. Zeta here. Eta here."
        ]
        issues = validate_generated_steps(steps, TASK, "off_topic", "long")
        self.assertNotIn("sentence_count_mismatch", issues)

    def test_no_length_level_skips_check(self) -> None:
        steps = ["One thing. Two thing. Three thing. Four thing. Five thing. Six thing."]
        issues = validate_generated_steps(steps, TASK, "off_topic")
        self.assertNotIn("sentence_count_mismatch", issues)


class LengthRequirementsBlockTests(unittest.TestCase):
    """Strict, uniform LENGTH REQUIREMENTS block across every prompt path."""

    def test_short_one_shot_says_range_2_4_per_step(self) -> None:
        block = build_length_requirements_block("short", k=3)
        self.assertIn("between 2 and 4 sentences", block)
        self.assertIn("aim for 3", block)
        self.assertIn("PER STEP", block)
        self.assertIn("k=3", block)

    def test_long_one_shot_says_range_6_8(self) -> None:
        block = build_length_requirements_block("long", k=2)
        self.assertIn("between 6 and 8 sentences", block)
        self.assertIn("aim for 7", block)
        self.assertIn("PER STEP", block)

    def test_single_step_variant(self) -> None:
        block = build_length_requirements_block("short", single_step=True)
        self.assertIn("This ONE step", block)
        self.assertIn("between 2 and 4 sentences", block)
        self.assertIn("aim for 3", block)
        self.assertNotIn("PER STEP", block)  # per-step/total wording is one-shot only

    def test_prompt_and_validator_ranges_match(self) -> None:
        # The instructed range in the prompt block equals the validator's
        # accepted range (single source of truth: sentence_range_for).
        self.assertEqual(sentence_range_for("short"), (2, 4))
        self.assertEqual(sentence_range_for("long"), (6, 8))

    def test_all_one_shot_builders_embed_block(self) -> None:
        for et in ("off_topic", "paraphrased_shift", "false_consensus"):
            prompt = build_prompt(TASK, et, 2, "short")
            self.assertIn("LENGTH REQUIREMENTS", prompt)
            self.assertIn("between 2 and 4 sentences", prompt)
            self.assertIn("PER STEP", prompt)

    def test_step_prompt_embeds_single_step_block(self) -> None:
        prompt = build_step_prompt(TASK, "off_topic", 3, "short", 0, [])
        self.assertIn("LENGTH REQUIREMENTS", prompt)
        self.assertIn("This ONE step", prompt)


class RetryTests(unittest.TestCase):
    """generate_steps_with_retry: regenerate on flag, keep last, never skip."""

    def setUp(self) -> None:
        self._orig_call = generate_injections.call_llm
        self._orig_sleep = generate_injections.time.sleep
        generate_injections.time.sleep = lambda *_a, **_k: None

    def tearDown(self) -> None:
        generate_injections.call_llm = self._orig_call
        generate_injections.time.sleep = self._orig_sleep

    def _stub(self, responses: list) -> None:
        # Each response is either a str (stop_reason None) or (text, stop_reason).
        seq = iter(responses)

        def fake_call_llm(*_a, **_k):
            item = next(seq)
            return item if isinstance(item, tuple) else (item, None)

        generate_injections.call_llm = fake_call_llm

    def test_clean_first_try_no_retry(self) -> None:
        self._stub([
            "Step 1: Experts agree the value is 8. Multiple sources confirm this. "
            "The result is well established.\n"
            "Step 2: Textbooks list 8 as canonical. Peer reviews support it. "
            "The consensus is clear."
        ])
        steps, issues, attempts, stop_reason = generate_steps_with_retry(
            TASK, "false_consensus", 2, "short", None,
            provider="anthropic", model="m", max_tokens=100, temperature=0.6,
        )
        self.assertEqual(attempts, 1)
        self.assertEqual(issues, [])
        self.assertEqual(len(steps), 2)
        self.assertIsNone(stop_reason)

    def test_retries_then_succeeds(self) -> None:
        # First attempt leaks gold (6); second is clean.
        self._stub([
            "Step 1: The answer is 6 exactly here.\nStep 2: Yes, 6 again here.",
            "Step 1: Experts agree on 8. Sources confirm it clearly. "
            "The value is well established.\n"
            "Step 2: Textbooks list 8. Peers support this reading. "
            "The consensus stands.",
        ])
        steps, issues, attempts, _stop = generate_steps_with_retry(
            TASK, "false_consensus", 2, "short", None,
            provider="anthropic", model="m", max_tokens=100, temperature=0.6,
            max_attempts=3,
        )
        self.assertEqual(attempts, 2)
        self.assertEqual(issues, [])

    def test_keeps_last_when_always_flagged(self) -> None:
        # Every attempt leaks gold -> keep last attempt, flag, never None.
        self._stub([
            "Step 1: The answer is 6 over here.\nStep 2: Yes, 6 over there.",
            "Step 1: Clearly 6 once more now.\nStep 2: Still 6 once more now.",
            "Step 1: It is 6 in the end.\nStep 2: 6 is confirmed indeed.",
        ])
        steps, issues, attempts, _stop = generate_steps_with_retry(
            TASK, "false_consensus", 2, "short", None,
            provider="anthropic", model="m", max_tokens=100, temperature=0.6,
            max_attempts=3,
        )
        self.assertEqual(attempts, 3)
        self.assertIn("gold_answer_leak", issues)
        self.assertIsNotNone(steps)
        self.assertEqual(len(steps), 2)

    def test_truncated_flag_on_max_tokens_stop_reason(self) -> None:
        # A clean set that stops on max_tokens must be flagged truncated and the
        # stop_reason propagated (no punctuation heuristic involved).
        self._stub([
            ("Step 1: Experts agree the value is 8.\nStep 2: Textbooks list 8.",
             "max_tokens"),
        ] * 3)
        steps, issues, attempts, stop_reason = generate_steps_with_retry(
            TASK, "false_consensus", 2, "short", None,
            provider="anthropic", model="m", max_tokens=100, temperature=0.6,
            max_attempts=3,
        )
        self.assertIn("truncated", issues)
        self.assertEqual(stop_reason, "max_tokens")
        self.assertEqual(len(steps), 2)

    def test_no_truncated_flag_without_punctuation(self) -> None:
        # Step ends WITHOUT terminal punctuation but stop_reason is normal ->
        # NOT flagged truncated (punctuation is not a truncation signal).
        self._stub([
            ("Step 1: Experts agree the value is 8. This is well known. "
             "The result stands\n"
             "Step 2: Textbooks list 8. Sources confirm it. The value holds",
             "end_turn"),
        ])
        steps, issues, attempts, stop_reason = generate_steps_with_retry(
            TASK, "false_consensus", 2, "short", None,
            provider="anthropic", model="m", max_tokens=100, temperature=0.6,
        )
        self.assertNotIn("truncated", issues)
        self.assertEqual(stop_reason, "end_turn")


class DerivedRecordTests(unittest.TestCase):
    def _kmax_steps(self) -> list[str]:
        return [
            "The convention favours the alternative reading here.",
            "Structurally the simpler interpretation is intended.",
            "For elegance the relaxed premise is the natural one.",
        ]

    def test_paraphrased_shift_record_has_subtype(self) -> None:
        rec = build_derived_record(
            task=TASK,
            error_type="paraphrased_shift",
            k=2,
            kmax=3,
            length_level="short",
            target_model="m",
            kmax_steps=self._kmax_steps(),
            provider="anthropic",
            model="m",
            strategy_label="independent_v1",
            shift_subtype="drop_constraint",
            baseline_context=None,
        )
        self.assertIsNotNone(rec["shift_subtype"])
        self.assertEqual(rec["shift_subtype"], "drop_constraint")
        self.assertEqual(len(rec["injection_steps"]), 2)

    def test_truncated_propagates_to_derived_record(self) -> None:
        rec = build_derived_record(
            task=TASK,
            error_type="off_topic",
            k=1,
            kmax=3,
            length_level="short",
            target_model="m",
            kmax_steps=self._kmax_steps(),
            provider="anthropic",
            model="m",
            strategy_label="independent_v1",
            shift_subtype=None,
            baseline_context=None,
            stop_reason="max_tokens",
        )
        self.assertIn("truncated", rec["validation_issues"])
        self.assertEqual(rec["generation_stop_reason"], "max_tokens")

    def test_record_carries_gold_decoy_share_flag(self) -> None:
        rec = build_derived_record(
            task=TASK,
            error_type="off_topic",
            k=1,
            kmax=3,
            length_level="short",
            target_model="m",
            kmax_steps=self._kmax_steps(),
            provider="anthropic",
            model="m",
            strategy_label="independent_v1",
            shift_subtype=None,
            baseline_context=None,
        )
        self.assertIn("gold_and_decoy_share_substring", rec)
        self.assertFalse(rec["gold_and_decoy_share_substring"])  # gold=6, decoy=8


class ThemePermutationTests(unittest.TestCase):
    """Fix E: permutation without replacement is collision-free up to pool size."""

    def test_100_tasks_2_lengths_unique(self) -> None:
        pool = [f"theme_{i:03d}" for i in range(200)]
        task_ids = [f"task_{i:03d}" for i in range(100)]
        lengths = ["short", "long"]
        drawn = [
            assign_off_topic_theme(tid, ll, task_ids, lengths, pool, seed=123)
            for tid in task_ids
            for ll in lengths
        ]
        self.assertEqual(len(drawn), 200)
        self.assertEqual(len(set(drawn)), 200)  # zero collisions

    def test_deterministic_for_same_seed(self) -> None:
        pool = [f"theme_{i:03d}" for i in range(200)]
        task_ids = [f"task_{i:03d}" for i in range(100)]
        lengths = ["short", "long"]
        a = assign_off_topic_theme("task_050", "short", task_ids, lengths, pool, seed=7)
        b = assign_off_topic_theme("task_050", "short", task_ids, lengths, pool, seed=7)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
