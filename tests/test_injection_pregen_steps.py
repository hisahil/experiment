"""Tests for Script 2 injection pre-generation trimming (adaptive vs three_turn).

Covers:
  - experiment_design.expected_pregenerated_steps resolution (auto / 1 / k).
  - grid_validation.validate_injection_record transition tolerance.
  - validate_injections_grid accepting a single-step record at k=3 in adaptive mode.
  - generate_injections prompt wording switching to single-pregen phrasing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_design import (  # noqa: E402
    DesignAxes,
    DIALOGUE_MODE_ADAPTIVE,
    DIALOGUE_MODE_THREE_TURN,
    expected_pregenerated_steps,
)
from grid_validation import (  # noqa: E402
    validate_injection_record,
    validate_injections_grid,
)


class ExpectedPregeneratedStepsTests(unittest.TestCase):
    def test_auto_adaptive_is_one(self) -> None:
        self.assertEqual(
            expected_pregenerated_steps(
                5, pregenerate_steps="auto", dialogue_mode=DIALOGUE_MODE_ADAPTIVE
            ),
            1,
        )

    def test_auto_three_turn_is_k(self) -> None:
        self.assertEqual(
            expected_pregenerated_steps(
                5, pregenerate_steps="auto", dialogue_mode=DIALOGUE_MODE_THREE_TURN
            ),
            5,
        )

    def test_explicit_one_and_k(self) -> None:
        self.assertEqual(
            expected_pregenerated_steps(
                3, pregenerate_steps="1", dialogue_mode=DIALOGUE_MODE_THREE_TURN
            ),
            1,
        )
        self.assertEqual(
            expected_pregenerated_steps(
                3, pregenerate_steps="k", dialogue_mode=DIALOGUE_MODE_ADAPTIVE
            ),
            3,
        )

    def test_numeric_input_clamped(self) -> None:
        self.assertEqual(expected_pregenerated_steps(3, pregenerate_steps=9), 3)
        self.assertEqual(expected_pregenerated_steps(3, pregenerate_steps=0), 1)
        self.assertEqual(expected_pregenerated_steps(3, pregenerate_steps=2), 2)

    def test_k_one_collapses(self) -> None:
        # A single-round dialogue pre-generates exactly its one step in any mode.
        self.assertEqual(
            expected_pregenerated_steps(
                1, pregenerate_steps="auto", dialogue_mode=DIALOGUE_MODE_ADAPTIVE
            ),
            1,
        )


class ValidateInjectionRecordTests(unittest.TestCase):
    def _rec(self, n_steps: int, *, step1_only: bool = False) -> dict:
        return {
            "task_id": "t1",
            "error_type": "off_topic",
            "k": 3,
            "length_level": "short",
            "target_model": "model-a",
            "injection_steps": [f"step {i}" for i in range(n_steps)],
            "step1_only": step1_only,
        }

    def test_strict_default_requires_k(self) -> None:
        self.assertEqual(validate_injection_record(self._rec(3), 3), [])
        self.assertTrue(validate_injection_record(self._rec(1), 3))

    def test_expected_steps_one_accepts_single(self) -> None:
        self.assertEqual(
            validate_injection_record(self._rec(1), 3, expected_steps=1), []
        )

    def test_expected_steps_one_tolerates_legacy_k(self) -> None:
        # Old full-k pilot records must still validate against an adaptive run.
        self.assertEqual(
            validate_injection_record(self._rec(3), 3, expected_steps=1), []
        )

    def test_expected_steps_rejects_partial_mismatch(self) -> None:
        # Two steps is neither the expected 1 nor the legacy k=3.
        issues = validate_injection_record(self._rec(2), 3, expected_steps=1)
        self.assertTrue(issues)

    def test_empty_step_flagged(self) -> None:
        rec = self._rec(1, step1_only=True)
        rec["injection_steps"] = ["   "]
        issues = validate_injection_record(
            rec, 3, allow_step1_only=True, expected_steps=1
        )
        self.assertTrue(any("empty" in i for i in issues))

    def test_allow_step1_only_branch(self) -> None:
        self.assertEqual(
            validate_injection_record(
                self._rec(1, step1_only=True), 3, allow_step1_only=True
            ),
            [],
        )


class ValidateInjectionsGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = [{"task_id": "t1", "domain": "math"}]
        self.axes = DesignAxes(
            variants=["stop", "recovery"],
            error_types=["off_topic"],
            k_injections=[3],
            length_levels=["short"],
            seeds=[42],
        )
        self.model = "model-a"

    def _single_step_record(self) -> dict:
        return {
            "task_id": "t1",
            "error_type": "off_topic",
            "k": 3,
            "length_level": "short",
            "target_model": "model-a",
            "injection_steps": ["only the round-1 step"],
            "step1_only": True,
        }

    def test_adaptive_accepts_single_step_at_k3(self) -> None:
        result = validate_injections_grid(
            [self._single_step_record()],
            self.tasks,
            self.axes,
            target_model=self.model,
            pregenerate_steps="auto",
            dialogue_mode=DIALOGUE_MODE_ADAPTIVE,
        )
        self.assertTrue(result.ok, msg=result.structural_errors)

    def test_three_turn_rejects_single_step_at_k3(self) -> None:
        result = validate_injections_grid(
            [self._single_step_record()],
            self.tasks,
            self.axes,
            target_model=self.model,
            pregenerate_steps="auto",
            dialogue_mode=DIALOGUE_MODE_THREE_TURN,
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.structural_errors)


class PromptWordingTests(unittest.TestCase):
    def setUp(self) -> None:
        import generate_injections as gi

        self.gi = gi
        self.task = {
            "task_id": "t1",
            "domain": "math",
            "prompt": "What is 2+2?",
            "gold_answer": "4",
            "decoy_answer": "5",
        }

    def test_single_pregen_wording(self) -> None:
        prompt = self.gi.build_step_prompt(
            self.task,
            "off_topic",
            k=3,
            length_level="short",
            step_index=0,
            prior_injections=[],
            baseline_context=None,
            n_pregen=1,
        )
        self.assertIn("opening adversarial round", prompt)
        self.assertNotIn("writing step 1 of 3", prompt)

    def test_full_k_wording_default(self) -> None:
        prompt = self.gi.build_step_prompt(
            self.task,
            "off_topic",
            k=3,
            length_level="short",
            step_index=0,
            prior_injections=[],
            baseline_context=None,
        )
        self.assertIn("step 1 of 3", prompt)

    def test_placeholder_single_pregen(self) -> None:
        label = self.gi.placeholder_step(
            self.task, "off_topic", 3, 0, [], None, n_pregen=1
        )
        self.assertIn("pregenerated round 1", label)


if __name__ == "__main__":
    unittest.main()
