"""Tests for post-hoc grid validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).resolve().parents[1] / "src"))

from experiment_design import DesignAxes  # noqa: E402
from grid_validation import (  # noqa: E402
    validate_baselines_grid,
    validate_injections_grid,
    validate_runs_grid,
)


class GridValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = [{"task_id": "t1", "domain": "math"}]
        self.axes = DesignAxes(
            variants=["stop", "recovery"],
            error_types=["off_topic"],
            k_injections=[1],
            length_levels=["short"],
            seeds=[42],
        )
        self.model = "model-a"

    def test_injections_missing_and_extra(self) -> None:
        records = [
            {
                "task_id": "t1",
                "error_type": "off_topic",
                "k": 1,
                "length_level": "short",
                "target_model": "model-a",
                "injection_steps": ["step"],
            },
            {
                "task_id": "t2",
                "error_type": "off_topic",
                "k": 1,
                "length_level": "short",
                "target_model": "model-a",
                "injection_steps": ["step"],
            },
        ]
        result = validate_injections_grid(
            records, self.tasks, self.axes, target_model=self.model
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.actual_count, 2)
        self.assertTrue(any("t2" in item for item in result.extra))

    def test_runs_wrong_count(self) -> None:
        base = {
            "task_id": "t1",
            "model": "model-a",
            "error_type": "off_topic",
            "k": 1,
            "length_level": "short",
            "seed": 42,
        }
        records = [
            {**base, "variant": "stop"},
            {**base, "variant": "stop"},
            {**base, "variant": "recovery"},
        ]
        result = validate_runs_grid(
            records, self.tasks, self.axes, model=self.model, variants=["stop", "recovery"]
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.wrong_counts)
        self.assertEqual(result.wrong_counts[0][1], 2)

    def test_baselines_ok(self) -> None:
        records = [{"task_id": "t1", "model": "model-a", "verified": True}]
        result = validate_baselines_grid(records, self.tasks, model=self.model)
        self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
