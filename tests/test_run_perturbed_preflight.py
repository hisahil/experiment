"""Tests for Script 3 injection preflight."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_perturbed import (  # noqa: E402
    index_injections,
    injection_key,
    preflight_injections,
    report_injection_preflight,
    validate_injection_record,
)


class InjectionPreflightTests(unittest.TestCase):
    def test_validate_injection_record_ok(self):
        rec = {"injection_steps": ["a", "b", "c"]}
        self.assertEqual(validate_injection_record(rec, 3), [])

    def test_validate_injection_record_bad(self):
        rec = {"injection_steps": ["a", ""]}
        issues = validate_injection_record(rec, 2)
        self.assertTrue(any("empty" in i for i in issues))

    def test_index_injections_filters_target_model(self):
        records = [
            {
                "task_id": "t1",
                "error_type": "off_topic",
                "k": 1,
                "length_level": "short",
                "target_model": "model-a",
                "injection_steps": ["x"],
            },
            {
                "task_id": "t1",
                "error_type": "off_topic",
                "k": 1,
                "length_level": "short",
                "target_model": "model-b",
                "injection_steps": ["y"],
            },
        ]
        idx = index_injections(records, target_model="model-a")
        key = injection_key("t1", "off_topic", 1, "short")
        self.assertEqual(idx[key]["injection_steps"], ["x"])

    def test_preflight_missing_and_mismatch(self):
        tasks = [{"task_id": "t1", "domain": "math"}]
        lists = {
            "error_types": ["off_topic"],
            "k_injections": [1],
            "length_levels": ["short"],
        }
        baseline_index = {
            "t1": {"steps": ["s1", "s2", "s3", "s4"]},
        }
        injection_index = {
            injection_key("t1", "off_topic", 1, "short"): {
                "injection_steps": ["inj"],
                "insertion_step": 1,
            }
        }
        missing, invalid, mismatches = preflight_injections(
            tasks, lists, injection_index, baseline_index
        )
        self.assertEqual(missing, [])
        self.assertEqual(invalid, [])
        self.assertEqual(len(mismatches), 1)
        self.assertIn("insertion_step=1", mismatches[0])
        self.assertIn("baseline insertion_step=2", mismatches[0])

    def test_report_preflight_aborts_on_missing(self):
        abort = report_injection_preflight(
            missing=["t1|off_topic|1|short"],
            invalid=[],
            insertion_mismatches=[],
            strict_insertion=False,
        )
        self.assertTrue(abort)

    def test_report_preflight_warn_only_on_mismatch(self):
        abort = report_injection_preflight(
            missing=[],
            invalid=[],
            insertion_mismatches=["t1|off_topic|1|short: x"],
            strict_insertion=False,
        )
        self.assertFalse(abort)


if __name__ == "__main__":
    unittest.main()
