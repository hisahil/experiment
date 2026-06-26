"""Tests for confirmatory significance (McNemar + Wilson, task-clustered)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import (  # noqa: E402
    _format_p,
    mcnemar_exact_p,
    run_confirmatory_tests,
    wilson_ci,
)


class WilsonTests(unittest.TestCase):
    def test_all_successes_ci_below_one(self):
        lo, hi = wilson_ci(10, 10)
        self.assertGreater(lo, 0.6)
        self.assertLessEqual(hi, 1.0 + 1e-9)

    def test_zero_n_is_nan(self):
        lo, hi = wilson_ci(0, 0)
        self.assertNotEqual(lo, lo)  # NaN


class McNemarTests(unittest.TestCase):
    def test_no_discordant_is_nan(self):
        self.assertNotEqual(mcnemar_exact_p(0, 0), mcnemar_exact_p(0, 0))

    def test_four_zero_two_sided(self):
        # exact two-sided binomial with all 4 on one side = 2 * 0.5^4 = 0.125
        self.assertAlmostEqual(mcnemar_exact_p(4, 0), 0.125, places=6)

    def test_symmetric_gives_one(self):
        self.assertAlmostEqual(mcnemar_exact_p(5, 5), 1.0, places=6)


class FormatPTests(unittest.TestCase):
    def test_never_literal_zero(self):
        self.assertEqual(_format_p(0.0), "<1e-4")
        self.assertEqual(_format_p(1e-12), "<1e-4")

    def test_readable_above_threshold(self):
        self.assertEqual(_format_p(0.05), "0.0500")


class TaskClusteringTests(unittest.TestCase):
    def test_n_clusters_caps_at_task_count(self):
        # 3 tasks, each with multiple conditions -> n_clusters must be 3, not 6.
        per_task = []
        for task in ("a", "b", "c"):
            for et in ("off_topic", "incorrect_reasoning"):
                per_task.append(
                    {"task_id": task, "error_type": et, "k": 1, "recovery_gain": 0.5}
                )
        rows = run_confirmatory_tests(per_task, runs=[])
        overall = next(r for r in rows if r["scope"] == "overall")
        self.assertEqual(overall["n_clusters"], 3)
        # p is never literally 0.0 in the display column
        self.assertNotEqual(overall["mcnemar_p_display"], "0.0")


if __name__ == "__main__":
    unittest.main()
