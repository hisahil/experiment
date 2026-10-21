"""First-Commit stop extraction and the corrected H3 task slopes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluator import extract_domain_answer  # noqa: E402
from evaluate import _task_k_slopes, build_pair_coverage  # noqa: E402
from first_commit import extract_stop_commitment  # noqa: E402
from model_family import build_stop_prefill  # noqa: E402


class FirstCommitTests(unittest.TestCase):
    def test_later_self_correction_does_not_replace_first_box(self):
        prefill = build_stop_prefill("math", "symbolic")
        raw = prefill + r"7} Wait, recalculating gives \boxed{11}"
        commitment = extract_stop_commitment(raw, domain="math", eval_type="symbolic")
        self.assertEqual(commitment["answer"], "7")
        self.assertNotIn("11", commitment["span"])
        self.assertEqual(
            extract_domain_answer(raw, "math", "symbolic", prefer_first_boxed=True),
            "7",
        )

    def test_nested_braces_stay_inside_the_first_box(self):
        prefill = build_stop_prefill("math", "symbolic")
        raw = prefill + r"\frac{\frac{1}{2}}{3}} later \boxed{9}"
        commitment = extract_stop_commitment(raw, domain="math", eval_type="symbolic")
        self.assertEqual(commitment["answer"], r"\frac{\frac{1}{2}}{3}")

    def test_unclosed_box_is_incomplete(self):
        prefill = build_stop_prefill("math", "symbolic")
        commitment = extract_stop_commitment(prefill + "12", domain="math")
        self.assertTrue(commitment["incomplete"])
        self.assertIsNone(commitment["answer"])

    def test_empty_output(self):
        commitment = extract_stop_commitment("  \n")
        self.assertTrue(commitment["empty_output"])


class StatisticRepairTests(unittest.TestCase):
    def test_one_slope_per_task(self):
        rows = []
        for task, gains in {"a": (0.8, 0.4), "b": (0.2, 0.2)}.items():
            for k, gain in zip((1, 3), gains):
                rows.append({"task_id": task, "k": k, "recovery_gain": gain})
        slopes = _task_k_slopes(rows)
        self.assertEqual(len(slopes), 2)
        self.assertAlmostEqual(min(slopes), -0.2)
        self.assertAlmostEqual(max(slopes), 0.0)

    def test_pair_coverage_keeps_short_and_long(self):
        runs = []
        for length in ("short", "long"):
            for variant in ("stop", "recovery"):
                runs.append(
                    {
                        "model": "m",
                        "error_type": "off_topic",
                        "k": 1,
                        "length_level": length,
                        "task_id": "t",
                        "seed": 1,
                        "variant": variant,
                        "_correct": 1.0,
                    }
                )
        rows = build_pair_coverage(runs)
        self.assertEqual(sorted(row["length_level"] for row in rows), ["long", "short"])
        self.assertTrue(all(row["n_pairs_design"] == 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
