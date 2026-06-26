"""Tests that YAML design block stays aligned with Python constants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_design import (  # noqa: E402
    DEFAULT_DESIGN,
    DEFAULT_SEEDS,
    ERROR_TYPES,
    K_INJECTIONS,
    LENGTH_LEVELS,
    VARIANTS,
)


class ExperimentDesignSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg_path = ROOT / "configs/experiment.yaml"
        with cfg_path.open(encoding="utf-8") as f:
            cls.design = yaml.safe_load(f).get("design", {})

    def test_variants_match(self) -> None:
        self.assertEqual(self.design.get("variants"), VARIANTS)

    def test_error_types_match(self) -> None:
        self.assertEqual(self.design.get("error_types"), ERROR_TYPES)

    def test_k_injections_match(self) -> None:
        self.assertEqual(self.design.get("k_injections"), K_INJECTIONS)

    def test_length_levels_match(self) -> None:
        self.assertEqual(self.design.get("length_levels"), LENGTH_LEVELS)

    def test_seeds_match(self) -> None:
        self.assertEqual(self.design.get("seeds"), DEFAULT_SEEDS)

    def test_default_design_bundle(self) -> None:
        self.assertEqual(DEFAULT_DESIGN["variants"], VARIANTS)
        self.assertEqual(DEFAULT_DESIGN["error_types"], ERROR_TYPES)


if __name__ == "__main__":
    unittest.main()
