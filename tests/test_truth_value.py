"""Tests for JustLogic truth_value grading and extraction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluator import (  # noqa: E402
    _extract_truth_value,
    extract_domain_answer,
    is_correct,
)
from load_justlogic import (  # noqa: E402
    depth_to_tier,
    justlogic_decoy,
    normalize_label,
)


def test_normalize_label():
    assert normalize_label("TRUE") == "True"
    assert normalize_label("FALSE") == "False"
    assert normalize_label("Uncertain") == "Uncertain"


def test_depth_to_tier_default():
    tiers = {"low": [1, 2], "mid": [3, 4], "high": [5, 6, 7]}
    assert depth_to_tier(1, tiers) == "low"
    assert depth_to_tier(4, tiers) == "mid"
    assert depth_to_tier(7, tiers) == "high"
    assert depth_to_tier(99, tiers) is None


def test_justlogic_decoy_not_gold():
    import random

    rng = random.Random(42)
    for gold in ("True", "False", "Uncertain"):
        decoy = justlogic_decoy(gold, rng)
        assert decoy != gold
        assert decoy in {"True", "False", "Uncertain"}


def test_extract_truth_value_final_answer():
    text = "Some reasoning here.\n\nFinal answer: False"
    assert _extract_truth_value(text) == "False"


def test_extract_truth_value_synonyms():
    assert _extract_truth_value("Final answer: entailed") == "True"
    assert _extract_truth_value("Final answer: contradicted") == "False"
    assert _extract_truth_value("Final answer: cannot be determined") == "Uncertain"


def test_extract_truth_value_avoids_untrue_trap():
    assert _extract_truth_value("Final answer: untrue") != "True"


def test_extract_domain_answer_truth_value():
    raw = "Therefore the statement is entailed.\nFinal answer: True"
    assert extract_domain_answer(raw, "logic", "truth_value") == "True"


def test_is_correct_truth_value():
    assert is_correct("True", "True", "truth_value") is True
    assert is_correct("False", "True", "truth_value") is False
    assert is_correct("Uncertain", "Uncertain", "truth_value") is True
    assert is_correct(None, "True", "truth_value") is None
