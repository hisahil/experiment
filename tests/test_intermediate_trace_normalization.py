"""Unit tests for Punkt 1 fixes (empty intermediate segments).

Covers:
- Schicht 1: `_normalize_trace_end` idempotence and edge cases (root cause of
  the 4.6 % empty intermediate paragraphs observed in the mini run).
- TypedTurn serialisation carries the new `intermediate_retried` /
  `intermediate_empty` audit flags introduced for Schicht 2.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_dialogue import (  # noqa: E402
    TypedTurn,
    _normalize_trace_end,
)


class NormalizeTraceEndTests(unittest.TestCase):
    SEP = "\n\n"

    def test_appends_when_no_trailing_newline(self) -> None:
        trace = "First paragraph.\n\nInjection about cheese."
        self.assertEqual(
            _normalize_trace_end(trace, self.SEP),
            "First paragraph.\n\nInjection about cheese.\n\n",
        )

    def test_collapses_double_separator(self) -> None:
        trace = "First paragraph.\n\nInjection about cheese.\n\n\n\n"
        self.assertEqual(
            _normalize_trace_end(trace, self.SEP),
            "First paragraph.\n\nInjection about cheese.\n\n",
        )

    def test_upgrades_single_newline(self) -> None:
        trace = "First paragraph.\n\nInjection\n"
        # rstrip('\n') removes both cases → one \n\n is appended.
        self.assertEqual(
            _normalize_trace_end(trace, self.SEP),
            "First paragraph.\n\nInjection\n\n",
        )

    def test_idempotent(self) -> None:
        trace = "content\n\n"
        once = _normalize_trace_end(trace, self.SEP)
        twice = _normalize_trace_end(once, self.SEP)
        self.assertEqual(once, twice)

    def test_empty_trace_untouched(self) -> None:
        self.assertEqual(_normalize_trace_end("", self.SEP), "")

    def test_no_separator_untouched(self) -> None:
        self.assertEqual(_normalize_trace_end("content", ""), "content")


class TypedTurnAuditFlagsTests(unittest.TestCase):
    def test_flags_omitted_when_none(self) -> None:
        turn = TypedTurn(turn_index=0, role="assistant", source="model_response", text="ok")
        d = turn.to_dict()
        self.assertNotIn("intermediate_retried", d)
        self.assertNotIn("intermediate_empty", d)

    def test_flags_included_when_set(self) -> None:
        turn = TypedTurn(turn_index=0, role="assistant", source="model_response", text="")
        turn.intermediate_retried = True
        turn.intermediate_empty = True
        d = turn.to_dict()
        self.assertIs(d["intermediate_retried"], True)
        self.assertIs(d["intermediate_empty"], True)


if __name__ == "__main__":
    unittest.main()
