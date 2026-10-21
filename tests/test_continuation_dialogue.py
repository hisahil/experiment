"""Deletion-model continuation runner (Phase 1)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_dialogue import (  # noqa: E402
    compute_continuation_prefix,
    init_dialogue_state,
    run_continuation_dialogues,
)
from llm_backend import DryRunBackend  # noqa: E402

MATH_PREFILL = "</think>\n\nTherefore, the final answer is \\boxed{"


def _stop_prefill(domain, eval_type):
    return MATH_PREFILL


def _make_state(k: int, continuation_steps: list[str], *, seed: int = 42):
    midpoint, prefix = compute_continuation_prefix(continuation_steps)
    task = {
        "task_id": f"t_k{k}",
        "prompt": "Compute the value of x.",
        "domain": "math",
        "eval_type": "symbolic",
        "gold_answer": "7",
        "decoy_answer": "5",
    }
    baseline = {"steps": continuation_steps, "step_count": len(continuation_steps)}
    state = init_dialogue_state(
        task=task,
        baseline=baseline,
        baseline_prefix=prefix,
        insertion_step=midpoint,
        error_type="off_topic",
        k=k,
        length_level="short",
        seed=seed,
        condition="adversarial",
        model_name="test-model",
        turn2_source="baseline_prefix",
        pregenerated_first=None,
        agent_role_id="persistent_off_topic",
    )
    state.generation_mode = "continuation"
    state.accumulated_trace = prefix
    state.add_trace_segment("baseline_prefix", prefix)
    state.injection_steps = [f"INJECTION_{i + 1}" for i in range(k)]
    return state


def _run(states):
    backend = DryRunBackend()
    return run_continuation_dialogues(
        states,
        backend,
        intermediate_max_tokens=128,
        stop_max_tokens=64,
        recovery_max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        select_stop_prefill=_stop_prefill,
        trace_separator="\n\n",
        intermediate_stop="\n\n",
    )


class ContinuationDeletionModelTests(unittest.TestCase):
    STEPS = [f"reasoning paragraph {i}" for i in range(1, 11)]  # n=10 -> midpoint 5

    def test_k1_no_intermediate(self) -> None:
        st = _make_state(1, self.STEPS)
        records, completed, failed, reason = _run([st])
        self.assertEqual(reason, "complete")
        self.assertEqual(len(failed), 0)
        self.assertEqual(len(completed), 1)
        # trace = prefix + inj1, no model paragraph
        self.assertIn("INJECTION_1", st.accumulated_trace)
        seg_types = [s["type"] for s in st.trace_segments]
        self.assertEqual(seg_types, ["baseline_prefix", "injection", "stop", "recovery"])
        self.assertEqual(len(records), 2)
        self.assertEqual({r["variant"] for r in records}, {"stop", "recovery"})

    def test_k2_one_intermediate_paragraph(self) -> None:
        st = _make_state(2, self.STEPS)
        _run([st])
        seg_types = [s["type"] for s in st.trace_segments]
        self.assertEqual(
            seg_types,
            ["baseline_prefix", "injection", "model", "injection", "stop", "recovery"],
        )
        # Both injections present, in order, with a model paragraph between them.
        i1 = st.accumulated_trace.index("INJECTION_1")
        i2 = st.accumulated_trace.index("INJECTION_2")
        self.assertLess(i1, i2)

    def test_k3_two_intermediate_paragraphs(self) -> None:
        st = _make_state(3, self.STEPS)
        _run([st])
        seg_types = [s["type"] for s in st.trace_segments]
        self.assertEqual(seg_types.count("model"), 2)
        self.assertEqual(seg_types.count("injection"), 3)
        for tag in ("INJECTION_1", "INJECTION_2", "INJECTION_3"):
            self.assertIn(tag, st.accumulated_trace)

    def test_prefix_is_midpoint_join(self) -> None:
        midpoint, prefix = compute_continuation_prefix(self.STEPS)
        self.assertEqual(midpoint, 5)
        self.assertEqual(prefix, "\n\n".join(self.STEPS[:5]))

    def test_short_baseline_supports_all_k(self) -> None:
        # 4-step baseline: deletion model only needs injection 1 in the baseline.
        short_steps = ["a", "b", "c", "d"]
        for k in (1, 2, 3):
            st = _make_state(k, short_steps)
            _, completed, failed, _ = _run([st])
            self.assertEqual(len(failed), 0, f"k={k} failed on 4-step baseline")
            self.assertEqual(len(completed), 1)

    def test_missing_injection_fails_cleanly(self) -> None:
        st = _make_state(3, self.STEPS)
        st.injection_steps = ["only_one"]  # k=3 but one step
        _, completed, failed, _ = _run([st])
        self.assertEqual(len(completed), 0)
        self.assertEqual(len(failed), 1)
        self.assertIn("missing", st.fail_reason)

    def test_transcript_includes_continuation_metadata(self) -> None:
        st = _make_state(2, self.STEPS)
        _run([st])
        rec = st.transcript_record()
        self.assertEqual(rec["generation_mode"], "continuation")
        self.assertIn("accumulated_trace_len", rec)
        self.assertEqual(rec["prompt_render_method"], "continuation_trace")
        self.assertTrue(len(rec["trace_segments"]) >= 5)

    def test_mixed_k_batch(self) -> None:
        states = [_make_state(1, self.STEPS), _make_state(2, self.STEPS), _make_state(3, self.STEPS)]
        records, completed, failed, _ = _run(states)
        self.assertEqual(len(completed), 3)
        self.assertEqual(len(failed), 0)
        self.assertEqual(len(records), 6)  # 3 dialogues x (stop + recovery)


if __name__ == "__main__":
    unittest.main()
