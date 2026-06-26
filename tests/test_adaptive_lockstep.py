"""Lockstep batching and localizer label tests for adaptive dialogues."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_dialogue import (  # noqa: E402
    TypedTurn,
    extract_localizer_labels,
    init_dialogue_state,
    run_dialogue_until_complete,
)
from llm_backend import DryRunBackend  # noqa: E402


def _task(task_id: str = "t1") -> dict:
    return {
        "task_id": task_id,
        "domain": "math",
        "eval_type": "exact",
        "prompt": "What is 2+2?",
        "gold_answer": "4",
        "decoy_answer": "5",
    }


def _baseline() -> dict:
    return {"steps": ["s1", "s2", "s3", "s4"], "gold_answer": "4"}


def _make_state(k: int, task_id: str):
    return init_dialogue_state(
        task=_task(task_id),
        baseline=_baseline(),
        baseline_prefix="step one",
        insertion_step=2,
        error_type="off_topic",
        k=k,
        length_level="short",
        seed=42,
        condition="adversarial",
        model_name="test-model",
        turn2_source="baseline_prefix",
        pregenerated_first=f"pre-inj-{task_id}-k{k}",
        agent_role_id="persistent_off_topic",
    )


def test_lockstep_mixed_k1_and_k3_terminates():
    backend = DryRunBackend()
    backend.gold_hint = "4"
    states = [_make_state(1, "k1_task"), _make_state(3, "k3_task")]

    call_log: list[tuple[str, int]] = []

    def get_injection(state):
        call_log.append((state.task["task_id"], state.current_round()))
        if state.current_round() == 1 and state.pregenerated_first:
            return state.pregenerated_first
        return f"adaptive-{state.task['task_id']}-r{state.current_round()}"

    records, completed, failed, stopped_reason = run_dialogue_until_complete(
        states,
        backend,
        get_injection,
        adaptive_from=2,
        intermediate_max_tokens=256,
        stop_max_tokens=50,
        recovery_max_tokens=512,
        temperature=0.6,
        top_p=0.95,
        select_stop_prefill=lambda d, e: "\n\nTherefore, the final answer is \\boxed{",
    )

    assert not failed
    assert stopped_reason == "complete"
    assert len(completed) == 2
    k1 = next(s for s in completed if s.k == 1)
    k3 = next(s for s in completed if s.k == 3)
    assert len(k1.prior_injections) == 1
    assert len(k3.prior_injections) == 3
    assert len(k1.model_turns) == 0
    assert len(k3.model_turns) == 2
    assert len(records) == 4

    k1_calls = [r for tid, r in call_log if tid == "k1_task"]
    k3_calls = [r for tid, r in call_log if tid == "k3_task"]
    assert k1_calls == [1]
    assert k3_calls == [1, 2, 3]


def test_extract_localizer_labels_from_transcript():
    turns = [
        TypedTurn(0, "user", "task", "Q"),
        TypedTurn(1, "assistant", "baseline_prefix", "prefix"),
        TypedTurn(
            2, "user", "injection", "Inj step one.\nInj step two.",
            injection_step_index=1, error_type="off_topic", adaptive=False,
        ),
        TypedTurn(
            4, "user", "injection", "Adaptive inj.",
            injection_step_index=2, error_type="off_topic", adaptive=True,
        ),
    ]
    labels = extract_localizer_labels(turns)
    assert labels["injection_steps"] == ["Inj step one.\nInj step two.", "Adaptive inj."]
    assert len(labels["injected_turn_spans"]) == 2
    assert labels["injected_turn_spans"][0]["adaptive"] is False
    assert labels["injected_turn_spans"][1]["adaptive"] is True
    assert len(labels["injected_step_offsets"]) >= 2
