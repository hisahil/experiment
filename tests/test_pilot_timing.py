"""Tests for balanced task selection and complete dialogue records."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_dialogue import build_complete_dialogue_record, init_dialogue_state  # noqa: E402
from experiment_scope import count_planned_dialogues, select_tasks_balanced  # noqa: E402
from llm_backend import GenerationResult  # noqa: E402


def _tasks():
    out = []
    for domain in ("math", "science", "logic"):
        for i in range(10):
            out.append({"task_id": f"{domain}_{i:02d}", "domain": domain})
    return out


def test_select_tasks_balanced():
    scope = {"domains": ["math", "science", "logic"]}
    selected = select_tasks_balanced(_tasks(), scope, 42, 5)
    assert len(selected) == 15
    from collections import Counter
    assert Counter(t["domain"] for t in selected) == {"math": 5, "science": 5, "logic": 5}


def test_count_planned_dialogues_full_factorial():
    tasks = [{"task_id": "t1", "domain": "math"}] * 15
    lists = {
        "error_types": ["a", "b", "c"],
        "k_injections": [1, 3, 5],
        "length_levels": ["short", "long"],
        "seeds": [42, 43, 44, 45, 46],
    }
    assert count_planned_dialogues(tasks, lists, include_neutral=False) == 15 * 90


def test_build_complete_dialogue_record_includes_final_turns():
    state = init_dialogue_state(
        task={"task_id": "t1", "domain": "math", "prompt": "Q?", "gold_answer": "4"},
        baseline={"steps": ["s1", "s2"]},
        baseline_prefix="prefix",
        insertion_step=1,
        error_type="off_topic",
        k=1,
        length_level="short",
        seed=42,
        condition="adversarial",
        model_name="test-model",
        turn2_source="baseline_prefix",
        pregenerated_first="inj",
        agent_role_id="persistent_off_topic",
    )
    stop = GenerationResult(text="stop ans", completion_tokens=10, finish_reason="stop", prompt_tokens=100, wall_time_s=1.0)
    rec = GenerationResult(text="recovery ans", completion_tokens=200, finish_reason="stop", prompt_tokens=100, wall_time_s=2.0)
    record = build_complete_dialogue_record(state, stop, rec)
    variants = [t.get("variant") for t in record["turns"]]
    assert "stop" in variants
    assert "recovery" in variants
    assert record["turns"][-1]["text"] == "recovery ans"
