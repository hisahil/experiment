"""Tests for adaptive dialogue message construction."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_backend import (  # noqa: E402
    COLLABORATING_AGENT_TAG,
    build_dialogue_messages,
    build_perturbed_messages,
    format_injection_text,
)


def test_format_injection_text_none():
    assert format_injection_text("Step A", injection_tag="none") == "Step A"


def test_format_injection_text_collaborating_agent():
    out = format_injection_text("Step A", injection_tag="collaborating_agent")
    assert out.startswith(COLLABORATING_AGENT_TAG)
    assert "Step A" in out


def test_build_dialogue_messages_k3():
    msgs = build_dialogue_messages(
        "Solve 2+2",
        "First I add the numbers.",
        [
            ("Injection 1 text", "Model reply 1"),
            ("Injection 2 text", "Model reply 2"),
            ("Injection 3 text", None),
        ],
        injection_tag="none",
    )
    assert len(msgs) == 7  # user, asst, (user, asst) x2, final user
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "Solve 2+2"
    assert msgs[1]["content"] == "First I add the numbers."
    assert msgs[2]["content"] == "Injection 1 text"
    assert msgs[3]["content"] == "Model reply 1"
    assert msgs[6]["content"] == "Injection 3 text"
    assert msgs[6]["role"] == "user"
    assert all(COLLABORATING_AGENT_TAG not in m["content"] for m in msgs)


def test_build_dialogue_messages_k1():
    msgs = build_dialogue_messages(
        "Q",
        "prefix",
        [("only injection", None)],
    )
    assert len(msgs) == 3
    assert msgs[-1]["content"] == "only injection"


def test_build_dialogue_messages_strips_think_tags():
    ot = "<" + "think" + ">"
    ct = "</" + "think" + ">"
    prefix = f"before{ot}hidden{ct}after"
    reply = f"part1{ot}mid{ct}part2"
    msgs = build_dialogue_messages(
        "Q",
        prefix,
        [("inj", reply), ("final inj", None)],
        injection_tag="none",
    )
    assert msgs[1]["content"] == "beforehiddenafter"
    assert msgs[3]["content"] == "part1midpart2"
    assert ot not in msgs[1]["content"]


def test_build_perturbed_messages_legacy_tag():
    msgs = build_perturbed_messages("Q", "prefix", ["step1", "step2"])
    assert len(msgs) == 3
    assert COLLABORATING_AGENT_TAG in msgs[2]["content"]


def test_build_perturbed_messages_no_tag():
    msgs = build_perturbed_messages(
        "Q", "prefix", ["step1"], injection_tag="none"
    )
    assert COLLABORATING_AGENT_TAG not in msgs[2]["content"]
    assert "step1" in msgs[2]["content"]
