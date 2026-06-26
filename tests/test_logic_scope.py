"""Tests for hierarchical logic task selection in experiment_scope."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_scope import select_tasks  # noqa: E402


def _logic_task(task_id: str, source: str, tier: str) -> dict:
    return {
        "task_id": task_id,
        "domain": "logic",
        "logic_source": source,
        "logic_difficulty_tier": tier,
    }


def _make_logic_pool() -> list[dict]:
    pool: list[dict] = []
    for tier in ("low", "mid", "high"):
        for j in range(4):
            pool.append(_logic_task(f"kk_{tier}_{j}", "kk", tier))
    for tier in ("low", "mid", "high"):
        for j in range(4):
            pool.append(_logic_task(f"jl_{tier}_{j}", "justlogic", tier))
    return pool


def test_logic_cap_splits_sources_fifty_fifty():
    pool = _make_logic_pool()
    scope = {"domains": ["logic"], "max_per_domain": {"logic": 8}}
    selected = select_tasks(pool, scope, experiment_seed=42)
    sources = {t["logic_source"] for t in selected}
    assert sources == {"kk", "justlogic"}
    assert sum(1 for t in selected if t["logic_source"] == "kk") == 4
    assert sum(1 for t in selected if t["logic_source"] == "justlogic") == 4


def test_logic_cap_preserves_tier_spread():
    pool = _make_logic_pool()
    scope = {"domains": ["logic"], "max_per_domain": {"logic": 12}}
    selected = select_tasks(pool, scope, experiment_seed=42)
    for source in ("kk", "justlogic"):
        subset = [t for t in selected if t["logic_source"] == source]
        tiers = {t["logic_difficulty_tier"] for t in subset}
        assert tiers == {"low", "mid", "high"}
        assert len(subset) == 6
