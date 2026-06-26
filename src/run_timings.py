"""Timing helpers for GPU pilot runs and full-thesis extrapolation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Full-thesis dialogue count = tasks × error_types × k × length × seeds.
# Authoritative full-run scope (docs/full_run_plan.md §1.2): 150 × 3 × 2 × 2 × 3 = 5_400.
# (Was 240 × 3 × 3 × 2 × 5 = 21_600, the abandoned legacy multi-domain "full" scope.)
FULL_THESIS_DIALOGUES = 150 * 3 * 2 * 2 * 3  # full_math_150: 5_400
HOURLY_RATE_USD = 1.36


def fmt_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def build_extrapolation(
    *,
    completed_dialogues: int,
    compute_elapsed_s: float,
    planned_tonight: int,
    scope: str | None = None,
) -> dict[str, Any]:
    if completed_dialogues <= 0 or compute_elapsed_s <= 0:
        return {
            "seconds_per_dialogue": None,
            "planned_tonight": planned_tonight,
            "planned_full_thesis": FULL_THESIS_DIALOGUES,
            "extrapolated_full_thesis_s": None,
            "extrapolated_full_thesis_h": None,
            "extrapolated_full_thesis_days": None,
            "extrapolated_full_cost_usd": None,
        }
    spd = compute_elapsed_s / completed_dialogues
    full_s = spd * FULL_THESIS_DIALOGUES
    full_h = full_s / 3600.0
    return {
        "seconds_per_dialogue": round(spd, 2),
        "planned_tonight": planned_tonight,
        "planned_full_thesis": FULL_THESIS_DIALOGUES,
        "extrapolated_full_thesis_s": round(full_s, 1),
        "extrapolated_full_thesis_h": round(full_h, 2),
        "extrapolated_full_thesis_days": round(full_s / 86400.0, 2),
        "extrapolated_full_cost_usd": round(full_h * HOURLY_RATE_USD, 2),
        "scope": scope,
    }


def write_run_timings(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
