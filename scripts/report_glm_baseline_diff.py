#!/usr/bin/env python3
"""Before/After report for the GLM baseline reprocess.

Compares the ORIGINAL 5x baseline file (steps segmented from the full
raw_output — contaminated with `</think>` mid-list, "wait/hmm" prose,
draft+fair-copy duplication) against the REPROCESSED file (baseline_strip_think
applied via prepare_baseline_raw_text) and prints/writes a Markdown report
covering the GATE 1 criteria from the plan:

    - verified count (must stay 100/100)
    - `</think>` occurrences in steps[]
    - post-think prose markers ("wait", "hmm", "let me check", "actually")
    - answer-in-prefix rate (first half of steps contains gold_answer)
    - min_steps distribution (must stay >= 2 for a usable midpoint)
    - pick-baseline drift (which task's chosen sample changed)

Usage (defaults match the Phase 4 command in the plan):
    python scripts/report_glm_baseline_diff.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROSE_MARKERS = ("wait", "hmm", "let me check", "let me verify", "actually")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def contains_prose(steps: list[str]) -> bool:
    return any(any(m in s.lower() for m in PROSE_MARKERS) for s in steps)


def contains_close_tag(steps: list[str]) -> bool:
    return any("</think>" in s for s in steps)


def answer_in_prefix(steps: list[str], gold: str | None) -> bool:
    """Check whether gold_answer appears in the first half of steps."""
    if not gold or not steps:
        return False
    half = max(1, len(steps) // 2)
    prefix = " ".join(steps[:half]).lower()
    return str(gold).lower() in prefix


def summarize(rows: list[dict], label: str) -> dict:
    verified = sum(1 for r in rows if r.get("verified"))
    has_close = sum(1 for r in rows if contains_close_tag(r.get("steps", [])))
    has_prose = sum(1 for r in rows if contains_prose(r.get("steps", [])))
    answer_pre = sum(1 for r in rows if answer_in_prefix(r.get("steps", []), r.get("gold_answer")))
    step_dist = Counter(r.get("step_count", 0) for r in rows)
    lt2 = sum(1 for r in rows if r.get("step_count", 0) < 2)
    lt4 = sum(1 for r in rows if r.get("step_count", 0) < 4)
    return {
        "label": label,
        "rows": len(rows),
        "verified": verified,
        "has_close_tag": has_close,
        "has_prose": has_prose,
        "answer_in_prefix": answer_pre,
        "answer_in_prefix_pct": 100.0 * answer_pre / max(1, len(rows)),
        "step_count_dist": step_dist,
        "min_step_count": min((r.get("step_count", 0) for r in rows), default=0),
        "lt2": lt2,
        "lt4": lt4,
    }


def picker_drift(old: dict[str, dict], new: dict[str, dict]) -> list[dict]:
    """Return tasks where the reprocessed picker chose a different seed."""
    drift: list[dict] = []
    for task_id in sorted(set(old) & set(new)):
        so = old[task_id].get("seed")
        sn = new[task_id].get("seed")
        if so != sn:
            drift.append({"task_id": task_id, "old_seed": so, "new_seed": sn})
    return drift


def gate1_verdict(new_sum: dict, drift: list[dict]) -> tuple[bool, list[str]]:
    """Apply plan's GATE-1 Go/No-Go criteria (Phase 4)."""
    reasons: list[str] = []
    ok = True
    if new_sum["verified"] != 100:
        ok = False
        reasons.append(f"verified != 100 (got {new_sum['verified']})")
    if new_sum["has_close_tag"] != 0:
        ok = False
        reasons.append(f"</think> in steps[] on {new_sum['has_close_tag']} tasks")
    if new_sum["lt2"] > 3:
        ok = False
        reasons.append(f"{new_sum['lt2']} tasks below min_steps=2 (>3 allowed)")
    if new_sum["answer_in_prefix_pct"] > 20.0:
        ok = False
        reasons.append(
            f"answer_in_prefix={new_sum['answer_in_prefix_pct']:.1f}% > 20%"
        )
    return ok, reasons


def render_md(old_sum: dict, new_sum: dict, drift: list[dict], go: bool, reasons: list[str]) -> str:
    lines: list[str] = []
    lines.append("# GLM Baseline Reprocess Report (Phase 4, GATE 1)")
    lines.append("")
    lines.append("| Metric | ALT (pre-strip) | NEU (baseline_strip_think) |")
    lines.append("|---|---|---|")
    lines.append(f"| rows | {old_sum['rows']} | {new_sum['rows']} |")
    lines.append(f"| verified | {old_sum['verified']} | {new_sum['verified']} |")
    lines.append(
        f"| `</think>` in steps[] | {old_sum['has_close_tag']} | {new_sum['has_close_tag']} |"
    )
    lines.append(f"| post-think prose markers | {old_sum['has_prose']} | {new_sum['has_prose']} |")
    lines.append(
        f"| answer-in-prefix | {old_sum['answer_in_prefix']} "
        f"({old_sum['answer_in_prefix_pct']:.1f}%) | "
        f"{new_sum['answer_in_prefix']} "
        f"({new_sum['answer_in_prefix_pct']:.1f}%) |"
    )
    lines.append(
        f"| min step_count | {old_sum['min_step_count']} | {new_sum['min_step_count']} |"
    )
    lines.append(f"| tasks with steps < 2 | {old_sum['lt2']} | {new_sum['lt2']} |")
    lines.append(f"| tasks with steps < 4 | {old_sum['lt4']} | {new_sum['lt4']} |")
    lines.append("")
    lines.append(
        f"**step_count distribution NEU:** "
        + ", ".join(f"{k}:{v}" for k, v in sorted(new_sum['step_count_dist'].items()))
    )
    lines.append("")
    lines.append(f"**Baseline-picker drift:** {len(drift)} tasks chose a different seed.")
    if drift:
        lines.append("")
        for d in drift[:20]:
            lines.append(f"- `{d['task_id']}`: seed {d['old_seed']} → {d['new_seed']}")
        if len(drift) > 20:
            lines.append(f"- … +{len(drift) - 20} more")
    lines.append("")
    lines.append("## GATE 1 — Baseline Go/No-Go")
    lines.append("")
    if go:
        lines.append("**Verdict: GO** — all criteria met.")
    else:
        lines.append("**Verdict: NO-GO** — see reasons:")
        for r in reasons:
            lines.append(f"- {r}")
    lines.append("")
    lines.append("Criteria (from plan Phase 4):")
    lines.append("- verified == 100/100")
    lines.append("- 0 tasks with `</think>` in steps[]")
    lines.append("- <=3 tasks with min_steps < 2")
    lines.append("- answer-in-prefix <= 20% (DeepSeek range: 12-18%)")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old",
        type=Path,
        default=ROOT
        / "backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x.jsonl",
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=ROOT
        / "backup_gpu_live/glm_baseline_5x/data/baselines/glm-z1-9b/baseline_cots_5x.reprocessed.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "backup_gpu_live/glm_baseline_5x/reprocess_report.md",
    )
    args = parser.parse_args()

    old = load_jsonl(args.old)
    new = load_jsonl(args.new)
    old_by_task = {r["task_id"]: r for r in old}
    new_by_task = {r["task_id"]: r for r in new}

    old_sum = summarize(old, "ALT")
    new_sum = summarize(new, "NEU")
    drift = picker_drift(old_by_task, new_by_task)
    go, reasons = gate1_verdict(new_sum, drift)

    md = render_md(old_sum, new_sum, drift, go, reasons)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
