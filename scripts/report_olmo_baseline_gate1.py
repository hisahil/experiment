#!/usr/bin/env python3
"""OLMo baseline Gate 1 audit report (before/after repair).

Modeled on scripts/report_glm_baseline_diff.py with the fuller Gate 1 criteria
from the OLMo baseline repair plan:

  - 100 unique tasks, >=5 samples each
  - 100/100 majority-verified
  - zero think-tag / placeholder / empty-step leaks in selected traces
  - zero hidden-think/draft provenance in selected CoTs
  - Step 1 start, >=3 steps, nonempty midpoint prefixes
  - zero silent judge fallbacks on samples; report det/judge disagreements
  - before/after seed drift vs pre-repair backup

Usage:
    python scripts/report_olmo_baseline_gate1.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

THINK_MARKERS = ("<think>", "</think>", "</think#")
PROMPT_FORMAT_MARKERS = (
    "the problem asks me to write",
    "format your response",
    "step-by-step solution",
    "put the answer after",
)
PLACEHOLDER_RE = re.compile(r"^\s*(?:\.{3}|…|TODO|TBD|N/?A)\s*$", re.I)

# Previously flagged selections (plan §2–3 + Gate 1 focused recheck).

SEMANTIC_REVIEW_PATH = ROOT / "data/baselines/olmo-3-7b/semantic_review.jsonl"


def load_semantic_review() -> dict[str, dict]:
    if not SEMANTIC_REVIEW_PATH.exists():
        return {}
    return {r["task_id"]: r for r in load_jsonl(SEMANTIC_REVIEW_PATH)}


def semantic_verdict(reviews: dict[str, dict], baselines: list[dict]) -> tuple[str, list[str]]:
    missing = [r["task_id"] for r in baselines if r["task_id"] not in reviews]
    if missing:
        return "NO-GO", [f"missing semantic review: {len(missing)} tasks"]
    bad = [r for r in reviews.values() if r.get("status") != "clean"]
    if bad:
        return "NO-GO", [f"non-clean semantic status: {len(bad)} tasks"]
    return "GO", []

FOCUSED_STEMS = {
    # Auto-resolved by strip / picker
    "algebra_251",
    "prealgebra_1784",
    "counting_and_probability_913",
    "number_theory_1287",
    "number_theory_1055",
    "counting_and_probability_51",
    "algebra_858",
    # Audited seed overrides
    "algebra_114",
    "prealgebra_307",
    "intermediate_algebra_894",
    "counting_and_probability_282",
    "intermediate_algebra_966",
    "intermediate_algebra_158",
    "geometry_967",
    "geometry_483",
    "intermediate_algebra_690",
    "number_theory_820",
    "algebra_1275",
    "prealgebra_1787",
    "number_theory_1002",
    # Targeted regen
    "precalculus_1252",
    "algebra_2277",
    "prealgebra_846",
    "algebra_661",
    "intermediate_algebra_1111",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def task_stem(task_id: str) -> str:
    name = task_id
    if name.startswith("math500_test_"):
        name = name[len("math500_test_") :]
    if name.endswith(".json"):
        name = name[: -len(".json")]
    return name


def selected_text(row: dict) -> str:
    """Prefer baseline_cot; fall back to joined steps (avoid double-counting)."""
    cot = (row.get("baseline_cot") or "").strip()
    if cot:
        return cot
    return "\n".join(row.get("steps") or [])


def has_think_leak(row: dict) -> bool:
    text = selected_text(row)
    steps_text = "\n".join(row.get("steps") or [])
    return any(m in text or m in steps_text for m in THINK_MARKERS)


def has_prompt_format_discussion(row: dict) -> bool:
    text = selected_text(row).lower()
    return any(m in text for m in PROMPT_FORMAT_MARKERS)


def has_placeholder_or_empty_step(row: dict) -> bool:
    steps = row.get("steps") or []
    if not steps:
        return True
    for s in steps:
        if not str(s).strip() or PLACEHOLDER_RE.match(str(s)):
            return True
    return False


def has_hidden_think_draft(row: dict) -> bool:
    """Detect draft+fair-copy duplication / think residue in selected CoT."""
    text = selected_text(row)
    steps = row.get("steps") or []
    steps_text = "\n".join(steps)
    if any(m in text or m in steps_text for m in THINK_MARKERS):
        return True
    # Duplicated "Step 1:" inside the fair-copy CoT (not steps[] + cot concat).
    return len(re.findall(r"(?im)^\s*step\s+1\s*:", text)) >= 2


def starts_at_step1(row: dict) -> bool:
    steps = row.get("steps") or []
    if not steps:
        return False
    first = str(steps[0]).lstrip()
    return bool(re.match(r"(?i)^step\s*1\b", first)) or bool(
        re.search(r"(?im)^\s*step\s*1\s*:", row.get("baseline_cot") or "")
    )


def midpoint_ok(row: dict) -> bool:
    steps = row.get("steps") or []
    if len(steps) < 2:
        return False
    mid = len(steps) // 2
    prefix = "\n".join(steps[:mid]).strip()
    return bool(prefix)


def answer_in_prefix(row: dict) -> bool:
    steps = row.get("steps") or []
    gold = row.get("gold_answer")
    if not gold or not steps:
        return False
    half = max(1, len(steps) // 2)
    prefix = " ".join(steps[:half]).lower()
    return str(gold).lower() in prefix


def picker_drift(old: dict[str, dict], new: dict[str, dict]) -> list[dict]:
    drift: list[dict] = []
    for task_id in sorted(set(old) & set(new)):
        so, sn = old[task_id].get("seed"), new[task_id].get("seed")
        if so != sn:
            drift.append(
                {
                    "task_id": task_id,
                    "old_seed": so,
                    "new_seed": sn,
                    "selection_tier": new[task_id].get("selection_tier"),
                }
            )
    return drift


def audit(new_rows: list[dict], samples: list[dict], old_rows: list[dict]) -> dict:
    by_task = {r["task_id"]: r for r in new_rows}
    sample_counts = Counter(s["task_id"] for s in samples)
    verified = sum(1 for r in new_rows if r.get("verified"))

    think_leaks = [r["task_id"] for r in new_rows if has_think_leak(r)]
    prompt_fmt = [r["task_id"] for r in new_rows if has_prompt_format_discussion(r)]
    placeholders = [r["task_id"] for r in new_rows if has_placeholder_or_empty_step(r)]
    hidden = [r["task_id"] for r in new_rows if has_hidden_think_draft(r)]
    not_step1 = [r["task_id"] for r in new_rows if not starts_at_step1(r)]
    short = [r["task_id"] for r in new_rows if int(r.get("step_count") or 0) < 3]
    bad_mid = [r["task_id"] for r in new_rows if not midpoint_ok(r)]
    answer_pre = sum(1 for r in new_rows if answer_in_prefix(r))

    fb = sum(1 for s in samples if s.get("judge_fallback"))
    conflicts = [
        s
        for s in samples
        if not s.get("adjudication")
        and (
            s.get("judge_det_conflict")
            or (
                not s.get("judge_fallback")
                and s.get("correct") is True
                and s.get("det_correct") is False
            )
        )
    ]
    adjudicated = [s for s in samples if s.get("adjudication")]
    focused = {}
    for stem in sorted(FOCUSED_STEMS):
        matches = [r for r in new_rows if task_stem(r["task_id"]) == stem]
        if not matches:
            focused[stem] = {"present": False}
            continue
        r = matches[0]
        focused[stem] = {
            "present": True,
            "task_id": r["task_id"],
            "seed": r.get("seed"),
            "selection_tier": r.get("selection_tier"),
            "step_count": r.get("step_count"),
            "verified": r.get("verified"),
            "think_leak": has_think_leak(r),
            "hidden_draft": has_hidden_think_draft(r),
            "starts_step1": starts_at_step1(r),
            "midpoint_ok": midpoint_ok(r),
        }

    old_by = {r["task_id"]: r for r in old_rows}
    drift = picker_drift(old_by, by_task)
    under_sampled = [tid for tid, n in sample_counts.items() if n < 5]
    missing_tasks = [tid for tid in by_task if tid not in sample_counts]

    reasons: list[str] = []
    ok = True
    if len(new_rows) != 100:
        ok = False
        reasons.append(f"rows != 100 (got {len(new_rows)})")
    if len(by_task) != 100:
        ok = False
        reasons.append(f"unique tasks != 100 (got {len(by_task)})")
    if verified != 100:
        ok = False
        reasons.append(f"verified != 100 (got {verified})")
    if under_sampled or missing_tasks:
        ok = False
        reasons.append(
            f"sample coverage incomplete (under={len(under_sampled)} missing={len(missing_tasks)})"
        )
    if think_leaks:
        ok = False
        reasons.append(f"think-tag leaks in {len(think_leaks)} selected traces")
    if prompt_fmt:
        ok = False
        reasons.append(f"prompt-format discussion in {len(prompt_fmt)} selected traces")
    if placeholders:
        ok = False
        reasons.append(f"placeholder/empty steps in {len(placeholders)} selected traces")
    if hidden:
        ok = False
        reasons.append(f"hidden-think/draft provenance in {len(hidden)} selected traces")
    if not_step1:
        ok = False
        reasons.append(f"not starting at Step 1: {len(not_step1)}")
    if short:
        ok = False
        reasons.append(f"step_count < 3: {len(short)}")
    if bad_mid:
        ok = False
        reasons.append(f"empty/invalid midpoint prefixes: {len(bad_mid)}")
    if fb:
        ok = False
        reasons.append(f"silent judge fallbacks on samples: {fb}")
    # conflicts require adjudication note but do not auto NO-GO if judged correct
    # was explicitly adjudicated; still report them.
    focused_fail = [
        stem
        for stem, info in focused.items()
        if info.get("present")
        and (
            info.get("think_leak")
            or info.get("hidden_draft")
            or not info.get("verified")
            or not info.get("starts_step1")
            or not info.get("midpoint_ok")
        )
    ]
    if focused_fail:
        ok = False
        reasons.append(f"focused recheck failures: {', '.join(focused_fail)}")

    return {
        "go": ok,
        "reasons": reasons,
        "n_rows": len(new_rows),
        "n_unique": len(by_task),
        "baseline_task_ids": sorted(by_task),
        "verified": verified,
        "n_samples": len(samples),
        "sample_tasks": len(sample_counts),
        "min_samples": min(sample_counts.values()) if sample_counts else 0,
        "think_leaks": think_leaks,
        "prompt_fmt": prompt_fmt,
        "placeholders": placeholders,
        "hidden": hidden,
        "not_step1": not_step1,
        "short": short,
        "bad_mid": bad_mid,
        "answer_in_prefix": answer_pre,
        "answer_in_prefix_pct": 100.0 * answer_pre / max(1, len(new_rows)),
        "step_dist": Counter(int(r.get("step_count") or 0) for r in new_rows),
        "judge_fallbacks": fb,
        "conflicts": len(conflicts),
        "conflict_task_ids": sorted({c["task_id"] for c in conflicts})[:50],
        "adjudicated": len(adjudicated),
        "adjudication_notes": sorted(
            {
                f"{s['task_id']} seed={s.get('seed')}: {s.get('adjudication')}"
                for s in adjudicated
            }
        ),
        "drift": drift,
        "focused": focused,
        "selection_tiers": Counter(r.get("selection_tier") for r in new_rows),
    }


def render_md(a: dict) -> str:
    lines: list[str] = []
    lines.append("# OLMo Baseline Repair — Gate 1 Report")
    lines.append("")
    sem_reviews = load_semantic_review()
    baselines = [{"task_id": tid} for tid in a.get("baseline_task_ids", [])]
    sem_v, sem_notes = semantic_verdict(sem_reviews, baselines)
    lines.append(f"## SEMANTIC: {sem_v}")
    for n in sem_notes:
        lines.append(f"- {n}")
    if sem_v == "GO":
        lines.append(f"- reviewed_tasks: {len(sem_reviews)}")
    lines.append("")
    lines.append(f"**Verdict: {'GO' if a['go'] else 'NO-GO'}**")
    lines.append("")
    if a["reasons"]:
        lines.append("Reasons:")
        for r in a["reasons"]:
            lines.append(f"- {r}")
        lines.append("")
    lines.append("## Corpus")
    lines.append("")
    lines.append(f"- baselines: {a['n_rows']} rows / {a['n_unique']} unique tasks")
    lines.append(f"- verified: {a['verified']}/100")
    lines.append(
        f"- samples: {a['n_samples']} across {a['sample_tasks']} tasks "
        f"(min/task={a['min_samples']})"
    )
    lines.append(
        f"- selection tiers: "
        + ", ".join(f"{k}:{v}" for k, v in sorted(a["selection_tiers"].items(), key=lambda x: (-x[1], str(x[0]))) )
    )
    lines.append("")
    lines.append("## Structural / provenance")
    lines.append("")
    lines.append(f"- think-tag leaks: {len(a['think_leaks'])}")
    lines.append(f"- prompt-format discussion: {len(a['prompt_fmt'])}")
    lines.append(f"- placeholder/empty steps: {len(a['placeholders'])}")
    lines.append(f"- hidden-think/draft provenance: {len(a['hidden'])}")
    lines.append(f"- not Step 1: {len(a['not_step1'])}")
    lines.append(f"- step_count < 3: {len(a['short'])}")
    lines.append(f"- bad midpoint prefixes: {len(a['bad_mid'])}")
    lines.append(
        f"- answer-in-prefix: {a['answer_in_prefix']} ({a['answer_in_prefix_pct']:.1f}%) "
        "(diagnostic only)"
    )
    lines.append(
        "- step_count dist: "
        + ", ".join(f"{k}:{v}" for k, v in sorted(a["step_dist"].items()))
    )
    lines.append("")
    lines.append("## Judge coverage")
    lines.append("")
    lines.append(f"- judge_fallback samples: {a['judge_fallbacks']}")
    lines.append(f"- unresolved judge/det conflicts: {a['conflicts']}")
    lines.append(f"- adjudicated disagreements: {a['adjudicated']}")
    for note in a.get("adjudication_notes") or []:
        lines.append(f"  - {note}")
    if a["conflict_task_ids"]:
        lines.append("- conflict tasks (sample): " + ", ".join(f"`{t}`" for t in a["conflict_task_ids"][:20]))
    lines.append("")
    lines.append(f"## Seed drift vs pre-repair backup ({len(a['drift'])} tasks)")
    lines.append("")
    for d in a["drift"][:40]:
        lines.append(
            f"- `{d['task_id']}`: seed {d['old_seed']} → {d['new_seed']} "
            f"({d.get('selection_tier')})"
        )
    if len(a["drift"]) > 40:
        lines.append(f"- … +{len(a['drift']) - 40} more")
    lines.append("")
    lines.append("## Focused recheck (original flagged / override / regen)")
    lines.append("")
    for stem, info in a["focused"].items():
        if not info.get("present"):
            lines.append(f"- `{stem}`: MISSING")
            continue
        flags = []
        if info["think_leak"]:
            flags.append("THINK")
        if info["hidden_draft"]:
            flags.append("DRAFT")
        if not info["verified"]:
            flags.append("UNVERIFIED")
        if not info["starts_step1"]:
            flags.append("NO_STEP1")
        if not info["midpoint_ok"]:
            flags.append("BAD_MID")
        status = "OK" if not flags else ",".join(flags)
        lines.append(
            f"- `{stem}`: {status} seed={info['seed']} "
            f"steps={info['step_count']} tier={info['selection_tier']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots.jsonl",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots_samples.jsonl",
    )
    parser.add_argument(
        "--old",
        type=Path,
        default=ROOT
        / "data/baselines/olmo-3-7b/backups/baseline_cots.pre_repair_20260712_052244.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/GATE1_REPORT.md",
    )
    args = parser.parse_args()

    new_rows = load_jsonl(args.new)
    samples = load_jsonl(args.samples)
    old_rows = load_jsonl(args.old) if args.old.exists() else []
    result = audit(new_rows, samples, old_rows)
    md = render_md(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote: {args.out}")
    raise SystemExit(0 if result["go"] else 1)


if __name__ == "__main__":
    main()
