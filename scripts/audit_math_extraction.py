#!/usr/bin/env python3
"""Audit math recovery extraction quality (categories A–E)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluator import extract_domain_answer, is_correct  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())


def _gold_variants(gold: str) -> list[str]:
    """String forms gold may take in raw text (handles \\frac vs a/b)."""
    g = str(gold).strip()
    variants = {g, _norm(g)}
    frac = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}", g)
    if frac:
        variants.add(f"{frac.group(1)}/{frac.group(2)}")
        variants.add(f"\\frac{{{frac.group(1)}}}{{{frac.group(2)}}}")
        variants.add(f"\\dfrac{{{frac.group(1)}}}{{{frac.group(2)}}}")
    slash = re.fullmatch(r"(-?\d+)/(-?\d+)", g)
    if slash:
        variants.add(f"\\frac{{{slash.group(1)}}}{{{slash.group(2)}}}")
        variants.add(f"\\dfrac{{{slash.group(1)}}}{{{slash.group(2)}}}")
    return [v for v in variants if v]


def _present(haystack: str, gold: str) -> bool:
    hay_norm = _norm(haystack)
    for v in _gold_variants(gold):
        if v in haystack or _norm(v) in hay_norm:
            return True
    g = str(gold).strip()
    if re.search(rf"(?<!\d){re.escape(g)}(?!\d)", haystack):
        return True
    if re.search(rf"=\s*{re.escape(g)}\b", haystack):
        return True
    return False


def gold_in_text(raw: str, gold: str) -> bool:
    return _present(raw, gold)


def get_final_region(raw: str) -> str:
    body = raw
    idx = raw.rfind("</think>")
    if idx != -1:
        body = raw[idx + len("</think>") :].strip() or raw
    markers = (
        r"\\boxed\{",
        r"\*\*Final\s+Answer",
        r"\*\*Answer",
        r"\*\*Conclusion",
        r"Final\s+Answer:",
    )
    last_pos = -1
    for marker in markers:
        for match in re.finditer(marker, body, re.I | re.M):
            last_pos = max(last_pos, match.start())
    # No explicit marker: only the last couple of lines count as "final"
    # (a gold token deep in scratch work must NOT be treated as the final answer).
    return body[last_pos:] if last_pos >= 0 else body[-250:]


def gold_in_final(raw: str, gold: str) -> bool:
    return _present(get_final_region(raw), gold)


def has_contradiction(raw: str, gold: str) -> bool:
    final = get_final_region(raw)
    answers = re.findall(r"\*\*Answer:?\*\*\s*(.+)$", final, re.I | re.M)
    if len(answers) > 1:
        return True
    nums = set(re.findall(r"\b\d{2,4}\b", final[-400:]))
    g = str(gold).strip()
    if g.isdigit() and len(nums) >= 2 and g in nums and any(n != g for n in nums):
        return True
    return False


def classify_run(run: dict) -> str:
    """
    Category by whether GOLD is the model's FINAL answer (not scratch work).

    A: graded correct, gold in final region.
    B: graded incorrect, gold NOT in final region (model's final answer is wrong).
    C: graded incorrect, gold IS in final region (parser failed to pick it) — real bug.
    D: graded correct, gold absent from text (parser pulled gold from scratch by luck).
    E: contradictory finals / truncated (no parseable final answer).

    Gold presence is judged on the FINAL region only. Gold appearing in
    intermediate computation does not count (that was the prior mis-categorization).
    """
    raw = run["raw_output"]
    gold = run["gold_answer"]
    ev = run.get("eval_type", "symbolic")
    extracted = extract_domain_answer(raw, "math", ev)
    graded = bool(is_correct(extracted, gold, ev)) if extracted else False
    gin_anywhere = gold_in_text(raw, gold)
    gfin = gold_in_final(raw, gold)
    contra = has_contradiction(raw, gold)

    if graded and gfin:
        return "A"
    if graded and not gin_anywhere:
        return "D"
    if extracted is None:
        # No parseable answer at all (token-limit truncation or no final marker).
        return "E"
    if not graded and gfin:
        return "C"  # gold IS the final answer but parser missed it → real bug
    if contra:
        return "E"
    # graded incorrect and gold not in final region → model's final answer is wrong
    return "B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/post_injection_runs.jsonl")
    args = parser.parse_args()

    runs = []
    with Path(args.input).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("variant") == "recovery" and r.get("domain") == "math":
                runs.append(r)

    counts = {k: 0 for k in "ABCDE"}
    by_cat: dict[str, list] = {k: [] for k in "ABCDE"}
    for run in runs:
        cat = classify_run(run)
        counts[cat] += 1
        if cat in {"C", "D", "E"}:
            ex = extract_domain_answer(run["raw_output"], "math", run.get("eval_type"))
            by_cat[cat].append(
                (run["task_id"], run["error_type"], ex, run["gold_answer"])
            )

    total = len(runs)
    print(f"Math recovery runs: {total}\n")
    for key in "ABCDE":
        print(f"  {key}: {counts[key]} ({100 * counts[key] / total:.1f}%)")
    for key in "CDE":
        print(f"\n{key} ({counts[key]}):")
        for row in by_cat[key][:15]:
            print(f"  {row}")


if __name__ == "__main__":
    main()
