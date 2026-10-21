#!/usr/bin/env python3
"""Re-extract stop-variant answers using first-boxed (prefill commit point)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_judge import resolve_eval_type  # noqa: E402
from evaluator import extract_domain_answer  # noqa: E402
from run_perturbed import deterministic_grade  # noqa: E402

MODEL_PATHS = {
    "1.5b": ROOT / "data" / "1.5b" / "post_injection_runs.rejudged.jsonl",
    "7b": ROOT / "data" / "post_injection_runs.rejudged.jsonl",
    "14b": ROOT / "data" / "14b" / "post_injection_runs.rejudged.jsonl",
    "32b": ROOT / "data" / "32b" / "post_injection_runs.rejudged.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_file(path: Path, *, out_path: Path, dry_run: bool) -> dict:
    rows = load_jsonl(path)
    changed = 0
    stop_rows = 0
    for row in rows:
        if row.get("variant") != "stop":
            continue
        stop_rows += 1
        raw = str(row.get("raw_output", "") or "")
        domain = row.get("domain", "math")
        eval_type = resolve_eval_type(domain, row.get("eval_type"), str(row.get("gold_answer", "")))
        new_ext = extract_domain_answer(raw, domain, eval_type, prefer_first_boxed=True)
        old_ext = row.get("extracted_answer")
        if (new_ext or "") == (old_ext or ""):
            continue
        changed += 1
        if dry_run:
            continue
        row["extracted_answer"] = new_ext
        gold = str(row.get("gold_answer", ""))
        det = deterministic_grade(
            extracted_answer=new_ext,
            gold_answer=gold,
            domain=domain,
            eval_type=eval_type,
        )
        row["det_correct"] = det
        row["judge_fallback"] = True
        row["reextracted_first_boxed"] = True

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_path, rows)

    return {"path": str(path), "out": str(out_path), "stop_rows": stop_rows, "changed": changed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_PATHS.keys()),
        choices=list(MODEL_PATHS.keys()),
    )
    parser.add_argument("--suffix", default=".pre_firstboxed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_changed = 0
    for model in args.models:
        path = MODEL_PATHS[model]
        if not path.exists():
            print(f"[SKIP] {model}: {path} not found", file=sys.stderr)
            continue
        out_path = path.with_name(path.name + ".first_commit.jsonl")
        stats = process_file(path, out_path=out_path, dry_run=args.dry_run)
        print(f"[{model}] stop_rows={stats['stop_rows']} changed={stats['changed']}")
        total_changed += stats["changed"]

    print(f"\nTotal changed stop rows: {total_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
