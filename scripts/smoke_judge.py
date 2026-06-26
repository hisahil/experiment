#!/usr/bin/env python3
"""Live smoke test for LLM-as-a-Judge API before expensive GPU runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer_judge import DEFAULT_JUDGE_MODEL, grade_sample  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify judge API/model before Script 1.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    judge_cfg = dict(cfg.get("judge", {}))
    model = judge_cfg.get("model", DEFAULT_JUDGE_MODEL)
    print(f"Judge smoke test: model={model}")

    result = grade_sample(
        question="What is 2+2?",
        gold_answer="4",
        answer_content="Step 1: 2+2=4\nFinal answer: 4",
        extracted_answer="4",
        domain="math",
        eval_type="exact",
        judge_cfg=judge_cfg,
        dry_run=False,
    )

    print(f"  judge_fallback={result.get('judge_fallback')}")
    print(f"  judge_output={result.get('judge_output')!r}")
    print(f"  judge_raw_output={result.get('judge_raw_output')!r}")
    print(f"  correct={result.get('correct')}")

    if result.get("judge_fallback"):
        print("FAIL: judge fell back (API error, missing key, or unparseable response)", file=sys.stderr)
        return 1
    if not result.get("correct"):
        print("FAIL: expected Correct for trivial 2+2=4 case", file=sys.stderr)
        return 1

    print("OK: judge API reachable and parsed Correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
