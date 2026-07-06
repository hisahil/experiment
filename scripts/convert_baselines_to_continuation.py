#!/usr/bin/env python3
"""
Convert verified baseline CoTs to a continuation-friendly format.

Why: our Script-1 baselines store reasoning as "Step N: ..." lines joined by a
single newline. DeepSeek-R1-Distill's natural think-block reasoning has no
"Step N:" prefixes and separates steps/paragraphs with a blank line ("\\n\\n").
For continuation mode (open <think> block, model continues its OWN trace) the
appended trace should look like the model's own output, so we:

  1. strip the "Step N:" prefix from each entry in `steps`
  2. strip pseudo-reasoning tags (<reasoning>, </thinking>, etc.)
  3. join the cleaned steps with "\\n\\n"

The original files are NOT modified. For each input we write a mirrored
`*_continuation.jsonl` next to it, preserving every original field and adding:

  - continuation_steps: list[str]   # steps without "Step N:" prefix
  - continuation_trace: str          # continuation_steps joined by "\\n\\n"

We derive these from `steps` (not `baseline_cot`) on purpose: `steps` excludes
the trailing "Final answer:" line, so the converted trace contains reasoning
only and the model is free to continue toward its own answer.

Usage:
  python scripts/convert_baselines_to_continuation.py            # all 3 models
  python scripts/convert_baselines_to_continuation.py --dry-run  # preview only
  python scripts/convert_baselines_to_continuation.py --models 7b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Matches a leading "Step 12:" / "Step 3 :" prefix (case-insensitive, tolerant
# of extra spaces). Only the prefix is removed; the step text is kept verbatim.
STEP_PREFIX_RE = re.compile(r"^\s*Step\s+\d+\s*:\s*", re.IGNORECASE)

# Reasoning-tag detector covering all model families we run cross-model:
#   DeepSeek / GLM / Phi-4:  <think>, </think>, </thinking>, ...
#   Gemma 4:                 <|channel|>thought, <|channel|>, <|think|>
#   Ministral / Magistral:   [THINK], [/THINK]
# Matches accidental tag leakage into the reasoning trace *after* step-prefix
# stripping (fires the sanity error in convert_file).
TAG_STRIP_RE = re.compile(
    r"(?:"
    r"</?\s*(?:reasoning|thinking|thought|think)\s*>"          # <think>, </thinking>, ...
    r"|<\|(?:channel|think)\|>(?:thought)?"                       # <|channel|>thought, <|channel|>, <|think|>
    r"|\[/?THINK\]"                                                # [THINK], [/THINK]
    r")",
    re.IGNORECASE,
)

# Legacy DeepSeek per-checkpoint slugs (default set); new cross-model slugs
# (glm-z1-9b, gemma-4-12b, ministral-3-14b) are passed explicitly via --models
# to avoid accidental conversions.
DEFAULT_MODELS = ["7b", "14b", "32b"]
TRACE_SEPARATOR = "\n\n"


def strip_step_prefix(step: str) -> str:
    """Remove a leading 'Step N:' prefix, pseudo-tags, and surrounding whitespace."""
    s = STEP_PREFIX_RE.sub("", str(step))
    s = TAG_STRIP_RE.sub("", s)
    return s.strip()


def build_continuation_fields(record: dict) -> dict:
    """Return {continuation_steps, continuation_trace} derived from record['steps']."""
    raw_steps = list(record.get("steps") or [])
    cont_steps = [s for s in (strip_step_prefix(s) for s in raw_steps) if s]
    return {
        "continuation_steps": cont_steps,
        "continuation_trace": TRACE_SEPARATOR.join(cont_steps),
    }


def convert_record(record: dict) -> dict:
    """Copy the record and add continuation fields (original fields preserved)."""
    out = dict(record)
    out.update(build_continuation_fields(record))
    return out


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc


def convert_file(src: Path, dst: Path, *, dry_run: bool) -> tuple[int, int]:
    """Convert one baseline file. Returns (records, empty_step_records)."""
    n = 0
    empty = 0
    out_lines: list[str] = []
    for rec in iter_jsonl(src):
        converted = convert_record(rec)
        if not converted["continuation_steps"]:
            empty += 1
        trace = converted["continuation_trace"]
        if TAG_STRIP_RE.search(trace):
            tid = rec.get("task_id", "?")
            raise ValueError(f"{src}: tag leaked into continuation_trace for {tid}")
        out_lines.append(json.dumps(converted, ensure_ascii=False))
        n += 1
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return n, empty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model subdirs under data/baselines/ to convert (default: 7b 14b 32b).",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=ROOT / "data" / "baselines",
        help="Base directory containing per-model baseline subdirectories.",
    )
    parser.add_argument(
        "--in-name",
        default="baseline_cots.jsonl",
        help="Input filename inside each model subdir.",
    )
    parser.add_argument(
        "--out-name",
        default="baseline_cots_continuation.jsonl",
        help="Output filename inside each model subdir.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    args = parser.parse_args()

    total = 0
    any_missing = False
    for model in args.models:
        src = args.baselines_dir / model / args.in_name
        dst = args.baselines_dir / model / args.out_name
        if not src.exists():
            print(f"[SKIP] {src} (not found)", file=sys.stderr)
            any_missing = True
            continue
        n, empty = convert_file(src, dst, dry_run=args.dry_run)
        total += n
        action = "would write" if args.dry_run else "wrote"
        note = f" ({empty} record(s) had no steps)" if empty else ""
        print(f"[{model}] {action} {n} record(s) -> {dst}{note}")

    print(f"\nDone. {total} record(s) across {len(args.models)} model(s).")
    return 1 if any_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
