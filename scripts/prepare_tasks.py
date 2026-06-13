#!/usr/bin/env python3
"""
Build data/experiment_tasks.jsonl from benchmark datasets (Hugging Face).

This is Stage 0 of the pipeline: it creates a fixed, reproducible TASK POOL.
Scripts 1–4 do not run here. Runtime scopes (pilot_gpu, full) subsample this
pool later via experiment_scope.select_tasks() in configs/experiment.yaml.

Usage:
  pip install datasets
  python scripts/prepare_tasks.py --config configs/datasets.yaml
  python scripts/prepare_tasks.py --config configs/datasets.yaml --output data/experiment_tasks_pilot_gpu.jsonl

Selection methodology (per domain, seed from configs/datasets.yaml, default 42):
  math    — MATH-500 test split; filter Level 4/5; random sample without
            replacement (np.random.RandomState). No subtopic stratification.
  science — SciBench; shuffle all rows; take first N with non-empty gold.
  logic   — Knights-and-Knaves test files; balanced buckets over n_people
            (see load_knights_knaves.build_kk_tasks). Optional (enabled flag).

Note: logic bucket balance built here can be diluted when a scope caps
max_per_domain below the pool size (see experiment_scope.select_tasks).

Output JSONL schema (one JSON object per line):
  task_id        — unique key across Scripts 1–4
  domain         — math | science | logic; filtered by experiment_scope
  prompt         — task text sent to the experiment model (Scripts 1–3)
  gold_answer    — reference answer; Scripts 1/3/4 grade via is_correct()
  eval_type      — grading rule: symbolic (math), numeric (SciBench), logic (K&K)
  decoy_answer   — plausible wrong answer for Script 2 (Claude injections)
  source_dataset — benchmark origin label (e.g. math500, scibench/atkins)
  source_id      — id within the source dataset (traceability)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from math_utils import (  # noqa: E402
    math500_task_id,
    math_decoy,
    numeric_decoy,
    parse_math500_gold,
)
from load_knights_knaves import build_kk_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Overwrite JSONL pool file (one compact JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Hugging Face dataset loading
# ---------------------------------------------------------------------------


def load_hf_dataset(ds_cfg: dict):
    """
    Load a Hugging Face split from domains.<name> in configs/datasets.yaml.

    ds_cfg keys: hf_path, split, optional config (e.g. scibench default).
    """
    from datasets import load_dataset

    path = ds_cfg["hf_path"]
    split = ds_cfg["split"]
    config = ds_cfg.get("config")
    if config:
        return load_dataset(path, config, split=split)
    return load_dataset(path, split=split)


# ---------------------------------------------------------------------------
# Math domain — MATH-500 (symbolic grading)
# ---------------------------------------------------------------------------


def _math500_level_num(level: str) -> int | None:
    """Parse 'Level 4' / 'Level 5' style labels to an integer difficulty level."""
    match = re.search(r"(\d+)", str(level))
    return int(match.group(1)) if match else None


def sample_math500(cfg: dict, seed: int) -> list[dict]:
    """
    Build math-domain rows from HuggingFaceH4/MATH-500 (test split).

    Steps:
      1. Filter to configured levels (default 4 and 5 — harder problems only).
      2. Assert the filtered pool is large enough for max_examples.
      3. Drop rows with empty gold after parse_math500_gold (ungradable).
      4. Draw max_examples indices without replacement (fixed seed).
      5. Parse gold from answer/solution; attach math_decoy for Script 2.

    Example (max_examples=80, levels=[4,5], seed=42):
      ~312 Level-4/5 rows in pool → 80 chosen → task_id like math500_test_algebra_529.json
    """
    ds_cfg = cfg["domains"]["math"]
    ds = load_hf_dataset(ds_cfg)
    levels = set(ds_cfg.get("levels", [4, 5]))

    # Eligibility filter: deliberate difficulty cap, not random noise.
    filtered: list[dict] = []
    for row in ds:
        level_num = _math500_level_num(row.get("level", ""))
        if level_num in levels:
            filtered.append(row)

    eligible: list[tuple[dict, str]] = []
    for row in filtered:
        gold = parse_math500_gold(str(row.get("answer", "")), str(row.get("solution", "")))
        if not gold.strip():
            continue
        eligible.append((row, gold))

    max_examples = int(ds_cfg["max_examples"])
    if len(eligible) < max_examples:
        raise ValueError(
            f"MATH-500 level {sorted(levels)} pool has {len(eligible)} rows with non-empty gold; "
            f"need at least {max_examples}."
        )

    # np RandomState for index draw; separate random.Random for decoy variation.
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(eligible), size=max_examples, replace=False)

    rows: list[dict] = []
    decoy_rng = random.Random(seed)
    for idx in chosen:
        row, gold = eligible[int(idx)]
        unique_id = str(row["unique_id"])
        rows.append(
            {
                "task_id": math500_task_id(unique_id),
                "domain": "math",
                "prompt": str(row["problem"]).strip(),
                "gold_answer": gold,
                "eval_type": "symbolic",
                "source_dataset": "math500",
                "source_id": unique_id,
                "decoy_answer": math_decoy(gold, decoy_rng),
            }
        )
    return rows


def sample_math(cfg: dict, rng: random.Random, seed: int) -> list[dict]:
    """Dispatch math sampling; only math500 is configured today."""
    ds_cfg = cfg["domains"]["math"]
    dataset = str(ds_cfg.get("dataset", "math500")).lower()
    if dataset != "math500":
        raise ValueError(f"Unsupported math dataset '{dataset}'; only math500 is configured.")
    return sample_math500(cfg, seed)


# ---------------------------------------------------------------------------
# Science domain — SciBench (numeric grading)
# ---------------------------------------------------------------------------


def _clean_scibench_number(value: str) -> str:
    """SciBench answer_number is a string that may have a leading '+'/spaces."""
    return str(value).strip().lstrip("+").strip()


def sample_scibench(cfg: dict, rng: random.Random) -> list[dict]:
    """
    Build science-domain rows from xw27/scibench.

    Steps:
      1. Shuffle all dataset indices (rng from main(), same seed as config).
      2. Walk shuffled order until max_examples rows with valid gold are collected.
      3. Skip rows with empty answer_number (ungradable — not a sampling bias).
      4. Append unit hint + fixed answer format to the prompt for parseable outputs.
      5. numeric_decoy for Script 2 (slightly perturbed number).

    Split is configured in datasets.yaml (currently train). Selection is uniform
    random over eligible rows, not stratified by SciBench subject/source.
    """
    ds_cfg = cfg["domains"]["science"]
    ds = load_hf_dataset(ds_cfg)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    rows: list[dict] = []
    for idx in indices:
        if len(rows) >= ds_cfg["max_examples"]:
            break
        row = ds[idx]
        gold = _clean_scibench_number(row.get("answer_number", ""))
        if not gold:
            continue
        source = str(row.get("source", "scibench")).strip() or "scibench"
        unit = str(row.get("unit", "")).strip()
        unit_hint = f" Give the numerical value in {unit}." if unit else ""
        rows.append(
            {
                "task_id": f"scibench_{source}_{idx}",
                "domain": "science",
                "prompt": (
                    f"{str(row['problem_text']).strip()}\n\n"
                    f"Solve the problem.{unit_hint} "
                    "End your response with: Final answer: <number>"
                ),
                "gold_answer": gold,
                "eval_type": "numeric",
                "source_dataset": f"scibench/{source}",
                "source_id": str(row.get("problemid", idx)).strip(),
                "decoy_answer": numeric_decoy(gold, rng),
            }
        )
    return rows


def sample_science(cfg: dict, rng: random.Random) -> list[dict]:
    """Dispatch science sampling; only scibench is configured today."""
    ds_cfg = cfg["domains"]["science"]
    dataset = str(ds_cfg.get("dataset", "scibench")).lower()
    if dataset != "scibench":
        raise ValueError(f"Unsupported science dataset '{dataset}'; only scibench is configured.")
    return sample_scibench(cfg, rng)


# ---------------------------------------------------------------------------
# Logic domain — Knights-and-Knaves (optional, exact logic grading)
# ---------------------------------------------------------------------------


def sample_logic(cfg: dict, rng: random.Random) -> list[dict]:
    """
    Optional logic domain (Knights-and-Knaves). Returns [] when disabled.

    Delegates to build_kk_tasks() in load_knights_knaves.py, which:
      - loads test JSONL files per n_people bucket (3, 4, 5 characters),
      - draws an equal number of puzzles per bucket (balanced difficulty),
      - sets eval_type=logic and a canonical gold/decoy role assignment.

    Rows may include n_people for traceability; experiment_scope caps tasks
    per domain without re-balancing buckets (pilot_gpu: 8 of 24).
    """
    ds_cfg = cfg["domains"].get("logic")
    if not ds_cfg or not ds_cfg.get("enabled", False):
        return []
    return build_kk_tasks(ds_cfg, rng)


# ---------------------------------------------------------------------------
# CLI — concatenate domain pools and write JSONL
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare experiment task pool from benchmarks.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/datasets.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output JSONL path (default: value of output in config).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)
    out_path = args.output if args.output is not None else ROOT / cfg["output"]

    try:
        # Stage 0 pool: math + science + logic (if enabled). Order is stable for logging.
        rows: list[dict] = []
        rows.extend(sample_math(cfg, rng, seed))
        rows.extend(sample_science(cfg, rng))
        rows.extend(sample_logic(cfg, rng))
    except ImportError:
        print("Install datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    write_jsonl(out_path, rows)
    by_domain: dict[str, int] = {}
    for row in rows:
        by_domain[row["domain"]] = by_domain.get(row["domain"], 0) + 1
    print(f"Wrote {len(rows)} tasks to {out_path}")
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
