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
  logic   — Knights-and-Knaves + JustLogic (optional, enabled flag); balanced
            over 3 difficulty buckets per source (n_people / depth tiers); equal
            max_examples per source (see load_knights_knaves, load_justlogic).

Note: logic bucket balance built here can be preserved when a scope caps
max_per_domain below pool size via experiment_scope hierarchical sampling.

Output JSONL schema (one JSON object per line):
  task_id        — unique key across Scripts 1–4
  domain         — math | science | logic; filtered by experiment_scope
  level          — math only: MATH-500 difficulty level 1..5 (int); else absent
  subject        — math only: MATH-500 subject (e.g. Algebra); else absent
  prompt         — task text sent to the experiment model (Scripts 1–3)
  gold_answer    — reference answer; Scripts 1/3/4 grade via is_correct()
  eval_type      — symbolic (math), numeric (SciBench), logic (K&K), truth_value (JustLogic)
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
from load_justlogic import build_justlogic_tasks  # noqa: E402


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


def _math500_row(row: dict, gold: str, decoy_rng: random.Random) -> dict:
    """Build one output task dict from a MATH-500 row (adds level + subject)."""
    unique_id = str(row["unique_id"])
    return {
        "task_id": math500_task_id(unique_id),
        "domain": "math",
        "level": _math500_level_num(row.get("level", "")),
        "subject": str(row.get("subject", "")).strip(),
        "prompt": str(row["problem"]).strip(),
        "gold_answer": gold,
        "eval_type": "symbolic",
        "source_dataset": "math500",
        "source_id": unique_id,
        "decoy_answer": math_decoy(gold, decoy_rng),
    }


def sample_math500(cfg: dict, seed: int) -> list[dict]:
    """
    Build math-domain rows from HuggingFaceH4/MATH-500 (test split).

    Two modes (controlled by domains.math.stratified):
      * Flat (default): filter to configured levels, draw max_examples without
        replacement.
      * Stratified (stratified: true): draw per_level tasks from EACH configured
        level so the candidate pool is balanced across difficulty. Used for the
        intersection task-selection (per_level=40, levels=[1..5] => 200 tasks).

    Every output row carries `level` (int) and `subject` (str) for downstream
    level-stratified analysis and intersection sampling.
    """
    ds_cfg = cfg["domains"]["math"]
    ds = load_hf_dataset(ds_cfg)
    levels = list(ds_cfg.get("levels", [4, 5]))
    level_set = set(levels)

    # Group eligible rows (non-empty gold) by difficulty level.
    by_level: dict[int, list[tuple[dict, str]]] = {lv: [] for lv in level_set}
    for row in ds:
        level_num = _math500_level_num(row.get("level", ""))
        if level_num not in level_set:
            continue
        gold = parse_math500_gold(str(row.get("answer", "")), str(row.get("solution", "")))
        if not gold.strip():
            continue
        by_level[level_num].append((row, gold))

    decoy_rng = random.Random(seed)
    rows: list[dict] = []

    if ds_cfg.get("stratified", False):
        per_level_cfg = ds_cfg["per_level"]
        # per_level may be an int (uniform across levels) or a dict mapping
        # level -> count. A count of "all" (or any value >= the level pool size)
        # means: take the ENTIRE pool for that level. This is used to maximise the
        # hard Level-5 yield in one pass (no iterative backfill needed).
        # Deterministic per-level draw: derive a distinct seed per level so the
        # draw is reproducible and independent across levels.
        for lv in sorted(level_set):
            pool = by_level[lv]
            if isinstance(per_level_cfg, dict):
                # YAML dict keys may be parsed as int or str; accept both.
                raw = per_level_cfg.get(lv, per_level_cfg.get(str(lv)))
                if raw is None:
                    raise ValueError(
                        f"stratified per_level dict has no entry for Level {lv}."
                    )
            else:
                raw = per_level_cfg
            if isinstance(raw, str) and raw.strip().lower() == "all":
                want = len(pool)
            else:
                want = int(raw)
            if want > len(pool):
                print(
                    f"[WARN] MATH-500 Level {lv}: requested {want} but only "
                    f"{len(pool)} gradable rows available; taking all {len(pool)}.",
                    file=sys.stderr,
                )
                want = len(pool)
            if want <= 0:
                raise ValueError(
                    f"per_level for Level {lv} resolved to {want} (must be > 0)."
                )
            rng = np.random.RandomState(seed + lv)
            chosen = rng.choice(len(pool), size=want, replace=False)
            for idx in chosen:
                row, gold = pool[int(idx)]
                rows.append(_math500_row(row, gold, decoy_rng))
        return rows

    # Flat mode (legacy): single draw across all eligible levels.
    eligible: list[tuple[dict, str]] = [pair for lv in level_set for pair in by_level[lv]]
    max_examples = int(ds_cfg["max_examples"])
    if len(eligible) < max_examples:
        raise ValueError(
            f"MATH-500 level {sorted(level_set)} pool has {len(eligible)} rows with non-empty gold; "
            f"need at least {max_examples}."
        )
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(eligible), size=max_examples, replace=False)
    for idx in chosen:
        row, gold = eligible[int(idx)]
        rows.append(_math500_row(row, gold, decoy_rng))
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
# Logic domain — Knights-and-Knaves + JustLogic (optional)
# ---------------------------------------------------------------------------


def sample_logic(cfg: dict, rng: random.Random) -> list[dict]:
    """
    Optional logic domain (K&K + JustLogic). Returns [] when disabled.

    Each source uses max_examples // 3 tasks per difficulty bucket:
      K&K        — n_people 3/4/5 (eval_type=logic)
      JustLogic  — depth tiers low/mid/high (eval_type=truth_value)

    Set equal max_examples on both sub-sources for a 50/50 logic pool.
    """
    ds_cfg = cfg["domains"].get("logic")
    if not ds_cfg or not ds_cfg.get("enabled", False):
        return []
    rows: list[dict] = []
    kk_cfg = ds_cfg.get("knights_knaves")
    if kk_cfg and kk_cfg.get("enabled", True):
        rows.extend(build_kk_tasks(kk_cfg, rng))
    jl_cfg = ds_cfg.get("justlogic")
    if jl_cfg and jl_cfg.get("enabled", True):
        rows.extend(build_justlogic_tasks(jl_cfg, rng))
    return rows


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
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        choices=["math", "science", "logic"],
        default=None,
        help="Restrict pool to these domains (default: all configured). "
        "Use '--domains math' for the stratified intersection candidate pool.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)
    out_path = args.output if args.output is not None else ROOT / cfg["output"]
    selected = set(args.domains) if args.domains else {"math", "science", "logic"}

    try:
        # Stage 0 pool: math + science + logic (if enabled). Order is stable for logging.
        rows: list[dict] = []
        if "math" in selected:
            rows.extend(sample_math(cfg, rng, seed))
        if "science" in selected:
            rows.extend(sample_science(cfg, rng))
        if "logic" in selected:
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
