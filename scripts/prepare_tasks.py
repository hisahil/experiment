#!/usr/bin/env python3
"""
Build data/experiment_tasks.jsonl from benchmark datasets (Hugging Face).

Usage:
  pip install datasets
  python scripts/prepare_tasks.py --config configs/datasets.yaml
  python scripts/prepare_tasks.py --config configs/datasets.yaml --output data/experiment_tasks_math500.jsonl
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


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_prompt_question(rows: list[dict]) -> list[dict]:
    """
    Guarantee every task carries BOTH `prompt` and `question` with identical
    content (downstream readers accept either key). Applied to all domains and
    fallback loaders so the contract holds for the whole pool.
    """
    for row in rows:
        if "prompt" not in row and "question" in row:
            row["prompt"] = row["question"]
        row["question"] = row.get("prompt", "")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", answer)
    return nums[-1] if nums else answer.strip()


def load_hf_dataset(ds_cfg: dict):
    from datasets import load_dataset

    path = ds_cfg["hf_path"]
    split = ds_cfg["split"]
    config = ds_cfg.get("config")
    if config:
        return load_dataset(path, config, split=split)
    return load_dataset(path, split=split)


def sample_gsm8k(cfg: dict, rng: random.Random) -> list[dict]:
    ds_cfg = cfg["domains"]["math_gsm8k"] if "math_gsm8k" in cfg["domains"] else cfg["domains"]["math"]
    ds = load_hf_dataset(ds_cfg)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows: list[dict] = []
    for idx in indices[: ds_cfg["max_examples"]]:
        row = ds[idx]
        gold = parse_gsm8k_answer(row["answer"])
        rows.append(
            {
                "task_id": f"gsm8k_{idx}",
                "domain": "math",
                "prompt": row["question"].strip(),
                "gold_answer": gold,
                "eval_type": "numeric",
                "source_dataset": "gsm8k",
                "source_id": str(idx),
                "decoy_answer": numeric_decoy(gold, rng),
            }
        )
    return rows


def _math500_level_num(level: str) -> int | None:
    match = re.search(r"(\d+)", str(level))
    return int(match.group(1)) if match else None


def sample_math500(cfg: dict, seed: int) -> list[dict]:
    ds_cfg = cfg["domains"]["math"]
    ds = load_hf_dataset(ds_cfg)
    levels = set(ds_cfg.get("levels", [4, 5]))
    filtered: list[dict] = []
    for row in ds:
        level_num = _math500_level_num(row.get("level", ""))
        if level_num in levels:
            filtered.append(row)

    max_examples = int(ds_cfg["max_examples"])
    if len(filtered) < max_examples:
        raise ValueError(
            f"MATH-500 level {sorted(levels)} pool has {len(filtered)} rows; "
            f"need at least {max_examples}."
        )

    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(filtered), size=max_examples, replace=False)

    rows: list[dict] = []
    decoy_rng = random.Random(seed)
    for idx in chosen:
        row = filtered[int(idx)]
        unique_id = str(row["unique_id"])
        gold = parse_math500_gold(str(row.get("answer", "")), str(row.get("solution", "")))
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
    ds_cfg = cfg["domains"]["math"]
    dataset = str(ds_cfg.get("dataset", "gsm8k")).lower()
    if dataset == "math500":
        return sample_math500(cfg, seed)
    return sample_gsm8k(cfg, rng)


def _clean_scibench_number(value: str) -> str:
    """SciBench answer_number is a string that may have a leading '+'/spaces."""
    return str(value).strip().lstrip("+").strip()


def sample_scibench(cfg: dict, rng: random.Random) -> list[dict]:
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


def sample_mmlu(cfg: dict, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds_cfg = cfg["domains"]["science_mmlu"] if "science_mmlu" in cfg["domains"] else cfg["domains"]["science"]
    subjects = ds_cfg["subjects"]
    per_subject = max(1, ds_cfg["max_examples"] // len(subjects))
    rows: list[dict] = []
    for subject in subjects:
        ds = load_dataset(ds_cfg["hf_path"], subject, split=ds_cfg["split"])
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices[:per_subject]:
            row = ds[idx]
            choices = row["choices"]
            answer_idx = int(row["answer"])
            gold = choices[answer_idx]
            letters = "ABCD"
            choice_lines = "\n".join(
                f"{letters[i]}. {choices[i]}" for i in range(len(choices))
            )
            wrong_idx = (answer_idx + rng.randint(1, len(choices) - 1)) % len(choices)
            rows.append(
                {
                    "task_id": f"mmlu_{subject}_{idx}",
                    "domain": "science",
                    "prompt": f"{row['question'].strip()}\n\n{choice_lines}\n\nReply with the letter only.",
                    "gold_answer": letters[answer_idx],
                    "eval_type": "exact",
                    "source_dataset": f"mmlu/{subject}",
                    "source_id": str(idx),
                    "decoy_answer": letters[wrong_idx],
                }
            )
    return rows[: ds_cfg["max_examples"]]


def sample_science(cfg: dict, rng: random.Random) -> list[dict]:
    ds_cfg = cfg["domains"]["science"]
    dataset = str(ds_cfg.get("dataset", "mmlu")).lower()
    if dataset == "scibench":
        return sample_scibench(cfg, rng)
    return sample_mmlu(cfg, rng)


def sample_logic(cfg: dict, rng: random.Random) -> list[dict]:
    """Optional logic domain (Knights-and-Knaves). Returns [] when disabled."""
    ds_cfg = cfg["domains"].get("logic")
    if not ds_cfg or not ds_cfg.get("enabled", False):
        return []
    return build_kk_tasks(ds_cfg, rng)


def parse_mbpp_gold(test_list: list, rng: random.Random) -> tuple[str, str] | None:
    if not test_list:
        return None
    test = test_list[0]
    match = re.search(r"==\s*([^)\]]+)", test)
    if not match:
        return None
    gold = match.group(1).strip().strip("'\"")
    try:
        v = int(gold)
        decoy = str(v + rng.choice([1, 2, -1, 5]))
    except ValueError:
        decoy = gold + "_wrong"
    return gold, decoy


def sample_mbpp(cfg: dict, rng: random.Random) -> list[dict]:
    ds_cfg = cfg["domains"]["code"]
    ds = load_hf_dataset(ds_cfg)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows: list[dict] = []
    for idx in indices:
        if len(rows) >= ds_cfg["max_examples"]:
            break
        row = ds[idx]
        parsed = parse_mbpp_gold(row["test_list"], rng)
        if parsed is None:
            continue
        gold, decoy = parsed
        rows.append(
            {
                "task_id": f"mbpp_{row.get('task_id', idx)}",
                "domain": "code",
                "prompt": (
                    f"{row['text'].strip()}\n\n"
                    "Solve in Python. At the end write: Final answer: <value from the first test assertion>"
                ),
                "gold_answer": gold,
                "eval_type": "exact",
                "source_dataset": "mbpp",
                "source_id": str(row.get("task_id", idx)),
                "decoy_answer": decoy,
            }
        )
    return rows


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
        rows: list[dict] = []
        rows.extend(sample_math(cfg, rng, seed))
        rows.extend(sample_science(cfg, rng))
        rows.extend(sample_mbpp(cfg, rng))
        rows.extend(sample_logic(cfg, rng))
    except ImportError:
        print("Install datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    ensure_prompt_question(rows)
    write_jsonl(out_path, rows)
    by_domain: dict[str, int] = {}
    for row in rows:
        by_domain[row["domain"]] = by_domain.get(row["domain"], 0) + 1
    print(f"Wrote {len(rows)} tasks to {out_path}")
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
