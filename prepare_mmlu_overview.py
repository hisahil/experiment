#!/usr/bin/env python3
"""
Export MMLU science-task overviews for browsing (CSV + Excel).

Uses the same Hugging Face dataset, subjects, split, seed, and sampling
logic as scripts/prepare_tasks.py / configs/datasets.yaml.

Usage:
  python prepare_mmlu_overview.py
  python prepare_mmlu_overview.py --config configs/datasets.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "datasets.yaml"
OUTPUT_DIR = ROOT / "data" / "mmlu"
LETTERS = "ABCD"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {choices[i]}" for i in range(len(choices)))


def row_to_record(subject: str, idx: int, row: dict) -> dict:
    choices = list(row["choices"])
    answer_idx = int(row["answer"])
    gold_letter = LETTERS[answer_idx]
    question = str(row["question"]).strip()
    return {
        "task_id": f"mmlu_{subject}_{idx}",
        "subject": subject,
        "source_idx": idx,
        "question": question,
        "choices_formatted": format_choices(choices),
        "choice_a": choices[0] if len(choices) > 0 else "",
        "choice_b": choices[1] if len(choices) > 1 else "",
        "choice_c": choices[2] if len(choices) > 2 else "",
        "choice_d": choices[3] if len(choices) > 3 else "",
        "gold_letter": gold_letter,
        "gold_text": choices[answer_idx],
        "question_preview": question[:180] + ("…" if len(question) > 180 else ""),
    }


def load_subject_frames(ds_cfg: dict) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for subject in ds_cfg["subjects"]:
        ds = load_dataset(ds_cfg["hf_path"], subject, split=ds_cfg["split"])
        records = [row_to_record(subject, i, ds[i]) for i in range(len(ds))]
        frames[subject] = pd.DataFrame(records)
    return frames


def sample_experiment_pool(
    frames: dict[str, pd.DataFrame], ds_cfg: dict, seed: int
) -> pd.DataFrame:
    """Match sample_mmlu() in scripts/prepare_tasks.py."""
    rng = random.Random(seed)
    subjects = ds_cfg["subjects"]
    max_examples = int(ds_cfg["max_examples"])
    per_subject = max(1, max_examples // len(subjects))

    sampled: list[pd.DataFrame] = []
    for subject in subjects:
        df = frames[subject]
        indices = list(range(len(df)))
        rng.shuffle(indices)
        picked = df.iloc[indices[:per_subject]].copy()
        sampled.append(picked)

    out = pd.concat(sampled, ignore_index=True)
    return out.iloc[:max_examples].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MMLU overview tables for browsing.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["domains"]["science"]
    seed = int(cfg.get("seed", 42))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_frames = load_subject_frames(ds_cfg)
    all_df = pd.concat(subject_frames.values(), ignore_index=True)
    sample_df = sample_experiment_pool(subject_frames, ds_cfg, seed)

    def with_id(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.insert(0, "id", range(1, len(out) + 1))
        return out

    column_order = [
        "id",
        "task_id",
        "subject",
        "source_idx",
        "question",
        "question_preview",
        "choices_formatted",
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "gold_letter",
        "gold_text",
    ]

    all_out = with_id(all_df)[column_order]
    sample_out = with_id(sample_df)[column_order]

    all_csv = OUTPUT_DIR / "mmlu_science_all.csv"
    sample_csv = OUTPUT_DIR / "mmlu_science_sample_80.csv"
    all_out.to_csv(all_csv, index=False)
    sample_out.to_csv(sample_csv, index=False)

    excel_path = OUTPUT_DIR / "mmlu_overview.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        all_out.to_excel(writer, sheet_name="all_subjects", index=False)
        sample_out.to_excel(writer, sheet_name="sample_80", index=False)
        for subject, df in subject_frames.items():
            sheet = subject[:31]
            subject_out = with_id(df)[column_order]
            subject_out.to_excel(writer, sheet_name=sheet, index=False)
            per_csv = OUTPUT_DIR / f"mmlu_{subject}.csv"
            subject_out.to_csv(per_csv, index=False)

    print(f"Dataset: {ds_cfg['hf_path']} (per-subject config, split={ds_cfg['split']})")
    print(f"Subjects: {', '.join(ds_cfg['subjects'])}")
    print(f"Seed / max_examples: {seed} / {ds_cfg['max_examples']}")
    print(f"Saved Excel: {excel_path}")
    print(f"Saved CSVs: {all_csv}, {sample_csv}, plus one per subject in {OUTPUT_DIR}")
    print(f"Row counts — all_subjects: {len(all_out)}, sample_80: {len(sample_out)}")
    for subject, df in subject_frames.items():
        print(f"  {subject}: {len(df)} (full test split)")


if __name__ == "__main__":
    main()
