"""JustLogic loader for the logic domain (coexists with Knights-and-Knaves).

Dataset: michaelchenkj/JustLogic (Hugging Face).
Given premises (``paragraph``), the model judges whether ``question`` is
True, False, or Uncertain. Graded by ``eval_type=truth_value`` in
``src/evaluator.py``. Imported by ``scripts/prepare_tasks.py``.
"""

from __future__ import annotations

import random

JL_HF_PATH = "michaelchenkj/JustLogic"
DEFAULT_DATA_FILE = "validate_dataset.csv"
DEFAULT_TIERS: dict[str, list[int]] = {
    "low": [1, 2],
    "mid": [3, 4],
    "high": [5, 6, 7],
}

JL_PROMPT_SUFFIX = (
    "\n\nBased solely on the premises above, determine whether the following "
    "statement is True, False, or Uncertain (cannot be determined from the "
    "premises alone).\n\n"
    "Statement: {question}\n\n"
    "End your response with a single line in exactly this format:\n"
    "Final answer: True/False/Uncertain"
)


def normalize_label(label: str) -> str:
    """Map dataset labels (TRUE/FALSE/Uncertain) to canonical True/False/Uncertain."""
    text = label.strip()
    upper = text.upper()
    if upper == "TRUE":
        return "True"
    if upper == "FALSE":
        return "False"
    if upper == "UNCERTAIN":
        return "Uncertain"
    return text


def justlogic_decoy(gold: str, rng: random.Random) -> str:
    """Pick one of the two wrong truth-value labels."""
    gold_norm = normalize_label(gold)
    options = {"True", "False", "Uncertain"}
    others = list(options - {gold_norm})
    return rng.choice(others)


def depth_to_tier(depth: int, tiers: dict[str, list[int]]) -> str | None:
    for tier_name, depths in tiers.items():
        if depth in depths:
            return tier_name
    return None


def jl_task_id(source_id: int) -> str:
    return f"justlogic_{source_id}"


def _parse_tiers(ds_cfg: dict) -> dict[str, list[int]]:
    tiers = ds_cfg.get("tiers", DEFAULT_TIERS)
    return {str(k): [int(d) for d in v] for k, v in tiers.items()}


def build_justlogic_tasks(ds_cfg: dict, rng: random.Random) -> list[dict]:
    """
    Build logic-domain task rows from JustLogic validate split.

    ds_cfg keys:
      hf_path       (default michaelchenkj/JustLogic)
      data_file     (default validate_dataset.csv)
      tiers         depth -> tier mapping (default low/mid/high)
      max_examples  total tasks across tiers (default 12)
    """
    from datasets import load_dataset

    hf_path = ds_cfg.get("hf_path", JL_HF_PATH)
    data_file = ds_cfg.get("data_file", DEFAULT_DATA_FILE)
    tiers = _parse_tiers(ds_cfg)
    tier_names = list(tiers.keys())
    max_examples = int(ds_cfg.get("max_examples", 12))
    if not tier_names:
        return []
    per_bucket = max(1, max_examples // len(tier_names))

    ds = load_dataset(
        hf_path,
        data_files={"validation": data_file},
        split="validation",
    )

    by_tier: dict[str, list[int]] = {t: [] for t in tier_names}
    for idx in range(len(ds)):
        depth = int(ds[idx]["depth"])
        tier = depth_to_tier(depth, tiers)
        if tier is not None:
            by_tier[tier].append(idx)

    rows: list[dict] = []
    for tier_name in tier_names:
        indices = by_tier[tier_name]
        rng.shuffle(indices)
        for idx in indices[:per_bucket]:
            row = ds[idx]
            depth = int(row["depth"])
            source_id = int(row["id"])
            gold = normalize_label(str(row["label"]))
            paragraph = str(row["paragraph"]).strip()
            question = str(row["question"]).strip()
            prompt = paragraph + JL_PROMPT_SUFFIX.format(question=question)
            rows.append(
                {
                    "task_id": jl_task_id(source_id),
                    "domain": "logic",
                    "prompt": prompt,
                    "gold_answer": gold,
                    "eval_type": "truth_value",
                    "source_dataset": "justlogic",
                    "source_id": str(source_id),
                    "decoy_answer": justlogic_decoy(gold, rng),
                    "depth": depth,
                    "logic_source": "justlogic",
                    "logic_difficulty_tier": tier_name,
                }
            )
    return rows[:max_examples]
