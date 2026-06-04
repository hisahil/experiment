"""Knights-and-Knaves (K&K) loader for the OPTIONAL logic domain.

Dataset: K-and-K/knights-and-knaves (Hugging Face).
Difficulty is controlled by the number of characters per puzzle (``n_people``);
the GPU pilot uses 3-5 people. Each puzzle has a unique role assignment, so
grading is judge-free exact match on the (Name -> knight/knave) tuple.

Gold answers are emitted in a canonical, easy-to-parse form, e.g.
    "Penelope=knave, David=knave, Zoey=knight"
which ``src/evaluator.py`` (logic eval_type) parses and compares against the
model's final assignment. This module is imported by scripts/prepare_tasks.py.
"""

from __future__ import annotations

import random


KK_HF_PATH = "K-and-K/knights-and-knaves"

# Instruction appended to each quiz so the model emits a parseable assignment.
KK_PROMPT_SUFFIX = (
    "\n\nDetermine for each inhabitant whether they are a knight or a knave. "
    "End your response with a single line in exactly this format:\n"
    "Final answer: Name1=knight/knave, Name2=knight/knave, ..."
)


def kk_role(is_knight: bool) -> str:
    """Map the dataset boolean solution (True=knight) to a canonical role word."""
    return "knight" if is_knight else "knave"


def kk_gold_answer(names: list[str], solution: list[bool]) -> str:
    """Canonical gold string: 'Name1=role1, Name2=role2, ...' (dataset order)."""
    return ", ".join(
        f"{name}={kk_role(bool(val))}" for name, val in zip(names, solution)
    )


def kk_decoy_answer(names: list[str], solution: list[bool], rng: random.Random) -> str:
    """Plausible wrong assignment: flip exactly one inhabitant's role."""
    if not solution:
        return ""
    flipped = [bool(v) for v in solution]
    idx = rng.randrange(len(flipped))
    flipped[idx] = not flipped[idx]
    return kk_gold_answer(names, flipped)


def kk_task_id(n_people: int, index: int) -> str:
    return f"kk_p{n_people}_{index}"


def _kk_data_file(split: str, n_people: int) -> str:
    """HF data_files glob for one people-count bucket (handles num100/num1000)."""
    return f"{split}/people{n_people}_num*.jsonl"


def build_kk_tasks(ds_cfg: dict, rng: random.Random) -> list[dict]:
    """
    Build logic-domain task rows from the K&K dataset.

    ds_cfg keys:
      hf_path      (default K-and-K/knights-and-knaves)
      split        (default 'test')
      n_people     list of character counts (default [3, 4, 5])
      max_examples total tasks across the n_people buckets (default 24)
    """
    from datasets import load_dataset

    hf_path = ds_cfg.get("hf_path", KK_HF_PATH)
    split = str(ds_cfg.get("split", "test"))
    n_people_list = [int(n) for n in ds_cfg.get("n_people", [3, 4, 5])]
    max_examples = int(ds_cfg.get("max_examples", 24))
    if not n_people_list:
        return []
    per_bucket = max(1, max_examples // len(n_people_list))

    rows: list[dict] = []
    for n_people in n_people_list:
        ds = load_dataset(hf_path, data_files=_kk_data_file(split, n_people), split="train")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices[:per_bucket]:
            row = ds[idx]
            names = list(row["names"])
            solution = list(row["solution"])
            source_index = int(row.get("index", idx))
            gold = kk_gold_answer(names, solution)
            rows.append(
                {
                    "task_id": kk_task_id(n_people, source_index),
                    "domain": "logic",
                    "prompt": str(row["quiz"]).strip() + KK_PROMPT_SUFFIX,
                    "gold_answer": gold,
                    "eval_type": "logic",
                    "source_dataset": f"knights_knaves/people{n_people}",
                    "source_id": str(source_index),
                    "decoy_answer": kk_decoy_answer(names, solution, rng),
                    "n_people": n_people,
                }
            )
    return rows[:max_examples]
