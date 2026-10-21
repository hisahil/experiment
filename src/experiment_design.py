"""Central factorial design constants and grid key helpers.

Python is the source of truth for default design axes; configs/experiment.yaml
scopes may override individual lists. A sync test keeps the YAML `design:` block
aligned with these defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Default full-factorial axes (thesis Exp 1 target design).
VARIANTS: list[str] = ["baseline", "stop", "recovery"]
ERROR_TYPES: list[str] = ["off_topic", "paraphrased_shift", "false_consensus"]
K_INJECTIONS: list[int] = [1, 2, 3]
LENGTH_LEVELS: list[str] = ["short", "long"]
DEFAULT_SEEDS: list[int] = [42, 43, 44, 45, 46]

DEFAULT_DESIGN: dict[str, list[Any]] = {
    "variants": VARIANTS,
    "error_types": ERROR_TYPES,
    "k_injections": K_INJECTIONS,
    "length_levels": LENGTH_LEVELS,
    "seeds": DEFAULT_SEEDS,
}

# script3.dialogue_mode values.
DIALOGUE_MODE_ADAPTIVE = "adaptive_alternating"
DIALOGUE_MODE_THREE_TURN = "three_turn"

# script3.dialogue_generation_mode values.
#   restart      = legacy multi-turn dialogue; the model restarts its think block
#                  after each injection (build_dialogue_messages + generate_chat).
#   continuation = Recum-style open think block; the model continues its OWN trace
#                  (build_continuation_prompt + raw completion). Deletion model:
#                  baseline prefix to midpoint, suffix deleted, injection 1 at the
#                  midpoint, injections 2..k after one model-generated paragraph each.
GENERATION_MODE_RESTART = "restart"
GENERATION_MODE_CONTINUATION = "continuation"
GENERATION_MODES = [GENERATION_MODE_RESTART, GENERATION_MODE_CONTINUATION]


def expected_pregenerated_steps(
    k: int,
    *,
    pregenerate_steps: int | str = "auto",
    dialogue_mode: str = DIALOGUE_MODE_ADAPTIVE,
) -> int:
    """How many injection steps Script 2 should pre-generate per cell.

    In adaptive_alternating dialogues (Script 3b), only injection_steps[0] is used
    as the round-1 injection; rounds 2..k are generated live. So a single
    pre-generated step is sufficient. Legacy three_turn (run_perturbed.py) injects
    all k steps bundled in Turn 3 and therefore needs all k.

    pregenerate_steps:
      - "auto": 1 when dialogue_mode == adaptive_alternating, else k.
      - "1" / 1:  force a single pre-generated step.
      - "k":      force all k steps (legacy bundled behaviour).
      - other int: clamped to [1, k].
    """
    k = int(k)
    if isinstance(pregenerate_steps, str):
        val = pregenerate_steps.strip().lower()
        if val in ("", "auto"):
            return 1 if dialogue_mode == DIALOGUE_MODE_ADAPTIVE else k
        if val == "k":
            return k
        n = int(val)
    else:
        n = int(pregenerate_steps)
    return max(1, min(n, k))


@dataclass(frozen=True)
class DesignAxes:
    variants: list[str]
    error_types: list[str]
    k_injections: list[int]
    length_levels: list[str]
    seeds: list[int]

    def as_dict(self) -> dict[str, list[Any]]:
        return {
            "variants": list(self.variants),
            "error_types": list(self.error_types),
            "k_injections": list(self.k_injections),
            "length_levels": list(self.length_levels),
            "seeds": list(self.seeds),
        }


def resolve_design_axes(cfg: dict, scope: dict) -> DesignAxes:
    """Resolve experimental axes: scope override, else YAML design, else Python defaults."""
    design = cfg.get("design", {})
    return DesignAxes(
        variants=list(scope.get("variants") or design.get("variants") or VARIANTS),
        error_types=list(scope.get("error_types") or design.get("error_types") or ERROR_TYPES),
        k_injections=[
            int(k) for k in (scope.get("k_injections") or design.get("k_injections") or K_INJECTIONS)
        ],
        length_levels=list(
            scope.get("length_levels") or design.get("length_levels") or LENGTH_LEVELS
        ),
        seeds=[int(s) for s in (scope.get("seeds") or design.get("seeds") or DEFAULT_SEEDS)],
    )


def injection_cell_key(task_id: str, error_type: str, k: int, length_level: str) -> str:
    """Lookup key for injections_generated.jsonl (matches Script 2 dedup minus target_model)."""
    return f"{task_id}|{error_type}|{k}|{length_level}"


def run_cell_key(
    task_id: str,
    error_type: str,
    k: int,
    length_level: str,
    seed: int,
    variant: str,
    model: str,
) -> tuple[str, str, int, str, int, str, str]:
    """Unique key for one post_injection_runs.jsonl row."""
    return (task_id, error_type, int(k), length_level, int(seed), variant, model)


def expected_injection_keys(tasks: list[dict], axes: DesignAxes) -> set[str]:
    """All injection keys implied by tasks × error_type × k × length."""
    keys: set[str] = set()
    for task in tasks:
        for error_type in axes.error_types:
            for k in axes.k_injections:
                for length_level in axes.length_levels:
                    keys.add(injection_cell_key(task["task_id"], error_type, k, length_level))
    return keys


def collect_required_injection_keys(tasks: list[dict], lists: dict[str, list[Any]]) -> list[str]:
    """All injection keys implied by the selected tasks and scope design axes."""
    axes = DesignAxes(
        variants=list(lists.get("variants") or VARIANTS),
        error_types=list(lists["error_types"]),
        k_injections=[int(k) for k in lists["k_injections"]],
        length_levels=list(lists["length_levels"]),
        seeds=[int(s) for s in lists["seeds"]],
    )
    return sorted(expected_injection_keys(tasks, axes))


def expected_run_keys(
    tasks: list[dict],
    axes: DesignAxes,
    *,
    model: str,
    variants: list[str] | None = None,
) -> set[tuple[str, str, int, str, int, str, str]]:
    """All run keys for stop/recovery variants over the factorial design."""
    perturbed = [v for v in (variants or axes.variants) if v in ("stop", "recovery")]
    keys: set[tuple[str, str, int, str, int, str, str]] = set()
    for task in tasks:
        for error_type in axes.error_types:
            for k in axes.k_injections:
                for length_level in axes.length_levels:
                    for seed in axes.seeds:
                        for variant in perturbed:
                            keys.add(
                                run_cell_key(
                                    task["task_id"],
                                    error_type,
                                    k,
                                    length_level,
                                    seed,
                                    variant,
                                    model,
                                )
                            )
    return keys


def expected_baseline_keys(
    tasks: list[dict],
    *,
    model: str,
) -> set[tuple[str, str]]:
    """Verified baseline keys: one row per (task_id, model)."""
    return {(task["task_id"], model) for task in tasks}
