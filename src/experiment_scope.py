"""Resolve the active experiment scope and select the task subset to run.

A "scope" (pilot, pilot_gpu, full, ...) is defined in configs/experiment.yaml and
controls which tasks and which design axes a run uses. This module:
  get_scope()         — fetch a scope block by name.
  select_tasks()      — apply domain filter + per-domain / total caps to the pool.
  design_lists()      — resolve the experimental axes (variants, k, seeds, ...).
  count_planned_runs()/print_run_plan() — preview run size before launching.

All randomness uses a fixed experiment_seed, so selection is reproducible.
"""

from __future__ import annotations

import random
from typing import Any


# Domains whose Stage-1 pool is built balanced across a stratum field. When such a
# domain is capped below its pool size, select_tasks draws evenly per stratum bucket
# instead of a global shuffle+truncate (which can unbalance the buckets).
# Example: logic puzzles are balanced over n_people (3/4/5 characters).
STRATUM_FIELD = {"logic": "n_people"}


def _stratified_cap(shuffled_pool: list[dict], cap: int, stratum_key: str) -> list[dict]:
    """Pick `cap` items balanced across stratum buckets from an already-shuffled pool.

    Within-bucket order is inherited from `shuffled_pool` (so randomness comes from the
    caller's single rng.shuffle, keeping shared-rng consumption identical to the plain
    path). Takes cap // n_buckets per bucket, then distributes the remainder round-robin
    in sorted bucket order, so the result is balanced and deterministic.

    Example: pool 3/3/3 over n_people {3,4,5}, cap=4 -> base=1 each (3 picked),
    remainder 1 goes to the first bucket -> 2/1/1.
    """
    # Group the (already shuffled) pool by stratum value, e.g. n_people 3/4/5.
    buckets: dict[Any, list[dict]] = {}
    for task in shuffled_pool:
        buckets.setdefault(task[stratum_key], []).append(task)

    # Equal share first: cap // n_buckets from each bucket (in shuffled order).
    keys = sorted(buckets.keys())
    base = cap // len(keys)
    selected: list[dict] = []
    idx: dict[Any, int] = {}  # next unused position per bucket
    for key in keys:
        selected.extend(buckets[key][:base])
        idx[key] = min(base, len(buckets[key]))

    # Distribute the leftover (cap not divisible by n_buckets) round-robin.
    remaining = cap - len(selected)
    while remaining > 0:
        progressed = False
        for key in keys:
            if remaining == 0:
                break
            if idx[key] < len(buckets[key]):  # this bucket still has items
                selected.append(buckets[key][idx[key]])
                idx[key] += 1
                remaining -= 1
                progressed = True
        # Safety brake: if a full pass added nothing, all buckets are empty
        # (cap > pool). Stop instead of looping forever; caller guards this too.
        if not progressed:
            break
    return selected


def get_scope(cfg: dict, scope_name: str | None = None) -> dict:
    """Return the scope config block by name (a copy, with `name` attached).

    Resolution order: explicit scope_name > cfg['active_scope'] > 'pilot'.
    Raises ValueError listing valid scopes if the name is unknown.
    """
    name = scope_name or cfg.get("active_scope", "pilot")
    scopes = cfg.get("scopes", {})
    if name not in scopes:
        raise ValueError(f"Unknown scope '{name}'. Available: {list(scopes.keys())}")
    scope = dict(scopes[name])  # copy so callers can mutate without touching cfg
    scope["name"] = name
    return scope


def select_tasks(tasks: list[dict], scope: dict, experiment_seed: int) -> list[dict]:
    """Filter the task pool down to the tasks this scope should run.

    Three stages, all seeded for reproducibility:
      1. domain filter — keep only scope['domains'].
      2. per-domain cap — max_per_domain (int for all, or dict per domain);
         logic is drawn balanced over n_people, others shuffle+truncate.
      3. total cap — max_tasks_total trims the combined set (no domain balance).
    Output is sorted by task_id for stable ordering downstream.
    """
    # Stage 1: domain filter (no domains key -> keep everything).
    domains = scope.get("domains")
    filtered = [t for t in tasks if not domains or t.get("domain") in domains]

    # Stage 2: per-domain caps.
    max_per_domain = scope.get("max_per_domain")
    if max_per_domain is not None:
        # Group surviving tasks by domain so each can be capped independently.
        by_domain: dict[str, list[dict]] = {}
        for task in filtered:
            by_domain.setdefault(task["domain"], []).append(task)
        rng = random.Random(experiment_seed)
        trimmed: list[dict] = []
        for domain in sorted(by_domain.keys()):  # sorted -> deterministic rng use
            pool = by_domain[domain]
            # Keep exactly one rng.shuffle(pool) per domain (unchanged shared-rng
            # consumption) so non-stratified domains are selected bit-identically.
            rng.shuffle(pool)
            # max_per_domain may be a single int (all domains) or a dict of
            # per-domain caps (e.g. {math: 20, science: 8, logic: 8}); a missing
            # domain key in the dict means "no cap" for that domain.
            if isinstance(max_per_domain, dict):
                cap = max_per_domain.get(domain)
            else:
                cap = max_per_domain
            stratum_key = STRATUM_FIELD.get(domain)
            if cap is None:
                # No cap for this domain (e.g. domain absent from the dict).
                trimmed.extend(pool)
            elif (
                stratum_key is not None
                and int(cap) < len(pool)
                and all(stratum_key in t for t in pool)
            ):
                # Logic capped below pool size: draw balanced per n_people bucket.
                trimmed.extend(_stratified_cap(pool, int(cap), stratum_key))
            else:
                # Math/science, or cap >= pool (full run): take first `cap` of shuffle.
                trimmed.extend(pool[: int(cap)])
        filtered = trimmed

    # Stage 3: global cap across all domains (rarely used; null in pilot). This can
    # break per-domain balance, so prefer max_per_domain when balance matters.
    max_total = scope.get("max_tasks_total")
    if max_total is not None and len(filtered) > int(max_total):
        rng = random.Random(experiment_seed)
        filtered = sorted(filtered, key=lambda t: t["task_id"])  # stable pre-shuffle
        rng.shuffle(filtered)
        filtered = filtered[: int(max_total)]

    # Stable output order regardless of how shuffling fell out.
    return sorted(filtered, key=lambda t: t["task_id"])


def design_lists(cfg: dict, scope: dict) -> dict[str, list[Any]]:
    """Resolve the experimental axes for a run: scope value, else design default.

    `or` means an empty/missing scope list falls back to cfg['design'] (then to a
    hardcoded default). These lists drive the full factorial in Scripts 3/4.
    """
    design = cfg.get("design", {})
    return {
        "variants": scope.get("variants") or design.get("variants", ["baseline"]),
        "error_types": scope.get("error_types") or design.get("error_types", []),
        "k_injections": scope.get("k_injections") or design.get("k_injections", [1]),
        "length_levels": scope.get("length_levels") or design.get("length_levels", ["short"]),
        "seeds": scope.get("seeds") or design.get("seeds", [42]),
    }


def count_planned_runs(tasks: list[dict], lists: dict[str, list[Any]]) -> int:
    """Predict the number of model runs a scope will produce (cost preview).

    baseline runs  = tasks x seeds (no injection axes).
    perturbed runs = tasks x error_types x k x length x seeds x {stop,recovery}.
    """
    n = 0
    variants = lists["variants"]
    # Baseline variant: one clean run per task per seed.
    if "baseline" in variants:
        n += len(tasks) * len(lists["seeds"])
    # Perturbed variants: full cross product of the injection design axes.
    perturbed = [v for v in variants if v in ("stop", "recovery")]
    if perturbed and lists["error_types"]:
        combos = (
            len(tasks)
            * len(lists["error_types"])
            * len(lists["k_injections"])
            * len(lists["length_levels"])
            * len(lists["seeds"])
        )
        n += combos * len(perturbed)
    return n


def print_run_plan(
    scope: dict,
    tasks: list[dict],
    lists: dict[str, list[Any]],
    runs_file: str,
) -> None:
    """Print a human-readable summary of the run before launching (no side effects)."""
    total = count_planned_runs(tasks, lists)
    print(f"Scope: {scope.get('name')} — {scope.get('description', '')}")
    print(f"Tasks selected: {len(tasks)}")
    # Per-domain counts so the balance is visible at a glance.
    by_domain: dict[str, int] = {}
    for t in tasks:
        by_domain[t["domain"]] = by_domain.get(t["domain"], 0) + 1
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count}")
    print(f"Variants: {lists['variants']}")
    print(f"Error types: {lists['error_types']}")
    print(f"k: {lists['k_injections']} | length: {lists['length_levels']} | seeds: {lists['seeds']}")
    print(f"Planned runs: {total}")
    print(f"Runs file: {runs_file}")
