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

from experiment_design import DesignAxes, resolve_design_axes


# Domains whose Stage-1 pool is built balanced across a stratum field. When such a
# domain is capped below its pool size, select_tasks draws evenly per stratum bucket
# instead of a global shuffle+truncate (which can unbalance the buckets).
STRATUM_FIELD: dict[str, str] = {}

# Logic coexists K&K + JustLogic: cap 50/50 by logic_source, then balance each
# half over logic_difficulty_tier (3 buckets per source).
HIERARCHICAL_STRATUM: dict[str, tuple[str, str]] = {
    "logic": ("logic_source", "logic_difficulty_tier"),
}


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


def _hierarchical_cap(
    shuffled_pool: list[dict],
    cap: int,
    primary_key: str,
    secondary_key: str,
) -> list[dict]:
    """Two-level stratified cap: split across primary buckets, then secondary."""
    by_primary: dict[Any, list[dict]] = {}
    for task in shuffled_pool:
        by_primary.setdefault(task[primary_key], []).append(task)

    primary_keys = sorted(by_primary.keys())
    if not primary_keys:
        return []

    base = cap // len(primary_keys)
    remainder = cap % len(primary_keys)
    selected: list[dict] = []
    for i, key in enumerate(primary_keys):
        allotment = base + (1 if i < remainder else 0)
        if allotment <= 0:
            continue
        sub_pool = by_primary[key]
        if allotment >= len(sub_pool):
            selected.extend(sub_pool)
        elif all(secondary_key in t for t in sub_pool):
            selected.extend(_stratified_cap(sub_pool, allotment, secondary_key))
        else:
            selected.extend(sub_pool[:allotment])
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
         logic uses hierarchical source/tier balance; others shuffle+truncate.
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
            hierarchical = HIERARCHICAL_STRATUM.get(domain)
            if cap is None:
                # No cap for this domain (e.g. domain absent from the dict).
                trimmed.extend(pool)
            elif (
                hierarchical is not None
                and int(cap) < len(pool)
                and all(hierarchical[0] in t and hierarchical[1] in t for t in pool)
            ):
                primary_key, secondary_key = hierarchical
                trimmed.extend(_hierarchical_cap(pool, int(cap), primary_key, secondary_key))
            elif (
                stratum_key is not None
                and int(cap) < len(pool)
                and all(stratum_key in t for t in pool)
            ):
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


def select_tasks_balanced(
    tasks: list[dict],
    scope: dict,
    experiment_seed: int,
    per_domain: int,
) -> list[dict]:
    """Select `per_domain` tasks from each scope domain (stratified cap when applicable)."""
    domains = scope.get("domains")
    if not domains:
        domains = sorted({t.get("domain", "") for t in tasks if t.get("domain")})
    balanced_scope = dict(scope)
    balanced_scope["max_per_domain"] = {str(d): int(per_domain) for d in domains}
    balanced_scope.pop("max_tasks_total", None)
    return select_tasks(tasks, balanced_scope, experiment_seed)


def parse_task_shard(spec: str | None) -> tuple[int, int] | None:
    """Parse a ``--task-shard I/N`` spec into a validated (index, count) pair.

    Returns None when ``spec`` is None/empty (no sharding). Index is 0-based and
    must satisfy ``0 <= I < N`` with ``N >= 1``. Raises ValueError on bad input.
    """
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec:
        return None
    if "/" not in spec:
        raise ValueError(f"--task-shard must be 'I/N' (got {spec!r})")
    i_str, n_str = spec.split("/", 1)
    try:
        shard_index = int(i_str)
        shard_count = int(n_str)
    except ValueError as exc:
        raise ValueError(f"--task-shard must be integers 'I/N' (got {spec!r})") from exc
    if shard_count < 1:
        raise ValueError(f"--task-shard N must be >= 1 (got {shard_count})")
    if not (0 <= shard_index < shard_count):
        raise ValueError(
            f"--task-shard index must satisfy 0 <= I < N (got I={shard_index}, N={shard_count})"
        )
    return shard_index, shard_count


def apply_task_shard(tasks: list[dict], shard: tuple[int, int] | None) -> list[dict]:
    """Partition ``tasks`` deterministically by interleaved stride ``tasks[I::N]``.

    Interleaving (not block slicing) spreads easy/hard and short/long tasks evenly
    across shards, which reduces straggler risk when shards run on separate GPUs.
    ``tasks`` is assumed pre-sorted (select_tasks sorts by task_id), so the union of
    all N shards is exactly the input and shards are pairwise disjoint.
    """
    if shard is None:
        return tasks
    shard_index, shard_count = shard
    return tasks[shard_index::shard_count]


def count_planned_dialogues(tasks: list[dict], lists: dict[str, list[Any]], *, include_neutral: bool) -> int:
    """Count adaptive dialogue runs for a task grid (not stop/recovery output rows)."""
    n_error = len(lists.get("error_types") or [])
    n_k = len(lists.get("k_injections") or [1])
    n_len = len(lists.get("length_levels") or ["short"])
    n_seeds = len(lists.get("seeds") or [42])
    per_task = n_error * n_k * n_len * n_seeds
    if include_neutral:
        per_task += n_k * n_len * n_seeds
    return len(tasks) * per_task


def design_lists(cfg: dict, scope: dict) -> dict[str, list[Any]]:
    """Resolve the experimental axes for a run: scope value, else design default."""
    return resolve_design_axes(cfg, scope).as_dict()


def count_planned_runs(tasks: list[dict], lists: dict[str, list[Any]] | DesignAxes) -> int:
    """Predict the number of model runs a scope will produce (cost preview).

    baseline runs  = tasks x seeds (no injection axes).
    perturbed runs = tasks x error_types x k x length x seeds x {stop,recovery}.
    """
    if isinstance(lists, DesignAxes):
        axes = lists
    else:
        axes = DesignAxes(
            variants=list(lists.get("variants") or []),
            error_types=list(lists.get("error_types") or []),
            k_injections=[int(k) for k in lists.get("k_injections") or [1]],
            length_levels=list(lists.get("length_levels") or ["short"]),
            seeds=[int(s) for s in lists.get("seeds") or [42]],
        )
    n = 0
    variants = axes.variants
    # Baseline variant: one clean run per task per seed.
    if "baseline" in variants:
        n += len(tasks) * len(axes.seeds)
    # Perturbed variants: full cross product of the injection design axes.
    perturbed = [v for v in variants if v in ("stop", "recovery")]
    if perturbed and axes.error_types:
        combos = (
            len(tasks)
            * len(axes.error_types)
            * len(axes.k_injections)
            * len(axes.length_levels)
            * len(axes.seeds)
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
