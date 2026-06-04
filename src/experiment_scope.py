"""Resolve active scope and filter task pool."""

from __future__ import annotations

import random
from typing import Any


def get_scope(cfg: dict, scope_name: str | None = None) -> dict:
    name = scope_name or cfg.get("active_scope", "pilot")
    scopes = cfg.get("scopes", {})
    if name not in scopes:
        raise ValueError(f"Unknown scope '{name}'. Available: {list(scopes.keys())}")
    scope = dict(scopes[name])
    scope["name"] = name
    return scope


def select_tasks(tasks: list[dict], scope: dict, experiment_seed: int) -> list[dict]:
    domains = scope.get("domains")
    filtered = [t for t in tasks if not domains or t.get("domain") in domains]

    max_per_domain = scope.get("max_per_domain")
    if max_per_domain is not None:
        by_domain: dict[str, list[dict]] = {}
        for task in filtered:
            by_domain.setdefault(task["domain"], []).append(task)
        rng = random.Random(experiment_seed)
        trimmed: list[dict] = []
        for domain in sorted(by_domain.keys()):
            pool = by_domain[domain]
            rng.shuffle(pool)
            # max_per_domain may be a single int (all domains) or a dict of
            # per-domain caps (e.g. {math: 20, science: 8, logic: 8}); a missing
            # domain key in the dict means "no cap" for that domain.
            if isinstance(max_per_domain, dict):
                cap = max_per_domain.get(domain)
            else:
                cap = max_per_domain
            trimmed.extend(pool if cap is None else pool[: int(cap)])
        filtered = trimmed

    max_total = scope.get("max_tasks_total")
    if max_total is not None and len(filtered) > int(max_total):
        rng = random.Random(experiment_seed)
        filtered = sorted(filtered, key=lambda t: t["task_id"])
        rng.shuffle(filtered)
        filtered = filtered[: int(max_total)]

    return sorted(filtered, key=lambda t: t["task_id"])


def design_lists(cfg: dict, scope: dict) -> dict[str, list[Any]]:
    design = cfg.get("design", {})
    return {
        "variants": scope.get("variants") or design.get("variants", ["baseline"]),
        "error_types": scope.get("error_types") or design.get("error_types", []),
        "k_injections": scope.get("k_injections") or design.get("k_injections", [1]),
        "length_levels": scope.get("length_levels") or design.get("length_levels", ["short"]),
        "seeds": scope.get("seeds") or design.get("seeds", [42]),
    }


def count_planned_runs(tasks: list[dict], lists: dict[str, list[Any]]) -> int:
    n = 0
    variants = lists["variants"]
    if "baseline" in variants:
        n += len(tasks) * len(lists["seeds"])
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
    total = count_planned_runs(tasks, lists)
    print(f"Scope: {scope.get('name')} — {scope.get('description', '')}")
    print(f"Tasks selected: {len(tasks)}")
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
