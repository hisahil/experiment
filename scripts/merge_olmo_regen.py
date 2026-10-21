#!/usr/bin/env python3
"""Merge RunPod regeneration batch into OLMo production baselines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reprocess_olmo_baselines import (  # noqa: E402
    build_baseline_record,
    load_jsonl,
    pick_baseline,
    write_jsonl_atomic,
)


def sample_detail_path(baseline_path: Path) -> Path:
    stem = baseline_path.stem
    if stem.endswith("_samples"):
        return baseline_path
    return baseline_path.with_name(stem.replace("baseline_cots", "baseline_cots_samples"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regen-baselines", type=Path, required=True)
    parser.add_argument("--regen-samples", type=Path, default=None)
    parser.add_argument(
        "--production-baselines",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots.jsonl",
    )
    parser.add_argument(
        "--production-samples",
        type=Path,
        default=ROOT / "data/baselines/olmo-3-7b/baseline_cots_samples.jsonl",
    )
    parser.add_argument("--regen-base-seed", type=int, default=200)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    regen_samples = args.regen_samples or sample_detail_path(args.regen_baselines)
    majority = args.n_samples // 2 + 1

    regen_by_task: dict[str, list[dict]] = {}
    for row in load_jsonl(regen_samples):
        row = dict(row)
        row["regeneration_batch"] = args.regen_base_seed
        row["active_for_selection"] = True
        regen_by_task.setdefault(row["task_id"], []).append(row)

    prod_samples = load_jsonl(args.production_samples)
    prod_baselines = {r["task_id"]: r for r in load_jsonl(args.production_baselines)}
    regen_baselines = {r["task_id"]: r for r in load_jsonl(args.regen_baselines)}

    merged_samples: list[dict] = []
    merged_baselines: list[dict] = []
    order = list(prod_baselines)

    for tid in order:
        if tid in regen_by_task:
            rows = sorted(regen_by_task[tid], key=lambda s: (s.get("attempt") or 0, s.get("seed") or 0))
            for s in rows:
                s["supersedes_seed_batch"] = "42-46"
            chosen, is_v, excl, degen = pick_baseline(rows, majority, args.min_steps)
            old = prod_baselines[tid]
            rec = build_baseline_record(
                task_id=tid,
                old_record=old,
                samples=rows,
                chosen=chosen,
                verified=is_v,
                exclude_reason=excl,
                degenerate_fallback=degen,
                n_samples=len(rows),
                selection_tier=f"semantic_regeneration_seed{args.regen_base_seed}",
            )
            rec["regenerated"] = True
            rec["regen_base_seed"] = args.regen_base_seed
            rec["needs_regeneration"] = False
            merged_baselines.append(rec)
            merged_samples.extend(rows)
            print(f"merged {tid} verified={is_v} seed={rec.get('seed')} solve_rate={rec.get('solve_rate')}")
        else:
            merged_baselines.append(prod_baselines[tid])
            merged_samples.extend([s for s in prod_samples if s["task_id"] == tid])

    write_jsonl_atomic(args.production_baselines, merged_baselines)
    write_jsonl_atomic(args.production_samples, merged_samples)
    print(f"wrote {len(merged_baselines)} baselines, {len(merged_samples)} samples")


if __name__ == "__main__":
    main()
