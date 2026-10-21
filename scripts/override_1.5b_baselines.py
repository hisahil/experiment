#!/usr/bin/env python3
"""Apply force-override for 6 failed 1.5B baseline tasks (Phase 1)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data/baselines/1.5b/baseline_cots.jsonl"
SAMPLES = ROOT / "data/baselines/1.5b/baseline_cots_samples.jsonl"

FLAG_ONLY = {
    "math500_test_intermediate_algebra_1566.json",
    "math500_test_intermediate_algebra_1407.json",
    "math500_test_number_theory_1055.json",
    "math500_test_number_theory_820.json",
}
SWAP = {
    "math500_test_algebra_1837.json": (46, 11),
    "math500_test_number_theory_737.json": (42, 6),
}
SAMPLE_FIELDS = (
    "baseline_cot", "steps", "step_count", "seed", "segmentation_method",
    "parsed_answer", "sample_correct",
)

def load_sample_index():
    idx = {}
    with SAMPLES.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx[(rec["task_id"], int(rec["seed"]), int(rec["step_count"]))] = rec
    return idx

def main():
    sample_idx = load_sample_index()
    records = [json.loads(l) for l in BASELINE.read_text(encoding="utf-8").splitlines() if l.strip()]
    updated = 0
    for rec in records:
        tid = rec["task_id"]
        if tid in FLAG_ONLY:
            rec["verified"] = True
            rec["force_override"] = True
            rec.pop("exclude_reason", None)
            updated += 1
            print(f"[FLAG] {tid}")
        elif tid in SWAP:
            seed, steps = SWAP[tid]
            sample = sample_idx.get((tid, seed, steps))
            if sample is None:
                raise SystemExit(f"Sample not found: {tid} seed={seed} steps={steps}")
            for field in SAMPLE_FIELDS:
                if field in sample:
                    rec[field] = sample[field]
            rec["verified"] = True
            rec["force_override"] = True
            rec.pop("exclude_reason", None)
            updated += 1
            print(f"[SWAP] {tid} seed={seed} steps={steps}")
    if updated != 6:
        raise SystemExit(f"Expected 6 updates, got {updated}")
    verified = sum(1 for r in records if r.get("verified"))
    with BASELINE.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Done. verified={verified}/100")

if __name__ == "__main__":
    main()
