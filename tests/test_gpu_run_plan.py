"""Tests for gpu_run_plan helpers."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpu_run_plan import (  # noqa: E402
    BASELINE_CANDIDATE_LIMITS,
    build_total_extrapolation,
    resolve_selected_tasks,
    write_jsonl,
)


def test_baseline_candidate_limits_schlank():
    assert BASELINE_CANDIDATE_LIMITS == {"math": 10, "science": 15, "logic": 24}


def test_build_total_extrapolation():
    out = build_total_extrapolation(
        script1_elapsed_s=1000.0,
        script1_tasks_processed=49,
        script2_elapsed_s=500.0,
        script2_tasks_processed=15,
        script3_completed_dialogues=100,
        script3_compute_elapsed_s=10000.0,
        script3_planned_dialogues=1350,
        script4_elapsed_s=60.0,
        model_load_s=120.0,
        pilot_tasks=15,
        scope="pilot_gpu_full_factorial",
    )
    assert out["seconds_per_baseline_task"] is not None
    assert out["seconds_per_injection_task"] is not None
    assert out["extrapolated_full_thesis_total_h"] is not None
    assert out["extrapolated_full_thesis_total_cost_usd"] is not None
    assert out["planned_full_thesis_dialogues"] == 5400


def test_resolve_selected_tasks(tmp_path):
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    tasks = []
    for domain in ("math", "science", "logic"):
        for i in range(8):
            tasks.append({"task_id": f"{domain}_{i:02d}", "domain": domain})
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(tasks_path, tasks)

    baselines = []
    for domain in ("math", "science", "logic"):
        for i in range(8):
            baselines.append(
                {
                    "task_id": f"{domain}_{i:02d}",
                    "domain": domain,
                    "model": cfg["model"]["name"],
                    "verified": True,
                    "steps": ["s1", "s2", "s3"],
                }
            )
    baseline_path = tmp_path / "baseline_cots.jsonl"
    write_jsonl(baseline_path, baselines)

    selected, counts = resolve_selected_tasks(
        cfg=cfg,
        scope_name="pilot_gpu_full_factorial",
        tasks_path=tasks_path,
        baseline_path=baseline_path,
        model_name=cfg["model"]["name"],
        balanced_per_domain=5,
        experiment_seed=42,
    )
    assert len(selected) == 15
    assert counts["selected_by_domain"] == {"logic": 5, "math": 5, "science": 5}
