#!/usr/bin/env bash
# Legacy: baseline prep only. Prefer: bash scripts/run_gpu_full.sh
#
# Resume-safe — does NOT auto-clean (Mac clean or explicit clean_for_gpu_run.sh --full).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

TASKS=data/experiment_tasks_pilot_gpu.jsonl
SCOPE=pilot_gpu_full_factorial
SELECTED=data/selected_pilot_tasks.jsonl

echo "==> Script 1: baselines (math 10 / science 15 / logic 24)"
for spec in "math:10" "science:15" "logic:24"; do
  domain="${spec%%:*}"
  limit="${spec##*:}"
  python scripts/generate_baselines.py \
    --tasks "$TASKS" \
    --domain "$domain" \
    --limit "$limit"
done

echo "==> Select tasks + Script 2 (selected-only)"
python - <<'PY'
import sys
from pathlib import Path
import yaml

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
from gpu_run_plan import resolve_selected_tasks, write_jsonl

cfg = yaml.safe_load(open("configs/experiment.yaml"))
model = cfg["model"]["name"]
selected, counts = resolve_selected_tasks(
    cfg=cfg,
    scope_name="pilot_gpu_full_factorial",
    tasks_path=Path("data/experiment_tasks_pilot_gpu.jsonl"),
    baseline_path=Path("data/baseline_cots.jsonl"),
    model_name=model,
    balanced_per_domain=5,
    experiment_seed=int(cfg.get("experiment", {}).get("seed", 42)),
)
write_jsonl(Path("data/selected_pilot_tasks.jsonl"), selected)
print("Selected:", counts)
PY

python scripts/generate_injections.py \
  --tasks "$SELECTED" \
  --baselines data/baseline_cots.jsonl \
  --output data/injections_generated.jsonl \
  --k-values 1 3 5 \
  --error-types incorrect_reasoning paraphrased_shift off_topic \
  --length-levels short long

echo "==> Preflight"
python scripts/preflight_gpu.py --scope "$SCOPE" --balanced-per-domain 5

echo ""
echo "If preflight passes, run: bash scripts/run_gpu_pilot.sh"
