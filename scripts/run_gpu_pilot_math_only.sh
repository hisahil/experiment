#!/usr/bin/env bash
# Fallback timing run: math-only (5 tasks) — use if science/logic baselines not ready.
# Still uses full factorial axes (k, length, seeds).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python scripts/run_pipeline.py \
  --scope pilot_gpu_full_factorial \
  --script3 adaptive \
  --from-stage 3 \
  --record-timings \
  --budget-hours "${BUDGET_HOURS:-7.0}" \
  --balanced-per-domain 5 \
  --domain math
