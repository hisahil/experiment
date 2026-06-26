#!/usr/bin/env bash
# HyperStack GPU pilot — one command after setup_gpu.sh
#
# Usage:
#   bash scripts/run_gpu_pilot.sh
#   bash scripts/run_gpu_pilot.sh --scope pilot_gpu_full_factorial --budget-hours 7.0 --balanced-per-domain 5
set -euo pipefail
cd "$(dirname "$0")/.."

SCOPE="${SCOPE:-pilot_gpu_full_factorial}"
BUDGET_HOURS="${BUDGET_HOURS:-7.0}"
BALANCED_PER_DOMAIN="${BALANCED_PER_DOMAIN:-5}"

if [[ ! -d .venv ]]; then
  echo "Missing .venv — run: bash scripts/setup_gpu.sh" >&2
  exit 1
fi
source .venv/bin/activate

python scripts/run_pipeline.py \
  --scope "${SCOPE}" \
  --script3 adaptive \
  --from-stage 3 \
  --record-timings \
  --budget-hours "${BUDGET_HOURS}" \
  --balanced-per-domain "${BALANCED_PER_DOMAIN}"

echo "---"
echo "Timings:     results/run_timings.json"
echo "Dialogues:   data/complete_dialogues.jsonl"
echo "REI outputs: results/"
