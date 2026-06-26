#!/usr/bin/env bash
# Remove generated pipeline outputs so the next GPU run starts from scratch.
#
# Usage:
#   bash scripts/clean_for_gpu_run.sh          # Stage 3+4 outputs + results/
#   bash scripts/clean_for_gpu_run.sh --full   # Also baseline_cots + injections (Script 1+2 regen)
set -euo pipefail
cd "$(dirname "$0")/.."

FULL=false
if [[ "${1:-}" == "--full" ]]; then
  FULL=true
fi

remove_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    rm -f "$path"
    echo "  removed $path"
  fi
}

echo "==> Cleaning generated outputs (full=$FULL)"

# Stage 3 outputs
remove_file data/post_injection_runs.jsonl
remove_file data/complete_dialogues.jsonl
remove_file data/adaptive_dialogues.jsonl

# Stage 1+2 outputs (only with --full)
if $FULL; then
  remove_file data/baseline_cots.jsonl
  remove_file data/baseline_cots_samples.jsonl
  remove_file data/injections_generated.jsonl
  if [[ -d data/judge_raw ]]; then
    rm -rf data/judge_raw
    echo "  removed data/judge_raw/"
  fi
fi

# Stage 4 outputs (keep .gitkeep)
if [[ -d results ]]; then
  find results -maxdepth 1 -type f ! -name '.gitkeep' -delete
  echo "  cleared results/ (kept .gitkeep)"
fi

# Old backup files from previous pilot runs
for bak in data/*.bak.*; do
  [[ -e "$bak" ]] || continue
  rm -f "$bak"
  echo "  removed $bak"
done

echo ""
echo "Kept (inputs, not generated):"
echo "  data/experiment_tasks_pilot_gpu.jsonl"
echo "  data/*_synthetic.jsonl (local tests)"
if ! $FULL; then
  echo "  data/baseline_cots.jsonl"
  echo "  data/injections_generated.jsonl"
fi
echo ""
if $FULL; then
  echo "Next on GPU: bash scripts/run_gpu_full.sh"
else
  echo "Next on GPU: bash scripts/run_gpu_pilot.sh (baselines/injections unchanged)"
fi
