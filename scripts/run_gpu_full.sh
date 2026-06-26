#!/usr/bin/env bash
# Unified GPU full run — ONE command from zero to finish (resume-safe).
#
# Automatically:
#   - installs python3-venv if missing (Ubuntu)
#   - runs setup_gpu.sh if .venv missing or broken
#   - runs Script 1 → 2 → preflight → 3 → 4
#
# Usage (on GPU, after rsync from Mac):
#   bash scripts/run_gpu_full.sh
#
# Mac clean (--full) should already be done; this script does NOT auto-clean.
set -euo pipefail
cd "$(dirname "$0")/.."

SCOPE="${SCOPE:-pilot_gpu_full_factorial}"
BUDGET_HOURS="${BUDGET_HOURS:-0}"
BALANCED_PER_DOMAIN="${BALANCED_PER_DOMAIN:-5}"

venv_ready() {
  [[ -d .venv ]] && .venv/bin/python -c "from vllm import LLM" 2>/dev/null
}

ensure_python_venv_package() {
  if python3 -m venv --help >/dev/null 2>&1; then
    return 0
  fi
  echo "==> Installing python3-venv (required for .venv)..."
  export DEBIAN_FRONTEND=noninteractive
  export NEEDRESTART_MODE=a
  sudo apt-get update -qq
  sudo apt-get install -y python3.10-venv
}

if ! venv_ready; then
  ensure_python_venv_package
  echo "==> Setting up GPU environment (first run or broken .venv)..."
  bash scripts/setup_gpu.sh
fi

source .venv/bin/activate

python scripts/run_gpu_full.py \
  --scope "${SCOPE}" \
  --budget-hours "${BUDGET_HOURS}" \
  --balanced-per-domain "${BALANCED_PER_DOMAIN}" \
  "$@"

echo "---"
echo "Timings: results/run_timings.json"
