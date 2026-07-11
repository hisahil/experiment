#!/usr/bin/env bash
# OLMo GPU pod setup — run ON the Vast/DeployGPU instance (Jupyter Terminal or SSH).
#
# Usage (after repo is present):
#   cd ~/msc_thesis_experiment && bash scripts/olmo_pod_setup.sh
#
# Or let this script clone the public repo first:
#   bash olmo_pod_setup.sh
#
# Env overrides:
#   OLMO_REPO_DIR=~/msc_thesis_experiment
#   OLMO_REPO_URL=https://github.com/sahil-ejaz/experiment.git
#   SKIP_SETUP=1          # skip setup_gpu.sh if .venv already exists
#   FORCE_SETUP=1         # delete .venv and rerun setup_gpu.sh cross
#   SKIP_SMOKE=1          # skip 5-task smoke in tmux
set -euo pipefail

REPO="${OLMO_REPO_DIR:-$HOME/msc_thesis_experiment}"
REPO_URL="${OLMO_REPO_URL:-https://github.com/sahil-ejaz/experiment.git}"
MODEL="${OLMO_MODEL:-allenai/Olmo-3-7B-Think}"
SMOKE_LIMIT="${OLMO_SMOKE_LIMIT:-5}"

log() { echo "[olmo_pod_setup] $*"; }

if [[ ! -f "$REPO/scripts/setup_gpu.sh" ]]; then
  log "Repo not found at $REPO — cloning $REPO_URL ..."
  rm -rf "$REPO"
  git clone --depth 1 "$REPO_URL" "$REPO"
fi

cd "$REPO"
mkdir -p logs data/baselines/olmo-3-7b

if [[ "${FORCE_SETUP:-0}" == "1" && -d .venv ]]; then
  log "FORCE_SETUP=1 — removing .venv"
  rm -rf .venv
fi

if [[ ! -d .venv ]]; then
  if [[ "${SKIP_SETUP:-0}" == "1" ]]; then
    echo "ERROR: .venv missing and SKIP_SETUP=1" >&2
    exit 1
  fi
  log "Running setup_gpu.sh cross (~15-25 min first time)..."
  bash scripts/setup_gpu.sh cross
else
  log ".venv exists — activating (set FORCE_SETUP=1 to reinstall)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "OLMo preflight..."
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('${MODEL}', trust_remote_code=True)
print('Olmo3 config OK:', cfg.model_type, getattr(cfg, 'architectures', None))
"
python -c "from vllm import LLM; import vllm; print('vLLM', vllm.__version__, 'OK')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "${SKIP_SMOKE:-0}" == "1" ]]; then
  log "SKIP_SMOKE=1 — setup complete without smoke run"
  exit 0
fi

if tmux has-session -t olmo_smoke 2>/dev/null; then
  log "tmux session olmo_smoke already running — attach with: tmux attach -t olmo_smoke"
  exit 0
fi

log "Starting 5-task OLMo smoke in tmux (olmo_smoke)..."
tmux new-session -d -s olmo_smoke \
  "cd '$REPO' && source .venv/bin/activate && \
   python scripts/generate_baselines.py \
     --tasks data/smoke_tasks.jsonl \
     --model '${MODEL}' \
     --limit ${SMOKE_LIMIT} \
     --keep-all \
     2>&1 | tee logs/olmo_baseline_smoke_\$(date +%Y%m%d_%H%M).log; \
   echo DONE_OLMO_SMOKE"

log "Smoke started. Monitor: tmux attach -t olmo_smoke"
log "Log tail: tail -f logs/olmo_baseline_smoke_*.log"
