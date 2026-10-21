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

install_cross_deps_inplace() {
  log "Installing cross-model deps (vLLM 0.24+cu129, transformers git, …)..."
  pip install --upgrade pip --quiet
  # PyPI vllm==0.24.0 pulls CUDA-13 wheels; Vast images often have CUDA 12.8 only.
  local vllm_wheel="${VLLM_CU129_WHEEL:-https://wheels.vllm.ai/ee0da84ab9e04ac7610e28580af62c365e898389/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl}"
  if ! python -c "from vllm import LLM" 2>/dev/null; then
    pip uninstall -y vllm 2>/dev/null || true
    pip install "$vllm_wheel" --quiet
  fi
  pip install --upgrade "mistral_common>=1.8.6" --quiet
  pip install --upgrade "git+https://github.com/huggingface/transformers.git" --quiet
  pip install --upgrade "tokenizers==0.22.2" --quiet
  pip install \
    pyyaml>=6.0 python-dotenv>=1.0 accelerate>=0.33.0 huggingface-hub>=0.24.0 \
    numpy>=1.26.0 pandas>=2.1.0 openpyxl>=3.1.0 scipy>=1.11.0 datasets>=2.14.0 \
    sympy>=1.12 anthropic>=0.40.0 \
    --quiet
  python -c "
import torch
assert torch.cuda.is_available(), 'CUDA NOT available after cross deps install'
print('  torch', torch.__version__, '| CUDA', torch.version.cuda)
"
  python -c "from vllm import LLM; import vllm; print('  vLLM', vllm.__version__, 'import OK')"
}

try_vast_image_venv() {
  local iv="${OLMO_IMAGE_VENV:-/venv/main}"
  [[ -f "$iv/bin/activate" ]] || return 1
  if ! "$iv/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    return 1
  fi
  log "Vast image venv at $iv has CUDA — using it (skip fresh setup_gpu.sh venv)"
  rm -rf .venv
  ln -sfn "$(readlink -f "$iv" 2>/dev/null || echo "$iv")" .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  install_cross_deps_inplace
  return 0
}

venv_cuda_ok() {
  [[ -f .venv/bin/python ]] && .venv/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null
}

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

if [[ -d .venv ]] && ! venv_cuda_ok; then
  log "Existing .venv has no CUDA — replacing with image venv or fresh setup"
  rm -rf .venv
fi

if [[ ! -d .venv ]]; then
  if [[ "${SKIP_SETUP:-0}" == "1" ]]; then
    echo "ERROR: .venv missing and SKIP_SETUP=1" >&2
    exit 1
  fi
  if try_vast_image_venv; then
    :
  else
    log "Running setup_gpu.sh cross (~15-25 min first time)..."
    bash scripts/setup_gpu.sh cross
  fi
else
  log ".venv exists with CUDA — activating (set FORCE_SETUP=1 to reinstall)"
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
     2>&1 | tee logs/olmo_baseline_smoke_\$(date +%Y%m%d_%H%M).log; \
   echo DONE_OLMO_SMOKE"

log "Smoke started. Monitor: tmux attach -t olmo_smoke"
log "Log tail: tail -f logs/olmo_baseline_smoke_*.log"
