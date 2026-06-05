#!/usr/bin/env bash
# GPU environment setup — RunPod A100 80GB (CUDA 12.8 / 13.0, driver 570+)
# Usage: bash scripts/setup_gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Working directory: $ROOT"

if command -v nvidia-smi &>/dev/null; then
  echo "==> GPU status:"
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. Are you on a GPU instance?"
fi

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip --quiet

# ── PyTorch cu121 (CUDA 12.1 wheel; compatible with driver 570 / CUDA 13.0) ───
# Do NOT install latest torch+vllm from PyPI — vllm 0.22+ links libcudart.so.13
# while torch cu128 ships libcudart.so.12, causing ImportError at runtime.
echo "==> Installing PyTorch 2.5.1 (cu121)..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "==> Verifying CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA NOT available — stop here'
print(f'  torch {torch.__version__}')
print(f'  CUDA  {torch.version.cuda}')
print(f'  GPU   {torch.cuda.get_device_name(0)}')
free, total = torch.cuda.mem_get_info(0)
print(f'  VRAM  {free/1e9:.1f} GB free / {total/1e9:.1f} GB total')
"

# ── vLLM 0.6.6 (pinned; matches requirements.txt, cu121-compatible) ───────────
echo "==> Installing vLLM 0.6.6..."
pip uninstall vllm -y 2>/dev/null || true
pip install vllm==0.6.6 --quiet

echo "==> Verifying vLLM import..."
python -c "from vllm import LLM; print('  vLLM import OK')"

# ── Remaining deps ─────────────────────────────────────────────────────────────
echo "==> Installing remaining dependencies..."
pip install \
  pyyaml>=6.0 \
  python-dotenv>=1.0 \
  transformers>=4.44.0 \
  accelerate>=0.33.0 \
  huggingface-hub>=0.24.0 \
  numpy>=1.26.0 \
  pandas>=2.1.0 \
  openpyxl>=3.1.0 \
  scipy>=1.11.0 \
  datasets>=2.14.0 \
  sympy>=1.12 \
  anthropic>=0.40.0 \
  --quiet

# ── HuggingFace auth ───────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo ""
  echo "NOTE: DeepSeek-R1-Distill-Qwen-32B is public — no token required."
  echo "If you hit a rate-limit error, set: export HF_TOKEN=your_token_here"
else
  echo "==> Hugging Face token detected."
fi

mkdir -p outputs logs data

echo ""
echo "============================================================"
echo "Setup complete. Next: run the GPU pilot (see RUNBOOK below)"
echo ""
echo "  1. Build task pool:"
echo "     python scripts/prepare_tasks.py --config configs/datasets.yaml \\"
echo "       --output data/experiment_tasks_math500.jsonl"
echo ""
echo "  2. Generate baselines (Script 1):"
echo "     python scripts/generate_baselines.py"
echo ""
echo "  3. Generate injections (Script 2):"
echo "     python scripts/generate_injections.py --baselines data/baseline_cots.jsonl"
echo ""
echo "  4. Manual 1-task check:"
echo "     python scripts/run_perturbed.py --scope pilot_gpu --domain math --limit 1"
echo ""
echo "  5. Full pilot:"
echo "     python scripts/run_perturbed.py --scope pilot_gpu"
echo "     python scripts/run_perturbed.py --scope pilot_gpu_techcheck"
echo ""
echo "  6. Evaluation:"
echo "     python scripts/evaluate.py"
echo "============================================================"
