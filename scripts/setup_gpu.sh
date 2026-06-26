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

# ── Virtual environment (always fresh to avoid mixed vllm/torch versions) ───────
echo "==> Creating fresh virtual environment..."
rm -rf .venv
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip --quiet

# ── PyTorch cu121 (CUDA 12.1 wheel; compatible with driver 570 / CUDA 13.0) ───
# Do NOT install latest torch+vllm from PyPI — vllm 0.22+ links libcudart.so.13
# while torch cu128 ships libcudart.so.12, causing ImportError at runtime.
echo "==> Installing PyTorch 2.5.1 (cu121)..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --quiet

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

# ── vLLM 0.6.6 + pinned transformers (avoid ProcessorMixin / float8 conflicts) ─
echo "==> Installing vLLM 0.6.6 + pinned deps..."
pip install vllm==0.6.6 --quiet
pip install "transformers>=4.44.0,<4.46" --quiet

echo "==> Verifying vLLM import..."
python -c "from vllm import LLM; print('  vLLM import OK')"

# ── Remaining deps ─────────────────────────────────────────────────────────────
echo "==> Installing remaining dependencies..."
pip install \
  pyyaml>=6.0 \
  python-dotenv>=1.0 \
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
echo "Setup complete. Next: GPU pipeline (Scripts 0–4)"
echo ""
echo "  Recommended — one command (stages 0–4 + grid validation):"
echo "     python scripts/run_pipeline.py --scope pilot_gpu --dry-run"
echo "     python scripts/run_pipeline.py --scope pilot_gpu"
echo "     python scripts/validate_grid.py --scope pilot_gpu --kind all"
echo ""
echo "  Manual stages (fallback):"
echo ""
echo "  0. (Pilot only) Pin the exact task subset for Script 1:"
echo "     python scripts/select_pilot_tasks.py \\"
echo "       --scope pilot_gpu \\"
echo "       --tasks data/experiment_tasks_pilot_gpu.jsonl \\"
echo "       --output data/pilot_gpu_tasks.jsonl"
echo ""
echo "  1. Build task pool (if not done yet):"
echo "     python scripts/prepare_tasks.py --config configs/datasets.yaml \\"
echo "       --output data/experiment_tasks_pilot_gpu.jsonl"
echo ""
echo "  2. Generate baselines (Script 1):"
echo "     python scripts/generate_baselines.py \\"
echo "       --tasks data/pilot_gpu_tasks.jsonl    # pilot"
echo "     # full run: omit --tasks or point at data/experiment_tasks.jsonl"
echo ""
echo "  3. Generate injections (Script 2) — must cover the scope design:"
echo "     python scripts/generate_injections.py --baselines data/baseline_cots.jsonl"
echo "     # pilot_gpu needs k=1,3 × short; add k=5 × long before techcheck scope"
echo "     # Re-run Script 2 if Script 1 baselines change (avoid insertion_step mismatch)."
echo ""
echo "  3. Local preflight (Mac, before deploy):"
echo "     python scripts/preflight_gpu.py"
echo ""
echo "  4. Transfer to GPU:"
echo "     GPU_HOST=user@IP bash scripts/transfer_to_gpu.sh.example"
echo ""
echo "  5. Manual 1-task check (dry-run, no GPU cost for preflight logic):"
echo "     python scripts/run_perturbed.py --dry-run --scope pilot_gpu --domain math --limit 1"
echo ""
echo "  5. Script 3 (GPU) — preflight runs before inference:"
echo "     python scripts/run_perturbed.py --scope pilot_gpu --domain math --limit 1"
echo "     python scripts/run_perturbed.py --scope pilot_gpu"
echo "     python scripts/run_perturbed.py --scope pilot_gpu_techcheck"
echo "     # full: python scripts/run_perturbed.py --scope full"
echo "     # Aborts if injections are missing/invalid for the selected scope."
echo "     # insertion_step mismatch → WARN (re-run Script 2); use --strict-insertion-step to abort."
echo "     # Debug partial runs only: --skip-injection-preflight"
echo ""
echo "  6. Evaluation (Script 4):"
echo "     python scripts/evaluate.py"
echo "============================================================"
