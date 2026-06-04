#!/usr/bin/env bash
# GPU setup for DeployGPU / Hyperstack A100 80GB
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

if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing requirements..."
pip install -r requirements.txt

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo ""
  echo "NOTE: Set HF_TOKEN if the model requires authentication:"
  echo "  export HF_TOKEN=your_token_here"
  echo "Or run: huggingface-cli login"
else
  echo "==> Hugging Face token detected."
fi

mkdir -p outputs logs

echo ""
echo "Setup complete. Next steps:"
echo "  source .venv/bin/activate"
echo "  python src/run_pilot.py --config configs/smoke_config.yaml --dry-run"
echo "  python src/run_pilot.py --config configs/smoke_config.yaml"
