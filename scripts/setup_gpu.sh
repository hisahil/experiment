#!/usr/bin/env bash
# GPU environment setup — RunPod A100 80GB (CUDA 12.8 / 13.0, driver 570+)
#
# Usage:
#   bash scripts/setup_gpu.sh            # default: legacy DeepSeek env (vLLM 0.6.6)
#   bash scripts/setup_gpu.sh cross      # cross-model env (vLLM 0.24+, mistral_common,
#                                        # transformers git) — required for
#                                        # GLM-Z1 / Gemma-4 / Ministral-3
#
# Cross-model rationale:
#   - GLM-Z1-9B          → needs --reasoning-parser glm45   (vLLM 0.7+)
#   - google/gemma-4-12B → needs --reasoning-parser gemma   (vLLM 0.11+)
#   - Ministral-3-14B    → needs --tokenizer_mode mistral   (vLLM 0.12 + mistral_common>=1.8.6
#                                                          + transformers 5.0.0rc0)
# Deep DeepSeek regression: re-run tests/test_continuation_prompt.py after
# upgrading; then a 5-task smoke on 1.5B via scripts/smoke_cross_model.py to
# catch bit-level tokenisation shifts before touching new models.
set -euo pipefail

MODE="${1:-legacy}"

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

# ── PyTorch cu128 (RunPod CUDA 12.8+; pairs with vLLM 0.20+) ─────────────────
# Legacy DeepSeek (vLLM 0.6.6) still uses cu121 below in the else branch only.
echo "==> Installing PyTorch..."
if [[ "$MODE" == "cross" ]]; then
  # Pin cu128 torch first — vLLM 0.24's default torch wheel can fail CUDA init on
  # Vast PyTorch images (driver 570 / CUDA 12.8 reports "driver too old").
  echo "    (cross) installing torch cu128 before vLLM 0.24.0"
  pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
else
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --quiet
fi

if [[ "$MODE" == "cross" ]]; then
  # ── Cross-model env: vLLM 0.24+, mistral_common, transformers (git) ────────
  echo "==> [cross-model] Installing vLLM 0.24.x (gemma4_unified) + mistral_common + transformers (git)..."
  pip install --upgrade "vllm==0.24.0" --quiet
  # vLLM may upgrade torch to an incompatible build — re-pin cu128.
  pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet --force-reinstall
  pip install --upgrade "mistral_common>=1.8.6" --quiet
  # Gemma-4 needs gemma4_unified (not in transformers 5.0.0rc0 PyPI wheel).
  pip install --upgrade "git+https://github.com/huggingface/transformers.git" --quiet
  # transformers 5.x requires tokenizers<=0.23.0; vLLM deps may pull 0.23.1 — pin.
  pip install --upgrade "tokenizers==0.22.2" --quiet
else
  # ── Legacy DeepSeek env: vLLM 0.6.6 + pinned transformers ─────────────────
  # (avoid ProcessorMixin / float8 conflicts; matches the historical DeepSeek
  # runs used as our baseline).
  echo "==> [legacy DeepSeek] Installing vLLM 0.6.6 + pinned deps..."
  pip install vllm==0.6.6 --quiet
  pip install "transformers>=4.44.0,<4.46" --quiet
fi


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

echo "==> Verifying vLLM import..."
python -c "from vllm import LLM; import vllm; print(f'  vLLM {vllm.__version__} import OK')"
if [[ "$MODE" == "cross" ]]; then
  python -c "import mistral_common; print(f'  mistral_common {mistral_common.__version__}')" || true
  python -c "import transformers; print(f'  transformers {transformers.__version__}')"
fi

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
  if [[ "$MODE" == "cross" ]]; then
    echo ""
    echo "WARNING: Gemma-4 (google/*) and Ministral (mistralai/*) are GATED."
    echo "  1. Accept the licence on the model page while logged into HF (once per account)."
    echo "  2. Set HF_TOKEN with read access to those namespaces."
    echo "     https://huggingface.co/google/gemma-4-12B-it"
    echo "     https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512"
  fi
else
  echo "==> Hugging Face token detected."
  if [[ "$MODE" == "cross" ]]; then
    echo "    Verify gated-model access with: hf auth whoami"
    echo "    Ensure Terms accepted for google/gemma-4-12B-it and"
    echo "    mistralai/Ministral-3-14B-Reasoning-2512."
  fi
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
