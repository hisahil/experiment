#!/usr/bin/env bash
# Re-run P12–P14 cross-model smokes on pod (sequential, one vLLM at a time).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
mkdir -p data/logs
LOG="$ROOT/data/logs/cross_smoke_p12_14.log"
# evaluate.py needs pandas/scipy; cross venv sometimes omits them if setup was partial.
pip install 'pandas>=2.1.0' 'scipy>=1.11.0' -q
exec > >(tee -a "$LOG") 2>&1

echo "=== cross_smoke start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# GLM: dialogues already complete (540 rows) — evaluate (+ optional judge refresh).
python scripts/smoke_cross_model.py \
  --model zai-org/GLM-Z1-9B-0414 \
  --scope full_math \
  --skip baselines convert injections dialogues judge \
  || echo "[WARN] GLM smoke non-zero exit: $?"

# Gemma: regenerate baselines with git-transformers chat template.
python scripts/smoke_cross_model.py \
  --model google/gemma-4-12B-it \
  --scope full_math \
  || echo "[WARN] Gemma smoke non-zero exit: $?"

# Ministral: baselines exist; reuse DeepSeek injections.
python scripts/smoke_cross_model.py \
  --model mistralai/Ministral-3-14B-Reasoning-2512 \
  --scope full_math \
  --skip baselines convert injections \
  || echo "[WARN] Ministral smoke non-zero exit: $?"

echo "=== cross_smoke done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
