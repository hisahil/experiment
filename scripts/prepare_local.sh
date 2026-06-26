#!/usr/bin/env bash
# Run on Mac BEFORE HyperStack deploy — no GPU needed.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Local GPU preflight"
python scripts/preflight_gpu.py "$@"
