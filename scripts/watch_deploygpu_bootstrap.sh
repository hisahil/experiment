#!/usr/bin/env bash
# Keep trying OLMo pod bootstrap until SSH works or MAX_HOURS elapsed.
# Reads connection from data/deploygpu_instance.env (edit host/port if dashboard changes).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/data/deploygpu_instance.env"
LOG="$ROOT/data/logs/watch_deploygpu_bootstrap.log"
MAX_HOURS="${MAX_HOURS:-6}"
INTERVAL="${INTERVAL:-60}"

mkdir -p "$ROOT/data/logs"
exec >>"$LOG" 2>&1

echo "=== watch_deploygpu_bootstrap $(date -u +%Y-%m-%dT%H:%M:%SZ) max_h=${MAX_HOURS} ==="

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
fi

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
attempt=0

while (( $(date +%s) < deadline )); do
  attempt=$((attempt + 1))
  echo "--- attempt $attempt $(date -u +%H:%M:%SZ) host=${DEPLOYGPU_HOST:-?} port=${DEPLOYGPU_PORT:-?} ---"
  if DEPLOYGPU_HOST="${DEPLOYGPU_HOST:-212.13.234.23}" \
     DEPLOYGPU_PORT="${DEPLOYGPU_PORT:-30342}" \
     DEPLOYGPU_KEY="${DEPLOYGPU_KEY:-$HOME/.ssh/deploygpu_ed25519}" \
     bash "$ROOT/scripts/bootstrap_deploygpu_olmo.sh"; then
    echo "=== bootstrap success $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    exit 0
  fi
  echo "bootstrap failed; retry in ${INTERVAL}s"
  sleep "$INTERVAL"
done

echo "ERROR: gave up after ${MAX_HOURS}h — check DeployGPU dashboard (instance Running? new IP/port?)"
exit 1
