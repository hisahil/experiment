#!/usr/bin/env bash
# One-shot bootstrap for DeployGPU / Vast.ai OLMo pod.
# Run from Mac repo root:
#   bash scripts/bootstrap_deploygpu_olmo.sh
#
# Connection priority:
#   1. Vast CLI proxy/direct SSH (VAST_INSTANCE_ID + ~/.config/vastai/vast_api_key)
#   2. DEPLOYGPU_HOST:DEPLOYGPU_PORT from data/deploygpu_instance.env
#
# If all SSH fails, use Jupyter workaround:
#   bash scripts/vast_jupyter_bootstrap_oneliner.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="$ROOT/data/deploygpu_instance.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
fi

HOST="${DEPLOYGPU_HOST:-212.13.234.23}"
PORT="${DEPLOYGPU_PORT:-30342}"
USER="${DEPLOYGPU_USER:-root}"
KEY="${DEPLOYGPU_KEY:-$HOME/.ssh/deploygpu_ed25519}"
KEY="${KEY/#\~/$HOME}"
VAST_ID="${VAST_INSTANCE_ID:-}"

# Try Vast CLI for real SSH endpoint (often works when DeployGPU Quick Connect doesn't)
if [[ -n "$VAST_ID" && -f "${VAST_API_KEY_FILE:-$HOME/.config/vastai/vast_api_key}" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/vast_resolve_ssh.sh"
  if vast_resolve_ssh "$VAST_ID"; then
    if [[ -n "${VAST_SSH_HOST:-}" && -n "${VAST_SSH_PORT:-}" ]]; then
      HOST="$VAST_SSH_HOST"
      PORT="$VAST_SSH_PORT"
      USER="${VAST_SSH_USER:-root}"
      echo "Using Vast-resolved SSH: ${USER}@${HOST}:${PORT}"
    fi
  fi
fi

REMOTE="${USER}@${HOST}"
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)
RSYNC_SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
LOG="$ROOT/data/logs/bootstrap_deploygpu_olmo_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/data/logs"

exec > >(tee -a "$LOG") 2>&1
echo "=== bootstrap_deploygpu_olmo start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    host=$HOST port=$PORT user=$USER key=$KEY vast_id=${VAST_ID:-none} log=$LOG"

# ── Wait for SSH ─────────────────────────────────────────────────────────────
MAX_WAIT="${SSH_WAIT_SEC:-3600}"
INTERVAL=15
elapsed=0
until "${SSH[@]}" "$REMOTE" 'echo SSH_READY' 2>/dev/null; do
  if (( elapsed >= MAX_WAIT )); then
    echo "ERROR: SSH not reachable after ${MAX_WAIT}s ($HOST:$PORT)" >&2
    echo "" >&2
    echo "Direct DeployGPU Quick Connect is often blocked from home networks." >&2
    echo "Workaround — Jupyter Terminal on Vast instance ${VAST_ID:-44470343}:" >&2
    echo "  bash scripts/vast_jupyter_bootstrap_oneliner.sh" >&2
    echo "" >&2
    echo "Or fix proxy SSH:" >&2
    echo "  echo YOUR_KEY > ~/.config/vastai/vast_api_key" >&2
    echo "  vastai ssh-url ${VAST_ID:-44470343}" >&2
    echo "  bash scripts/bootstrap_deploygpu_olmo.sh" >&2
    exit 1
  fi
  if (( elapsed > 0 && elapsed % 120 == 0 )) && [[ -n "$VAST_ID" ]]; then
    echo "Still waiting… try: vastai ssh $VAST_ID  (requires API key)"
  fi
  echo "Waiting for SSH... (${elapsed}s / ${MAX_WAIT}s)"
  sleep "$INTERVAL"
  elapsed=$((elapsed + INTERVAL))
done
echo "SSH OK"

# ── Sync repo ───────────────────────────────────────────────────────────────
echo "==> rsync project..."
rsync -avz --progress \
  -e "$RSYNC_SSH" \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude 'backup_gpu_live' \
  --exclude 'data/post_injection_runs.jsonl' \
  --exclude 'data/complete_dialogues.jsonl' \
  --exclude 'data/adaptive_dialogues.jsonl' \
  "$ROOT/" "$REMOTE:~/msc_thesis_experiment/"

if [[ -f "$ROOT/.env" ]]; then
  echo "==> copy .env"
  scp -P "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new \
    "$ROOT/.env" "$REMOTE:~/msc_thesis_experiment/.env"
fi

# ── Remote setup via olmo_pod_setup.sh ───────────────────────────────────────
echo "==> remote olmo_pod_setup.sh..."
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE_EOF'
set -euo pipefail
cd ~/msc_thesis_experiment
bash scripts/olmo_pod_setup.sh
REMOTE_EOF

echo "=== bootstrap_deploygpu_olmo done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Monitor: ssh -i $KEY -p $PORT $REMOTE 'tmux attach -t olmo_smoke'"
echo "Log: $LOG"
