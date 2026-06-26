#!/usr/bin/env bash
# Start Mac sidecars: live pod JSONL sync + optional rejudge_fallbacks (after judge smoke test).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

LIVE_DIR="${DEST_DIR:-$ROOT/backup_gpu_live}"
mkdir -p "$LIVE_DIR"
SYNC_LOG="$LIVE_DIR/sync.log"
REJUDGE_LOG="$LIVE_DIR/rejudge.log"

SYNC_PID=""
REJUDGE_PID=""

if pgrep -f 'sync_pod_live.sh' >/dev/null 2>&1; then
  SYNC_PID="$(pgrep -f 'sync_pod_live.sh' | head -1)"
  echo "sync_pod_live: already running (PID $SYNC_PID)"
else
  nohup bash "$ROOT/scripts/sync_pod_live.sh" >>"$SYNC_LOG" 2>&1 &
  SYNC_PID=$!
  echo "sync_pod_live: started (PID $SYNC_PID)"
fi

if pgrep -f 'rejudge_fallbacks.py' >/dev/null 2>&1; then
  REJUDGE_PID="$(pgrep -f 'rejudge_fallbacks.py' | head -1)"
  echo "rejudge_fallbacks: already running (PID $REJUDGE_PID)"
else
  if python3 "$ROOT/scripts/smoke_judge.py"; then
    REJUDGE_INPUT="${REJUDGE_INPUT:-$LIVE_DIR/post_injection_runs.jsonl}"
    nohup python3 "$ROOT/scripts/rejudge_fallbacks.py" \
      --config "$ROOT/configs/experiment.yaml" \
      --input "$REJUDGE_INPUT" \
      --tasks "$ROOT/data/experiment_tasks.jsonl" \
      --out "$ROOT/data/post_injection_runs.rejudged.jsonl" \
      >>"$REJUDGE_LOG" 2>&1 &
    REJUDGE_PID=$!
    echo "rejudge_fallbacks: started (PID $REJUDGE_PID)"
  else
    echo "rejudge_fallbacks: NOT started (smoke_judge failed)" >&2
  fi
fi

echo ""
echo "PIDs: sync=${SYNC_PID:-—}  rejudge=${REJUDGE_PID:-—}"
echo ""
echo "--- tail $SYNC_LOG ---"
tail -5 "$SYNC_LOG" 2>/dev/null || echo "(empty)"
echo ""
echo "--- tail $REJUDGE_LOG ---"
tail -5 "$REJUDGE_LOG" 2>/dev/null || echo "(empty)"
