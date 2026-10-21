#!/usr/bin/env bash
# Start Mac sidecars: live pod sync + hourly snapshots during an active pod run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIVE_DIR="${DEST_DIR:-$ROOT/backup_gpu_live}"
mkdir -p "$LIVE_DIR"
SYNC_LOG="$LIVE_DIR/sync.log"
HOURLY_LOG="$LIVE_DIR/hourly_snapshot.log"

if pgrep -f 'sync_pod_live.sh' >/dev/null 2>&1; then
  echo "sync_pod_live: already running (PID $(pgrep -f 'sync_pod_live.sh' | head -1))"
else
  nohup bash "$ROOT/scripts/sync_pod_live.sh" >>"$SYNC_LOG" 2>&1 &
  echo "sync_pod_live: started (PID $!)"
fi

if pgrep -f 'hourly_pod_snapshot.sh' >/dev/null 2>&1; then
  echo "hourly_pod_snapshot: already running (PID $(pgrep -f 'hourly_pod_snapshot.sh' | head -1))"
else
  nohup bash "$ROOT/scripts/hourly_pod_snapshot.sh" >>"$HOURLY_LOG" 2>&1 &
  echo "hourly_pod_snapshot: started (PID $!)"
fi

echo ""
tail -3 "$SYNC_LOG" 2>/dev/null || true
tail -3 "$HOURLY_LOG" 2>/dev/null || true
