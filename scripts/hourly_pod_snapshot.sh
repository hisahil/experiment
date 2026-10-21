#!/usr/bin/env bash
# Hourly full snapshot from RunPod pod to Mac (base64 tar; scp unreliable on proxy).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
DEST_ROOT="${DEST_DIR:-$ROOT/backup_gpu_live}"

mkdir -p "$DEST_ROOT"

fetch_snapshot_b64() {
  local dest_b64="$1"
  ssh -tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=120 -o LogLevel=ERROR "$POD" 2>/dev/null <<'EOF' \
    | tr -d '\r' \
    | sed $'s/\033\\[[0-9;?]*[a-zA-Z]//g' \
    | awk '/B64START/{flag=1;next}/B64END/{flag=0}flag' \
    | grep -E '^[A-Za-z0-9+/=]{20,}$' > "$dest_b64"
cd /root/msc_thesis_experiment
echo B64START
tar czf - \
  data/injections_generated.jsonl \
  data/post_injection_runs.jsonl \
  data/complete_dialogues.jsonl \
  data/14b \
  data/32b \
  data/logs \
  results \
  2>/dev/null | base64
echo B64END
exit
EOF
  [[ -s "$dest_b64" ]]
}

while true; do
  STAMP="$(date +%Y%m%d_%H%M%S)"
  DEST="$DEST_ROOT/hourly_${STAMP}"
  mkdir -p "$DEST"
  if fetch_snapshot_b64 "$DEST/snapshot.b64"; then
    if base64 --decode -i "$DEST/snapshot.b64" > "$DEST/snapshot.tar.gz" 2>/dev/null \
      && tar xzf "$DEST/snapshot.tar.gz" -C "$DEST" 2>/dev/null; then
      rm -f "$DEST/snapshot.b64" "$DEST/snapshot.tar.gz"
      echo "$(date '+%Y-%m-%d %H:%M:%S') — hourly snapshot -> $DEST"
    else
      echo "$(date '+%Y-%m-%d %H:%M:%S') — base64 decode failed for $DEST" >&2
      rm -f "$DEST/snapshot.b64" "$DEST/snapshot.tar.gz"
    fi
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') — ssh snapshot FAILED" >&2
    rm -f "$DEST/snapshot.b64"
  fi
  sleep "$INTERVAL_SECONDS"
done
