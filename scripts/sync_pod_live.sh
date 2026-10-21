#!/usr/bin/env bash
# sync_pod_live.sh — periodic backup of key pod JSONL files to Mac during a run.
set -uo pipefail

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
PORT="${PORT:-}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
REMOTE_BASE="${REMOTE_BASE:-/root/msc_thesis_experiment/data}"
DEST_DIR="${DEST_DIR:-backup_gpu_live}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
# 32b first, then 14b, then root — slow/failed fetches don't block later entries.
REMOTE_FILES=(
  32b/post_injection_runs.jsonl
  32b/complete_dialogues.jsonl
  14b/post_injection_runs.jsonl
  14b/complete_dialogues.jsonl
  post_injection_runs.jsonl
  complete_dialogues.jsonl
  injections_generated.jsonl
)

SSH_FETCH_OPTS=(
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=120
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=120
)

mkdir -p "$DEST_DIR" "$DEST_DIR/versions"

line_count() { [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0; }

fetch() {
  local remote_path="$1" tmp="$2"
  if [[ -n "$PORT" ]]; then
    scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no "$POD:$remote_path" "$tmp" 2>/dev/null && return 0
    set +o pipefail
    ssh -tt -i "$KEY" -p "$PORT" "${SSH_FETCH_OPTS[@]}" "$POD" 2>/dev/null <<EOF | grep '^{' > "$tmp" || true
cat '$remote_path'
exit
EOF
    set -o pipefail
  else
    scp -i "$KEY" -o StrictHostKeyChecking=no "$POD:$remote_path" "$tmp" 2>/dev/null && return 0
    set +o pipefail
    ssh -tt -i "$KEY" "${SSH_FETCH_OPTS[@]}" "$POD" 2>/dev/null <<EOF | grep '^{' > "$tmp" || true
cat '$remote_path'
exit
EOF
    set -o pipefail
  fi
  [[ -s "$tmp" ]]
}

echo "Live sync: $POD:$REMOTE_BASE/{${REMOTE_FILES[*]}} -> $DEST_DIR (every ${INTERVAL_SECONDS}s)"
while true; do
  TS="$(date '+%Y%m%d_%H%M%S')"
  for fname in "${REMOTE_FILES[@]}"; do
    mkdir -p "$(dirname "$DEST_DIR/$fname")" "$(dirname "$DEST_DIR/versions/$fname")"
    KEPT="$DEST_DIR/$fname"
    TMP="${DEST_DIR}/${fname}.${TS}.tmp"
    if fetch "$REMOTE_BASE/$fname" "$TMP"; then
      NEW_LINES="$(line_count "$TMP")"
      OLD_LINES="$(line_count "$KEPT")"
      if [[ "$NEW_LINES" -gt 0 && "$NEW_LINES" -ge "$OLD_LINES" ]]; then
        mv -f "$TMP" "$KEPT"
        cp -f "$KEPT" "$DEST_DIR/versions/${fname%.jsonl}.$TS.jsonl"
        echo "$(date '+%H:%M:%S') — $fname: $NEW_LINES rows (was $OLD_LINES) — updated"
      else
        rm -f "$TMP"
        echo "$(date '+%H:%M:%S') — $fname: fetched $NEW_LINES < kept $OLD_LINES; KEPT"
      fi
    else
      rm -f "$TMP"
      echo "$(date '+%H:%M:%S') — $fname: fetch FAILED"
    fi
  done
  sleep "$INTERVAL_SECONDS"
done
