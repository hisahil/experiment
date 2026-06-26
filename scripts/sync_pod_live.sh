#!/usr/bin/env bash
#
# sync_pod_live.sh — periodic, atomic, versioned backup of the GPU pod's run JSONL
# to the Mac while a resume is in progress. Run this on the MAC, not the pod.
#
# Safety properties:
#   * Atomic: scp to a .tmp file, then mv into place (no partial last line consumed).
#   * Non-destructive: only replaces the kept copy when the freshly fetched file has
#     MORE lines than the current one (a partial/truncated fetch never overwrites a
#     good backup).
#   * Versioned: every accepted update is also snapshotted with a timestamp.
#   * SCP fallback: if scp fails (e.g. PTY issues), falls back to `ssh ... cat`.
#
# Configure POD/PORT/KEY (from the RunPod dashboard) before running, then:
#   chmod +x scripts/sync_pod_live.sh
#   ./scripts/sync_pod_live.sh
#
set -uo pipefail

# ── Configuration (edit these) ──────────────────────────────────────────────
# RunPod proxy: 020gpg3omasfwu-64410d4f@ssh.runpod.io (no -P port needed)
POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
PORT="${PORT:-}"                       # optional; omit for RunPod ssh.runpod.io proxy
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
REMOTE_FILE="${REMOTE_FILE:-/root/msc_thesis_experiment/data/post_injection_runs.jsonl}"
DEST_DIR="${DEST_DIR:-backup_gpu_live}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
# ────────────────────────────────────────────────────────────────────────────

mkdir -p "$DEST_DIR" "$DEST_DIR/versions"
KEPT="$DEST_DIR/post_injection_runs.jsonl"

line_count() {
    [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0
}

fetch() {
    # Try scp first; RunPod proxy usually needs -tt heredoc fallback (banner stripped via grep).
    local tmp="$1"
    if [[ -n "$PORT" ]]; then
        scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no \
            "$POD:$REMOTE_FILE" "$tmp" 2>/dev/null && return 0
        ssh -tt -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no "$POD" 2>/dev/null <<EOF | grep '^{' > "$tmp"
cat '$REMOTE_FILE'
exit
EOF
    else
        scp -i "$KEY" -o StrictHostKeyChecking=no \
            "$POD:$REMOTE_FILE" "$tmp" 2>/dev/null && return 0
        ssh -tt -i "$KEY" -o StrictHostKeyChecking=no "$POD" 2>/dev/null <<EOF | grep '^{' > "$tmp"
cat '$REMOTE_FILE'
exit
EOF
    fi
    [[ -s "$tmp" ]]
}

echo "Live sync: $POD:$REMOTE_FILE -> $KEPT (every ${INTERVAL_SECONDS}s)"
while true; do
    TS="$(date '+%Y%m%d_%H%M%S')"
    TMP="$DEST_DIR/.post_injection_runs.$TS.tmp"

    if fetch "$TMP"; then
        NEW_LINES="$(line_count "$TMP")"
        OLD_LINES="$(line_count "$KEPT")"
        if [[ "$NEW_LINES" -gt 0 && "$NEW_LINES" -ge "$OLD_LINES" ]]; then
            mv -f "$TMP" "$KEPT"                       # atomic replace
            cp -f "$KEPT" "$DEST_DIR/versions/post_injection_runs.$TS.jsonl"
            echo "$(date '+%H:%M:%S') — $NEW_LINES rows (was $OLD_LINES) — updated"
        else
            rm -f "$TMP"
            echo "$(date '+%H:%M:%S') — fetched $NEW_LINES rows < kept $OLD_LINES; KEPT (no overwrite)"
        fi
    else
        rm -f "$TMP"
        echo "$(date '+%H:%M:%S') — fetch FAILED (scp + ssh cat); keeping previous backup"
    fi

    sleep "$INTERVAL_SECONDS"
done
