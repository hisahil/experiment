#!/usr/bin/env bash
# Poll RunPod until the GLM 5x baseline probe finishes (marker
# data/logs/glm_baseline_5x.DONE), then download the baseline file + log to the
# local workspace via base64-over-ssh. Exits after successful download.
#
# RunPod's SSH proxy requires -tt and only accepts commands via stdin (heredoc),
# not as ssh command arguments. Keeps the Mac awake (caffeinate tied to this PID)
# only while polling; caffeinate dies when this script exits.
#
# Usage: nohup bash scripts/watch_glm_baseline_5x.sh > data/logs/watch_glm_baseline_5x.local.log 2>&1 &
set -uo pipefail

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLL_INTERVAL="${POLL_INTERVAL:-300}"   # 5 minutes
MAX_POLLS="${MAX_POLLS:-48}"            # safety timeout: 48 * 300s = 4h

REMOTE_REPO="/root/msc_thesis_experiment"
REMOTE_BASELINE="data/baselines/glm-z1-9b/baseline_cots_5x.jsonl"
REMOTE_LOG="data/logs/glm_baseline_5x.log"
REMOTE_DONE="data/logs/glm_baseline_5x.DONE"

DEST_DIR="${ROOT}/backup_gpu_live/glm_baseline_5x"
AUTO_LOG="${DEST_DIR}/auto_download.log"

SSH_OPTS=(-tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 -o LogLevel=ERROR)

mkdir -p "$DEST_DIR"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$AUTO_LOG"; }

# Keep Mac awake while this script runs; caffeinate exits when this PID exits.
caffeinate -dimsu -w $$ &
CAFF_PID=$!
trap 'kill $CAFF_PID 2>/dev/null || true' EXIT

log "=== watch_glm_baseline_5x.sh started (PID $$, caffeinate PID $CAFF_PID) ==="
log "Pod: $POD | poll ${POLL_INTERVAL}s | max ${MAX_POLLS} polls (~$((POLL_INTERVAL*MAX_POLLS/3600))h)"
log "Watching marker: ${REMOTE_DONE}"

# Poll: emit STATUS_DONE / STATUS_RUNNING plus current line count of the baseline.
poll_status() {
    ssh "${SSH_OPTS[@]}" "$POD" 2>/dev/null <<RS | tr -d '\r' | sed $'s/\033\\[[0-9;?]*[a-zA-Z]//g' | grep -E '^STATUS_|^ROWS=' | tail -5
cd ${REMOTE_REPO}
if [ -f ${REMOTE_DONE} ]; then echo STATUS_DONE; else echo STATUS_RUNNING; fi
echo "ROWS=\$(wc -l < ${REMOTE_BASELINE} 2>/dev/null || echo 0)"
exit
RS
}

# Pull baseline + log as a single base64 tar (scp/rsync unreliable on RunPod proxy).
pull_results() {
    local b64="${DEST_DIR}/pull.b64"
    ssh "${SSH_OPTS[@]}" "$POD" 2>/dev/null <<RS | tr -d '\r' | sed $'s/\033\\[[0-9;?]*[a-zA-Z]//g' | awk '/B64START/{f=1;next}/B64END/{f=0}f' | grep -E '^[A-Za-z0-9+/=]{20,}$' > "$b64"
cd ${REMOTE_REPO}
echo B64START
tar czf - ${REMOTE_BASELINE} ${REMOTE_LOG} 2>/dev/null | base64
echo B64END
exit
RS
    [ -s "$b64" ] || return 1
    base64 --decode -i "$b64" > "${DEST_DIR}/pull.tar.gz" 2>/dev/null || return 1
    tar xzf "${DEST_DIR}/pull.tar.gz" -C "$DEST_DIR" 2>/dev/null || return 1
    rm -f "$b64" "${DEST_DIR}/pull.tar.gz"
    return 0
}

polls=0
while (( polls < MAX_POLLS )); do
    polls=$((polls+1))
    blob="$(poll_status || true)"
    rows="$(echo "$blob" | grep '^ROWS=' | tail -1 | cut -d= -f2)"
    if echo "$blob" | grep -q '^STATUS_DONE'; then
        log "DONE marker found (baseline rows=${rows:-?}). Pulling results..."
        if pull_results; then
            local_rows="$(wc -l < "${DEST_DIR}/${REMOTE_BASELINE}" 2>/dev/null | tr -d ' ' || echo 0)"
            log "Downloaded -> ${DEST_DIR}/${REMOTE_BASELINE} (${local_rows} rows) + log. Exiting."
            exit 0
        else
            log "Pull FAILED; retrying next cycle."
        fi
    else
        log "poll ${polls}/${MAX_POLLS}: still running (baseline rows=${rows:-0}) — sleep ${POLL_INTERVAL}s"
    fi
    sleep "$POLL_INTERVAL"
done

log "MAX_POLLS reached without DONE marker — giving up (job may still run on pod; pull manually)."
exit 2
