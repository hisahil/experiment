#!/usr/bin/env bash
# Poll RunPod until the 1.5B baseline generation (tmux session p1a_1_5b) finishes,
# then download baselines + logs to local workspace via base64-over-ssh.
# Exits after successful download. Safe to run multiple times.
#
# Must run inside a detached `screen` session so ssh -tt has a PTY.
# RunPod's SSH proxy requires -tt and only accepts commands via stdin (heredoc),
# not as ssh command arguments.
set -uo pipefail

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLL_INTERVAL=300   # 5 minutes
POOL_TARGET="${POOL_TARGET:-304}"

LOCAL_BASELINE_DIR="${ROOT}/data/task_selection/baselines/1.5b"
LOCAL_LOG_DIR="${ROOT}/data/task_selection/logs"
AUTO_LOG="${LOCAL_LOG_DIR}/auto_download.log"

SSH_OPTS=(-tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 -o LogLevel=ERROR)

mkdir -p "$LOCAL_BASELINE_DIR" "$LOCAL_LOG_DIR"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$AUTO_LOG"; }

# Keep Mac awake while this script runs
caffeinate -dims -w $$ &
CAFF_PID=$!
trap 'kill $CAFF_PID 2>/dev/null || true' EXIT

log "=== watch_1.5b_download.sh started (PID $$, caffeinate PID $CAFF_PID) ==="
log "Pod: $POD"
log "Poll interval: ${POLL_INTERVAL}s"
log "Local baseline dir: $LOCAL_BASELINE_DIR"
log "Watching tmux session: p1a_1_5b (pool target: ${POOL_TARGET} lines)"

parse_poll_kv() {
    local key="$1" blob="$2"
    echo "$blob" | grep "^POLL_${key}=" | tail -1 | cut -d= -f2-
}

# Poll remote job status. Emits only POLL_* lines (RunPod PTY echoes heredoc source;
# do not capture raw ssh output in command substitution without filtering).
poll_pod_status() {
    local err_file poll_lines
    err_file="$(mktemp "${TMPDIR:-/tmp}/watch_1.5b_ssh.XXXXXX")"

    poll_lines="$(
        ssh "${SSH_OPTS[@]}" "$POD" 2>"$err_file" <<'RS' | tr -d '\r' | sed $'s/\033\\[[0-9;?]*[a-zA-Z]//g' | grep '^POLL_' | tail -10
cd /root/msc_thesis_experiment

_tmux_run=0
_tmux_sess=""
for sess in p1a_1_5b; do
    if tmux has-session -t "$sess" 2>/dev/null; then
        _tmux_run=1
        _tmux_sess="$sess"
        break
    fi
done

_base100=0
_pool=0
if [ -f data/task_selection/baselines/1.5b/baseline_cots_100tasks.jsonl ]; then
    _base100=$(wc -l < data/task_selection/baselines/1.5b/baseline_cots_100tasks.jsonl | tr -d ' ')
fi
if [ -f data/task_selection/baselines/1.5b/baseline_cots_pool.jsonl ]; then
    _pool=$(wc -l < data/task_selection/baselines/1.5b/baseline_cots_pool.jsonl | tr -d ' ')
fi

_log_done=0
if grep -q "1.5B on 304 pool DONE" data/task_selection/logs/1.5b_*.log 2>/dev/null; then
    _log_done=1
fi

echo "POLL_TMUX_RUNNING=${_tmux_run}"
echo "POLL_TMUX_SESSION=${_tmux_sess}"
echo "POLL_BASELINE_100_LINES=${_base100}"
echo "POLL_POOL_LINES=${_pool}"
echo "POLL_LOG_DONE=${_log_done}"
exit 0
RS
    )"

    if [[ -z "$poll_lines" ]]; then
        if [[ -s "$err_file" ]]; then
            log "SSH poll failed (no POLL_ lines). stderr: $(tr '\n' ' ' < "$err_file" | sed 's/  */ /g')"
        else
            log "SSH poll failed (no POLL_ lines, empty stderr — check PTY/screen and pod connectivity)"
        fi
        rm -f "$err_file"
        return 1
    fi

    if [[ -s "$err_file" ]]; then
        log "SSH poll stderr (non-fatal): $(tr '\n' ' ' < "$err_file" | sed 's/  */ /g')"
    fi
    rm -f "$err_file"

    POLL_BLOB="$poll_lines"
    return 0
}

# --- Polling loop ---
while true; do
    if ! poll_pod_status; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    TMUX_RUNNING="$(parse_poll_kv TMUX_RUNNING "$POLL_BLOB")"
    TMUX_SESSION="$(parse_poll_kv TMUX_SESSION "$POLL_BLOB")"
    BASELINE_100_LINES="$(parse_poll_kv BASELINE_100_LINES "$POLL_BLOB")"
    POOL_LINES="$(parse_poll_kv POOL_LINES "$POLL_BLOB")"
    LOG_DONE="$(parse_poll_kv LOG_DONE "$POLL_BLOB")"

    if [[ "$LOG_DONE" == "1" ]]; then
        log "Log marker '1.5B on 304 pool DONE' found — job finished. Starting download..."
        break
    fi

    if [[ "$TMUX_RUNNING" == "0" && "${POOL_LINES:-0}" -ge "$POOL_TARGET" ]]; then
        log "tmux session gone and baseline_cots_pool.jsonl has ${POOL_LINES}/${POOL_TARGET} lines — job finished. Starting download..."
        break
    fi

    if [[ "$TMUX_RUNNING" == "1" ]]; then
        log "tmux ${TMUX_SESSION:-p1a_1_5b} still running (100-task=${BASELINE_100_LINES:-0}, pool=${POOL_LINES:-0}/${POOL_TARGET}, log_done=${LOG_DONE:-0}) — sleeping ${POLL_INTERVAL}s"
        sleep "$POLL_INTERVAL"
        continue
    fi

    log "Job not complete yet (tmux=${TMUX_RUNNING:-?}, 100-task=${BASELINE_100_LINES:-0}, pool=${POOL_LINES:-0}/${POOL_TARGET}, log_done=${LOG_DONE:-0}). Retrying in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

# --- Check remote files ---
log "Checking remote files..."
REMOTE_CHECK="$(ssh "${SSH_OPTS[@]}" "$POD" 2>>"$AUTO_LOG" <<'CHKFILES' | tr -d '\r' | grep '^RFILE=' || true
cd /root/msc_thesis_experiment
for f in \
    data/task_selection/baselines/1.5b/baseline_cots_100tasks.jsonl \
    data/task_selection/baselines/1.5b/baseline_cots_pool.jsonl; do
    if [ -f "$f" ]; then
        echo "RFILE=OK $(wc -l < "$f") $f"
    else
        echo "RFILE=MISSING $f"
    fi
done
ls data/task_selection/logs/1.5b_*.log 2>/dev/null | while read -r lf; do
    echo "RFILE=OK $(wc -l < "$lf") $lf"
done
exit 0
CHKFILES
)"
echo "$REMOTE_CHECK" | while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    log "  ${line#RFILE=}"
done

if echo "$REMOTE_CHECK" | grep -q "RFILE=MISSING"; then
    log "[WARN] Some expected files missing on pod — downloading what exists anyway"
fi

# --- Pack on the pod ---
log "Packing remote files into tarball..."
ssh "${SSH_OPTS[@]}" "$POD" 2>>"$AUTO_LOG" <<'PACK' | tr -d '\r' | grep '^PACK_' | while IFS= read -r l; do log "  ${l#PACK_}"; done || true
cd /root/msc_thesis_experiment
tar czf /tmp/baseline_1.5b_results.tar.gz \
    data/task_selection/baselines/1.5b/ \
    $(ls data/task_selection/logs/1.5b_*.log 2>/dev/null) \
    2>/dev/null
echo "PACK_$(ls -lh /tmp/baseline_1.5b_results.tar.gz)"
exit 0
PACK

# --- Stream tarball via base64 ---
log "Streaming tarball via base64..."
ssh "${SSH_OPTS[@]}" "$POD" 2>>"$AUTO_LOG" <<'B64DL' | tr -d '\r' | sed -n '/^B64_START$/,/^B64_END$/p' | grep -v '^B64_' | base64 -d > /tmp/baseline_1.5b_results.tar.gz
echo "B64_START"
base64 < /tmp/baseline_1.5b_results.tar.gz
echo "B64_END"
exit 0
B64DL

TARSIZE="$(ls -lh /tmp/baseline_1.5b_results.tar.gz 2>/dev/null | awk '{print $5}')"
if [[ -z "$TARSIZE" ]] || [[ ! -s /tmp/baseline_1.5b_results.tar.gz ]]; then
    log "[FAIL] Downloaded tarball is empty or missing. Aborting."
    exit 1
fi
log "Downloaded tarball: ${TARSIZE}"

# --- Extract and copy without clobbering ---
TMPEXT="$(mktemp -d)"
tar xzf /tmp/baseline_1.5b_results.tar.gz -C "$TMPEXT"

log "Extracted contents:"
find "$TMPEXT" -type f | while read -r f; do
    log "  ${f#$TMPEXT/}"
done

if [[ -d "$TMPEXT/data/task_selection/baselines/1.5b" ]]; then
    for f in "$TMPEXT/data/task_selection/baselines/1.5b/"*; do
        [[ -f "$f" ]] || continue
        BASENAME="$(basename "$f")"
        DEST="$LOCAL_BASELINE_DIR/$BASENAME"
        if [[ -f "$DEST" ]]; then
            log "[SKIP] $BASENAME already exists locally"
        else
            cp "$f" "$DEST"
            log "[COPIED] $BASENAME"
        fi
    done
fi

shopt -s nullglob
for f in "$TMPEXT/data/task_selection/logs/"1.5b_*.log; do
    [[ -f "$f" ]] || continue
    BASENAME="$(basename "$f")"
    DEST="$LOCAL_LOG_DIR/$BASENAME"
    if [[ -f "$DEST" ]]; then
        log "[SKIP] $BASENAME already exists locally"
    else
        cp "$f" "$DEST"
        log "[COPIED] $BASENAME"
    fi
done

# --- Cleanup ---
rm -rf "$TMPEXT" /tmp/baseline_1.5b_results.tar.gz
ssh "${SSH_OPTS[@]}" "$POD" 2>>"$AUTO_LOG" <<'RMTMP' || true
rm -f /tmp/baseline_1.5b_results.tar.gz
exit 0
RMTMP

# --- Verification ---
log "=== Verification ==="
for f in "$LOCAL_BASELINE_DIR"/*; do
    [[ -f "$f" ]] || continue
    LINES="$(wc -l < "$f" | tr -d ' ')"
    SIZE="$(ls -lh "$f" | awk '{print $5}')"
    log "  $(basename "$f"): ${LINES} lines, ${SIZE}"
done
for f in "$LOCAL_LOG_DIR"/1.5b_*.log; do
    [[ -f "$f" ]] || continue
    LINES="$(wc -l < "$f" | tr -d ' ')"
    log "  $(basename "$f"): ${LINES} lines"
done

if command -v osascript >/dev/null 2>&1; then
    osascript -e 'display notification "1.5B baseline download complete!" with title "MSc Thesis" sound name "Glass"' || true
fi

log "=== watch_1.5b_download.sh completed successfully ==="
