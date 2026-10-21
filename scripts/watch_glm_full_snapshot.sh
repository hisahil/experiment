#!/usr/bin/env bash
# Periodically snapshot the GLM full run from RunPod using one base64 tar stream.
# Each successful pull atomically replaces backup_gpu_live/glm_full_live/.
set -uo pipefail

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1800}"
MAX_HOURS="${MAX_HOURS:-6}"

REMOTE_REPO="/root/msc_thesis_experiment"
DEST_DIR="${ROOT}/backup_gpu_live/glm_full_live"
DEST_PARENT="$(dirname "$DEST_DIR")"
SSH_OPTS=(-tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 -o LogLevel=ERROR)

mkdir -p "$DEST_PARENT"

log() {
    echo "[$(date -u +%FT%TZ)] $*"
}

if ! [[ "$SNAPSHOT_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    log "ERROR: SNAPSHOT_INTERVAL must be a positive integer (got: $SNAPSHOT_INTERVAL)"
    exit 2
fi
if ! [[ "$MAX_HOURS" =~ ^[1-9][0-9]*$ ]]; then
    log "ERROR: MAX_HOURS must be a positive integer (got: $MAX_HOURS)"
    exit 2
fi
if [[ ! -r "$KEY" ]]; then
    log "ERROR: SSH key is not readable: $KEY"
    exit 2
fi

CAFF_PID=""
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu -w $$ &
    CAFF_PID=$!
fi

cleanup() {
    if [[ -n "$CAFF_PID" ]]; then
        kill "$CAFF_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

status_value() {
    local key="$1"
    awk -F= -v wanted="$key" '
        $1 == wanted {
            sub(/^[^=]*=/, "")
            value = $0
        }
        END { print value }
    '
}

poll_status() {
    local payload
    payload="$(
        base64 <<'RS' | tr -d '\n'
cd /root/msc_thesis_experiment || exit 1

_tmux=0
tmux has-session -t glm_full 2>/dev/null && _tmux=1

_process=0
pgrep -af 'python .*scripts/run_adaptive_dialogue.py' >/dev/null && _process=1

_complete_rows=0
_post_rows=0
[ -f data/glm-z1-9b/complete_dialogues.jsonl ] &&
    _complete_rows=$(wc -l < data/glm-z1-9b/complete_dialogues.jsonl | tr -d ' ')
[ -f data/glm-z1-9b/post_injection_runs.jsonl ] &&
    _post_rows=$(wc -l < data/glm-z1-9b/post_injection_runs.jsonl | tr -d ' ')

_stopped_reason=missing
if [ -f results/glm-z1-9b/run_metadata.json ]; then
    _stopped_reason=$(
        .venv/bin/python -c \
            'import json; print(json.load(open("results/glm-z1-9b/run_metadata.json")).get("stopped_reason", "missing"))' \
            2>/dev/null || echo unreadable
    )
fi

_done=0
for _marker in \
    data/glm-z1-9b/GLM_FULL.DONE \
    data/glm-z1-9b/PIPELINE_COMPLETE \
    results/glm-z1-9b/GLM_FULL.DONE \
    results/glm-z1-9b/PIPELINE_COMPLETE \
    logs/glm_full.DONE; do
    [ -f "$_marker" ] && _done=1
done

_latest_log=none
_latest_log_rows=0
_candidate=$(ls -1t logs/glm_full_*.log 2>/dev/null | awk 'NR == 1 { print; exit }')
if [ -n "$_candidate" ] && [ -f "$_candidate" ]; then
    _latest_log="$_candidate"
    _latest_log_rows=$(wc -l < "$_candidate" | tr -d ' ')
fi

printf '\nSNAP_TMUX=%s\n' "$_tmux"
printf 'SNAP_PROCESS=%s\n' "$_process"
printf 'SNAP_COMPLETE_ROWS=%s\n' "$_complete_rows"
printf 'SNAP_POST_ROWS=%s\n' "$_post_rows"
printf 'SNAP_STOPPED_REASON=%s\n' "$_stopped_reason"
printf 'SNAP_DONE=%s\n' "$_done"
printf 'SNAP_LATEST_LOG=%s\n' "$_latest_log"
printf 'SNAP_LATEST_LOG_ROWS=%s\n' "$_latest_log_rows"
exit
RS
    )"
    printf 'echo %s | base64 -d | bash\nexit\n' "$payload" |
        ssh "${SSH_OPTS[@]}" "$POD" 2>/dev/null |
        tr -d '\r' |
        awk 'match($0, /SNAP_[A-Z_]+=/) { print substr($0, RSTART) }'
}

pull_snapshot() {
    local b64 tarball stage previous payload had_previous=0
    b64="$(mktemp "${DEST_PARENT}/.glm_full_pull.b64.XXXXXX")" || return 1
    tarball="$(mktemp "${DEST_PARENT}/.glm_full_pull.tar.gz.XXXXXX")" || {
        rm -f "$b64"
        return 1
    }
    stage="$(mktemp -d "${DEST_PARENT}/.glm_full_live.stage.XXXXXX")" || {
        rm -f "$b64" "$tarball"
        return 1
    }

    payload="$(
        base64 <<'RS' | tr -d '\n'
cd /root/msc_thesis_experiment || exit 1
_files=()
for _file in \
    data/glm-z1-9b/complete_dialogues.jsonl \
    data/glm-z1-9b/post_injection_runs.jsonl \
    results/glm-z1-9b/run_timings.json \
    results/glm-z1-9b/run_metadata.json; do
    [ -f "$_file" ] && _files+=("$_file")
done
shopt -s nullglob
for _file in logs/glm_full_*.log; do
    [ -f "$_file" ] && _files+=("$_file")
done
printf '\nB64START\n'
if [ "${#_files[@]}" -gt 0 ]; then
    tar czf - "${_files[@]}" 2>/dev/null | base64
fi
printf 'B64END\n'
exit
RS
    )"
    printf 'echo %s | base64 -d | bash\nexit\n' "$payload" |
        ssh "${SSH_OPTS[@]}" "$POD" 2>/dev/null |
        tr -d '\r' |
        awk '
            /^B64START$/ { inside=1; next }
            /^B64END$/ { inside=0; exit }
            inside && $0 ~ /^[A-Za-z0-9+\/=]+$/ { print }
        ' >"$b64"

    if [[ ! -s "$b64" ]] ||
       ! base64 --decode -i "$b64" >"$tarball" 2>/dev/null ||
       [[ ! -s "$tarball" ]] ||
       ! tar tzf "$tarball" >/dev/null 2>&1 ||
       ! tar xzf "$tarball" -C "$stage" 2>/dev/null; then
        rm -rf "$b64" "$tarball" "$stage"
        return 1
    fi

    previous="${DEST_PARENT}/.glm_full_live.previous.$$"
    rm -rf "$previous"
    if [[ -d "$DEST_DIR" ]]; then
        mv "$DEST_DIR" "$previous" || {
            rm -rf "$b64" "$tarball" "$stage"
            return 1
        }
        had_previous=1
    fi
    if ! mv "$stage" "$DEST_DIR"; then
        [[ "$had_previous" == "1" ]] && mv "$previous" "$DEST_DIR" 2>/dev/null || true
        rm -rf "$b64" "$tarball" "$stage"
        return 1
    fi
    rm -rf "$previous" "$b64" "$tarball"
    return 0
}

local_rows() {
    local file="$1"
    if [[ -f "$file" ]]; then
        wc -l <"$file" | tr -d ' '
    else
        echo 0
    fi
}

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
iteration=0
seen_running=0

log "watcher started pid=$$ pod=$POD interval=${SNAPSHOT_INTERVAL}s max=${MAX_HOURS}h"
log "destination=$DEST_DIR caffeinate_pid=${CAFF_PID:-none}"

while (( $(date +%s) < deadline )); do
    iteration=$((iteration + 1))
    blob="$(poll_status || true)"

    tmux_running="$(printf '%s\n' "$blob" | status_value SNAP_TMUX)"
    process_running="$(printf '%s\n' "$blob" | status_value SNAP_PROCESS)"
    complete_rows="$(printf '%s\n' "$blob" | status_value SNAP_COMPLETE_ROWS)"
    post_rows="$(printf '%s\n' "$blob" | status_value SNAP_POST_ROWS)"
    stopped_reason="$(printf '%s\n' "$blob" | status_value SNAP_STOPPED_REASON)"
    done_marker="$(printf '%s\n' "$blob" | status_value SNAP_DONE)"
    latest_log="$(printf '%s\n' "$blob" | status_value SNAP_LATEST_LOG)"
    latest_log_rows="$(printf '%s\n' "$blob" | status_value SNAP_LATEST_LOG_ROWS)"

    [[ "$tmux_running" == "1" || "$process_running" == "1" ]] && seen_running=1
    log "iteration=$iteration remote_rows complete=${complete_rows:-?} post=${post_rows:-?} tmux=${tmux_running:-?} process=${process_running:-?} metadata=${stopped_reason:-?} done=${done_marker:-?} log=${latest_log:-?} log_rows=${latest_log_rows:-?}"

    if pull_snapshot; then
        local_complete="$(local_rows "$DEST_DIR/data/glm-z1-9b/complete_dialogues.jsonl")"
        local_post="$(local_rows "$DEST_DIR/data/glm-z1-9b/post_injection_runs.jsonl")"
        log "snapshot updated destination=$DEST_DIR local_rows complete=$local_complete post=$local_post"
    else
        log "snapshot pull failed or no files exist yet; previous snapshot preserved"
    fi

    if [[ "$done_marker" == "1" || "$stopped_reason" == "complete" ]]; then
        log "completion detected; waiting briefly for process/log flush before final pull"
        sleep 5
        if pull_snapshot; then
            log "final snapshot complete local_rows complete=$(local_rows "$DEST_DIR/data/glm-z1-9b/complete_dialogues.jsonl") post=$(local_rows "$DEST_DIR/data/glm-z1-9b/post_injection_runs.jsonl")"
            exit 0
        fi
        log "final pull failed; retrying next iteration"
    elif [[ "$seen_running" == "1" && "$tmux_running" == "0" && "$process_running" == "0" ]]; then
        log "run stopped without complete metadata/DONE marker; performing final diagnostic pull"
        pull_snapshot || true
        exit 3
    fi

    sleep "$SNAPSHOT_INTERVAL"
done

log "MAX_HOURS reached; performing final pull before exit"
pull_snapshot || true
log "timeout exit local_rows complete=$(local_rows "$DEST_DIR/data/glm-z1-9b/complete_dialogues.jsonl") post=$(local_rows "$DEST_DIR/data/glm-z1-9b/post_injection_runs.jsonl")"
exit 2
