#!/usr/bin/env bash
# Poll RunPod until task-selection finishes, then download all results to local
# backup_task_selection_<UTC>/ via the base64-over-ssh-heredoc trick (RunPod proxy
# blocks scp/rsync). Idempotent and safe to run multiple times.
set -euo pipefail

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLL_INTERVAL="${POLL_INTERVAL:-180}"     # seconds between checks (default 3 min)
POLL_MAX="${POLL_MAX:-7200}"              # total seconds before giving up (default 2 h)
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${DEST:-$ROOT/backup_task_selection_${STAMP}}"
LOG="${LOG:-$ROOT/backup_task_selection_${STAMP}.log}"

mkdir -p "$DEST"
exec > >(tee -a "$LOG") 2>&1

echo "=== auto_download_task_selection start $(date -u +%FT%TZ) ==="
echo "Pod: $POD"
echo "Dest: $DEST"
echo "Poll every ${POLL_INTERVAL}s, max ${POLL_MAX}s"

remote_status() {
  ssh -tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 \
      -o LogLevel=ERROR "$POD" <<'RS' 2>/dev/null | tr -d '\r' | grep -E '^(DONE|RUNNING|MISSING)'
cd /root/msc_thesis_experiment
if [[ -f data/task_selection/logs/run_timings.json ]]; then
  echo "DONE $(wc -l < data/task_selection/output/experiment_tasks.jsonl 2>/dev/null || echo 0)"
elif tmux has-session -t ts34 2>/dev/null; then
  PROG=$(grep -a 'progress:' data/task_selection/logs/32b.log 2>/dev/null | tail -1 | tr -d '\n')
  echo "RUNNING ${PROG:-no_progress}"
else
  echo "MISSING tmux gone, timings absent"
fi
exit
RS
}

elapsed=0
while :; do
  status="$(remote_status || true)"
  echo "[$(date -u +%T)Z] $status"
  case "$status" in
    DONE*) echo "Run completed — starting download."; break ;;
    MISSING*) echo "[FAIL] tmux gone and no timings file. Check pod manually."; exit 2 ;;
  esac
  if (( elapsed >= POLL_MAX )); then
    echo "[FAIL] poll timeout (${POLL_MAX}s) — run not finished."
    exit 3
  fi
  sleep "$POLL_INTERVAL"
  elapsed=$((elapsed + POLL_INTERVAL))
done

echo "==> Packing remote results into tarball..."
ssh -tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 -o LogLevel=ERROR "$POD" <<'PACK'
cd /root/msc_thesis_experiment
tar czf /tmp/task_selection_results.tar.gz \
  data/task_selection \
  data/experiment_tasks.jsonl 2>/dev/null || true
ls -la /tmp/task_selection_results.tar.gz
exit
PACK

echo "==> Streaming tarball down via base64 ssh heredoc..."
ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 -o LogLevel=ERROR "$POD" \
  'base64 < /tmp/task_selection_results.tar.gz' \
  | base64 -d > "$DEST/task_selection_results.tar.gz"

echo "==> Local tarball:"
ls -la "$DEST/task_selection_results.tar.gz"

echo "==> Extracting..."
mkdir -p "$DEST/extracted"
tar xzf "$DEST/task_selection_results.tar.gz" -C "$DEST/extracted"

echo "==> Summary:"
FINAL="$DEST/extracted/data/experiment_tasks.jsonl"
TIMINGS="$DEST/extracted/data/task_selection/logs/run_timings.json"
if [[ -f "$FINAL" ]]; then
  echo "  experiment_tasks.jsonl: $(wc -l < "$FINAL") lines"
fi
if [[ -f "$TIMINGS" ]]; then
  echo "  run_timings.json:"
  cat "$TIMINGS" | sed 's/^/    /'
fi

echo "==> Cleaning up pod tarball..."
ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 -o LogLevel=ERROR "$POD" \
  'rm -f /tmp/task_selection_results.tar.gz' || true

echo "=== auto_download_task_selection done $(date -u +%FT%TZ) ==="
echo "Backup: $DEST"
echo "Log:    $LOG"
