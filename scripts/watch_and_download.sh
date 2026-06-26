#!/usr/bin/env bash
# Poll the pod every INTERVAL seconds for run_timings.json; when present,
# download the final results and notify the user (macOS).
set -euo pipefail

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${INTERVAL:-180}"   # seconds between polls
MARKER="/root/msc_thesis_experiment/data/task_selection/logs/run_timings.json"
LOGFILE="${ROOT}/watch_and_download.log"

echo "==> Watcher started at $(date -u +%FT%TZ). Polling every ${INTERVAL}s." | tee -a "$LOGFILE"

while true; do
  STATUS="$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 \
              -o LogLevel=ERROR "$POD" \
              "[ -f $MARKER ] && cat $MARKER | tr -d '\n' || echo NOTYET" 2>>"$LOGFILE")"

  if [[ "$STATUS" == NOTYET* ]]; then
    echo "[$(date +%H:%M:%S)] still running..." | tee -a "$LOGFILE"
    sleep "$INTERVAL"
    continue
  fi

  echo "[$(date +%H:%M:%S)] timings.json present — downloading final results" | tee -a "$LOGFILE"
  bash "$ROOT/scripts/pull_pod_results.sh" final 2>&1 | tee -a "$LOGFILE"

  # macOS notification + spoken alert
  if command -v osascript >/dev/null 2>&1; then
    osascript -e 'display notification "Task Selection fertig — Ergebnisse heruntergeladen." with title "MSc Thesis" sound name "Glass"' || true
  fi
  if command -v say >/dev/null 2>&1; then
    say "Task Selection ist fertig. Ergebnisse heruntergeladen." || true
  fi

  echo "==> Done." | tee -a "$LOGFILE"
  break
done
