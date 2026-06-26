#!/usr/bin/env bash
# Pull task-selection results from RunPod to local Mac via base64+heredoc
# (scp/rsync don't work through the RunPod proxy; tar via ssh stdout does).
set -euo pipefail

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-snapshot}"   # "snapshot" or "final"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST_DIR="${ROOT}/backup_pod_${TAG}_${STAMP}"

mkdir -p "$DEST_DIR"
echo "==> Destination: $DEST_DIR"

# Build a tar of the result dirs ON THE POD and stream it via ssh as base64.
B64="$(
  ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 -o LogLevel=ERROR "$POD" \
    "cd /root/msc_thesis_experiment && tar czf - \
       data/task_selection/baselines \
       data/task_selection/meta \
       data/task_selection/logs \
       data/task_selection/output \
       data/task_selection/pool \
       \$( [ -f data/experiment_tasks.jsonl ] && echo data/experiment_tasks.jsonl ) \
       2>/dev/null | base64"
)"

echo "==> Got $(echo -n "$B64" | wc -c | tr -d ' ') base64 bytes"
echo "$B64" | base64 -d > "${DEST_DIR}/results.tar.gz"
tar xzf "${DEST_DIR}/results.tar.gz" -C "$DEST_DIR"
rm "${DEST_DIR}/results.tar.gz"

echo "==> Extracted to $DEST_DIR"
ls "${DEST_DIR}/data/task_selection/baselines/" 2>/dev/null || true
if [[ -f "${DEST_DIR}/data/experiment_tasks.jsonl" ]]; then
  echo "    experiment_tasks.jsonl: $(wc -l < "${DEST_DIR}/data/experiment_tasks.jsonl" | tr -d ' ') lines"
fi
