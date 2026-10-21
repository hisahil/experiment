#!/usr/bin/env bash
# Deploy mini_run overnight: code patch + data + .env + start tmux on RunPod.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
TMUX_SESSION="${TMUX_SESSION:-mini_run}"

echo "==> Step 1: deploy code patch"
bash "$ROOT/scripts/deploy_patch_to_pod.sh"

echo "==> Step 2: upload data + .env"
DATA_TAR="/tmp/mini_run_data.tar.gz"
tar czf "$DATA_TAR" -C "$ROOT" \
  data/experiment_tasks.jsonl \
  data/baselines/7b/baseline_cots_continuation.jsonl

B64_DATA="$(base64 < "$DATA_TAR")"
ssh -tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 "$POD" <<EOF
mkdir -p /root/msc_thesis_experiment/data/baselines/7b
base64 -d > /tmp/mini_run_data.tar.gz << 'B64END'
${B64_DATA}
B64END
tar xzf /tmp/mini_run_data.tar.gz -C /root/msc_thesis_experiment
rm -f /tmp/mini_run_data.tar.gz
wc -l /root/msc_thesis_experiment/data/experiment_tasks.jsonl
wc -l /root/msc_thesis_experiment/data/baselines/7b/baseline_cots_continuation.jsonl
exit
EOF

if [[ -f "$ROOT/.env" ]]; then
  ENV_B64="$(grep -E '^ANTHROPIC_API_KEY=' "$ROOT/.env" | base64 | tr -d '\n')"
  ssh -tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 "$POD" <<EOF
base64 -d << 'EEND' | tee -a /root/msc_thesis_experiment/.env > /dev/null
${ENV_B64}
EEND
grep -q ANTHROPIC_API_KEY /root/msc_thesis_experiment/.env && echo ENV_OK
exit
EOF
else
  echo "[WARN] no local .env — pod must already have ANTHROPIC_API_KEY"
fi

echo "==> Step 3: start overnight run (tmux or nohup fallback)"
ssh -tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 "$POD" <<'EOF'
apt-get update -qq && apt-get install -y -qq tmux 2>/dev/null || true
cd /root/msc_thesis_experiment
mkdir -p data/logs
chmod +x scripts/run_mini_run_overnight.sh scripts/setup_gpu.sh 2>/dev/null || true
pkill -f run_mini_run_overnight.sh 2>/dev/null || true
START_WRAPPER="data/logs/mini_run_start.sh"
cat > "$START_WRAPPER" << 'WRAP'
#!/usr/bin/env bash
set -euo pipefail
cd /root/msc_thesis_experiment
export PYTHONUNBUFFERED=1
if [[ ! -x .venv/bin/python ]]; then
  echo "=== setup_gpu START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  bash scripts/setup_gpu.sh >> data/logs/setup_gpu.log 2>&1
  echo "=== setup_gpu DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
fi
exec bash scripts/run_mini_run_overnight.sh
WRAP
chmod +x "$START_WRAPPER"
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t mini_run 2>/dev/null; then tmux kill-session -t mini_run; fi
  tmux new-session -d -s mini_run "bash $START_WRAPPER 2>&1 | tee -a data/logs/mini_run_overnight.log; exec bash"
  tmux has-session -t mini_run && echo TMUX_STARTED
else
  nohup bash "$START_WRAPPER" >> data/logs/mini_run_overnight.log 2>&1 &
  echo NOHUP_PID=$!
  sleep 1
  pgrep -af mini_run_start.sh || pgrep -af run_mini_run_overnight || echo START_CHECK_FAILED
fi
exit
EOF

echo "==> Deploy complete. Monitor: ssh -i $KEY $POD 'tail -f /root/msc_thesis_experiment/data/logs/mini_run_overnight.log'"
