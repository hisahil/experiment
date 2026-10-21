#!/usr/bin/env bash
# P7 for GLM-Z1-9B on RunPod: judge_batch → fallback retries → evaluate.
# Mirrors DeepSeek full_*_pipeline judge pattern; writes only under glm-z1-9b paths.
set -eo pipefail
cd /root/msc_thesis_experiment
source .venv/bin/activate
set -a && source .env && set +a
export PYTHONUNBUFFERED=1
export ANTHROPIC_PROMPT_CACHE=1
mkdir -p logs data/glm-z1-9b results/glm-z1-9b

# Preserve Script-3 timings (evaluate does not overwrite, but keep backups anyway)
cp -n results/glm-z1-9b/run_metadata.json results/glm-z1-9b/run_metadata.script3.json 2>/dev/null || true
cp -n results/glm-z1-9b/run_timings.json results/glm-z1-9b/run_timings.script3.json 2>/dev/null || true

STAMP=$(date +%Y%m%d_%H%M)
LOG=logs/glm_judge_${STAMP}.log
EC=0

count_fb() {
  local f="$1"
  .venv/bin/python - "$f" <<'PY'
import json, sys
path = sys.argv[1]
n = fb = 0
with open(path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        n += 1
        if json.loads(line).get("judge_fallback"):
            fb += 1
print(f"{fb}/{n}")
PY
}

{
  echo "=== P7 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "LOG=$LOG"
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY set: yes"
  else
    echo "ANTHROPIC_API_KEY set: NO"
    exit 2
  fi
  echo "input rows: $(wc -l < data/glm-z1-9b/post_injection_runs.jsonl | tr -d ' ')"
  echo "baselines: $(wc -l < data/baselines/glm-z1-9b/baseline_cots.jsonl | tr -d ' ')"

  echo "=== STEP 1 judge_batch (all fallbacks) ==="
  T0=$(date +%s)
  python scripts/judge_batch.py --only-fallbacks \
    --config configs/experiment.yaml \
    --input data/glm-z1-9b/post_injection_runs.jsonl \
    --tasks data/experiment_tasks.jsonl \
    --out data/glm-z1-9b/post_injection_runs.rejudged.jsonl
  echo "STEP1_ELAPSED_S=$(( $(date +%s) - T0 ))"
  echo "fallbacks_after_step1=$(count_fb data/glm-z1-9b/post_injection_runs.rejudged.jsonl)"

  echo "=== STEP 2 fallback retries (max 5) ==="
  for i in 1 2 3 4 5; do
    FB=$(count_fb data/glm-z1-9b/post_injection_runs.rejudged.jsonl | cut -d/ -f1)
    echo "retry_pass=$i fallbacks=$FB"
    if [ "$FB" = "0" ]; then
      echo "0 fallbacks — coverage complete"
      break
    fi
    T1=$(date +%s)
    python scripts/judge_batch.py --only-fallbacks \
      --config configs/experiment.yaml \
      --input data/glm-z1-9b/post_injection_runs.rejudged.jsonl \
      --tasks data/experiment_tasks.jsonl \
      --out data/glm-z1-9b/post_injection_runs.rejudged.jsonl
    echo "RETRY${i}_ELAPSED_S=$(( $(date +%s) - T1 ))"
    echo "fallbacks_after_retry${i}=$(count_fb data/glm-z1-9b/post_injection_runs.rejudged.jsonl)"
  done

  echo "=== STEP 3 evaluate ==="
  T2=$(date +%s)
  python scripts/evaluate.py \
    --config configs/experiment.yaml \
    --input data/glm-z1-9b/post_injection_runs.rejudged.jsonl \
    --baselines data/baselines/glm-z1-9b/baseline_cots.jsonl \
    --output-dir results/glm-z1-9b \
    --require-judge-coverage
  echo "EVAL_ELAPSED_S=$(( $(date +%s) - T2 ))"

  touch data/glm-z1-9b/PIPELINE_COMPLETE
  echo "=== P7 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT=0 ==="
} 2>&1 | tee -a "$LOG" || EC=$?

if [ "$EC" -ne 0 ]; then
  echo "=== P7 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT=$EC ===" | tee -a "$LOG"
fi
exit "$EC"
