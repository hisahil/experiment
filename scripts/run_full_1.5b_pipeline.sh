#!/usr/bin/env bash
set -eo pipefail
cd /root/msc_thesis_experiment
source .venv/bin/activate
set -a && source .env && set +a
export PYTHONUNBUFFERED=1
LOG=data/logs/full_1.5b_pipeline.log
mkdir -p data/logs data/1.5b results/1.5b
EC=0
{
  echo "=== STEP 0 preflight ==="
  python scripts/preflight_gpu.py --scope full_math --balanced-per-domain 100 || true

  echo "=== STEP 1 GPU 1.5B run ==="
  python scripts/run_adaptive_dialogue.py --scope full_math \
    --generation-mode continuation --record-timings

  echo "=== STEP 2 smoke gate ==="
  python scripts/smoke_intermediate_segments.py \
    --dialogues data/1.5b/complete_dialogues.jsonl \
    --max-empty-rate 0.005 --max-double-seps 0

  echo "=== STEP 3 judge batch ==="
  python scripts/judge_batch.py --only-fallbacks \
    --input data/1.5b/post_injection_runs.jsonl \
    --out data/1.5b/post_injection_runs.rejudged.jsonl

  echo "=== STEP 4 evaluate ==="
  python scripts/evaluate.py \
    --input data/1.5b/post_injection_runs.rejudged.jsonl \
    --baselines data/baselines/1.5b/baseline_cots.jsonl \
    --output-dir results/1.5b \
    --require-judge-coverage

  touch data/1.5b/PIPELINE_COMPLETE
  echo "=== PIPELINE_EXIT=0 ==="
} 2>&1 | tee -a "$LOG" || EC=$?

if [[ "$EC" -ne 0 ]]; then
  echo "=== PIPELINE_EXIT=$EC ===" | tee -a "$LOG"
  exit "$EC"
fi
