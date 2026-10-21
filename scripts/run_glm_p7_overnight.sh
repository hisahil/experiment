#!/usr/bin/env bash
# GLM P7: Judge → Rejudge-until-0 → Evaluate (overnight-safe)
set -uo pipefail
REPO=/root/msc_thesis_experiment
cd "$REPO"
mkdir -p logs results/glm-z1-9b data/logs
LOG="logs/glm_p7_$(date -u +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== GLM P7 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "log=$LOG host=$(hostname)"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
export PYTHONUNBUFFERED=1 ANTHROPIC_PROMPT_CACHE=1

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "[FATAL] ANTHROPIC_API_KEY missing"
  exit 1
fi

IN=data/glm-z1-9b/post_injection_runs.jsonl
OUT=data/glm-z1-9b/post_injection_runs.rejudged.jsonl
BASE=data/baselines/glm-z1-9b/baseline_cots.jsonl
TASKS=data/experiment_tasks.jsonl
RES=results/glm-z1-9b

n=$(wc -l < "$IN" | tr -d ' ')
echo "input_rows=$n"
if [ "$n" != "10800" ]; then
  echo "[FATAL] expected 10800 rows, got $n — aborting (will not judge wrong file)"
  exit 1
fi

echo "=== P7.1 judge_batch (full) ==="
python scripts/judge_batch.py \
  --config configs/experiment.yaml \
  --input "$IN" \
  --tasks "$TASKS" \
  --out "$OUT"
echo "judge_batch exit=$?"

count_fb() {
  python3 -c "
import json
fb=0
for line in open('$OUT'):
    if line.strip() and json.loads(line).get('judge_fallback') is True:
        fb+=1
print(fb)
"
}

echo "=== P7.2 rejudge fallbacks until 0 ==="
for pass in 1 2 3 4 5 6 7 8; do
  fb=$(count_fb)
  echo "pass=$pass fallbacks=$fb $(date -u +%H:%M:%SZ)"
  if [ "$fb" = "0" ]; then
    echo "fallbacks cleared"
    break
  fi
  python scripts/judge_batch.py \
    --config configs/experiment.yaml \
    --only-fallbacks \
    --input "$OUT" \
    --tasks "$TASKS" \
    --out "$OUT"
  echo "rejudge pass=$pass exit=$?"
done

fb=$(count_fb)
echo "final_fallbacks=$fb"
if [ "$fb" != "0" ]; then
  echo "[WARN] still $fb fallbacks after loops — evaluate may fail coverage gate"
fi

echo "=== P7.3 evaluate ==="
python scripts/evaluate.py \
  --config configs/experiment.yaml \
  --input "$OUT" \
  --baselines "$BASE" \
  --output-dir "$RES" \
  --require-judge-coverage
ev=$?
echo "evaluate exit=$ev"

if [ $ev -eq 0 ] && [ "$fb" = "0" ]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > data/glm-z1-9b/P7_COMPLETE
  echo "=== GLM P7 COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a data/glm-z1-9b/P7_COMPLETE
else
  echo "=== GLM P7 FINISHED WITH ISSUES exit=$ev fb=$fb $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "INCOMPLETE ev=$ev fb=$fb $(date -u +%Y-%m-%dT%H:%M:%SZ)" > data/glm-z1-9b/P7_INCOMPLETE
fi
