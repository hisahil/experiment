#!/usr/bin/env bash
# Phase B: first-boxed stop extraction + rejudge + evaluate (all 4 models, no GPU)
set -eo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
set -a && source .env && set +a
export PYTHONUNBUFFERED=1 ANTHROPIC_PROMPT_CACHE=1

echo "=== Phase B1: reextract stop answers (first-boxed) ==="
python scripts/reextract_stop_answers.py --models 1.5b 7b 14b 32b

rejudge_model() {
  local input="$1"
  echo "=== Rejudge fallbacks: $input ==="
  for pass in 1 2 3 4 5; do
    fb=$(python3 -c "
import json
fb=0
for line in open('$input'):
    if line.strip() and json.loads(line).get('judge_fallback'): fb+=1
print(fb)")
    echo "  pass $pass: fallbacks=$fb"
    if [ "$fb" = "0" ]; then break; fi
    python scripts/judge_batch.py --only-fallbacks --input "$input" --out "$input"
  done
}

rejudge_model data/1.5b/post_injection_runs.rejudged.jsonl
rejudge_model data/post_injection_runs.rejudged.jsonl
rejudge_model data/14b/post_injection_runs.rejudged.jsonl
rejudge_model data/32b/post_injection_runs.rejudged.jsonl

echo "=== Phase B2: evaluate all models ==="
python scripts/evaluate.py --input data/1.5b/post_injection_runs.rejudged.jsonl \
  --baselines data/baselines/1.5b/baseline_cots.jsonl --output-dir results/1.5b --require-judge-coverage
python scripts/evaluate.py --input data/post_injection_runs.rejudged.jsonl \
  --baselines data/baselines/7b/baseline_cots.jsonl --output-dir results --require-judge-coverage
python scripts/evaluate.py --input data/14b/post_injection_runs.rejudged.jsonl \
  --baselines data/baselines/14b/baseline_cots.jsonl --output-dir results/14b --require-judge-coverage
python scripts/evaluate.py --input data/32b/post_injection_runs.rejudged.jsonl \
  --baselines data/baselines/32b/baseline_cots.jsonl --output-dir results/32b --require-judge-coverage

echo "=== Phase B done ==="
