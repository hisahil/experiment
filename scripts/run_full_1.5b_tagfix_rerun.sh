#!/usr/bin/env bash
# 1.5B tagfix rerun: GPU only (918 dialogues) + merge + judge + evaluate
set -eo pipefail
cd /root/msc_thesis_experiment
source .venv/bin/activate
set -a && source .env && set +a
export PYTHONUNBUFFERED=1
LOG=data/logs/full_1.5b_tagfix_rerun.log
mkdir -p data/logs data/1.5b results/1.5b
EC=0
{
  echo "=== TAGFIX RERUN $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

  echo "=== STEP 1 GPU (resume 918 missing) ==="
  python scripts/run_adaptive_dialogue.py --scope full_math \
    --generation-mode continuation --record-timings

  echo "=== STEP 2 line counts ==="
  wc -l data/1.5b/complete_dialogues.jsonl data/1.5b/post_injection_runs.jsonl

  echo "=== STEP 3 smoke gate ==="
  python scripts/smoke_intermediate_segments.py \
    --dialogues data/1.5b/complete_dialogues.jsonl \
    --max-empty-rate 0.005 --max-double-seps 0

  echo "=== STEP 4 merge rejudged (kept + fresh) ==="
  python scripts/merge_1.5b_post_rerun.py

  echo "=== STEP 5 judge batch ==="
  python scripts/judge_batch.py --only-fallbacks \
    --input data/1.5b/post_injection_runs.rejudged.jsonl \
    --out data/1.5b/post_injection_runs.rejudged.jsonl

  for pass in 1 2 3 4 5; do
    fb=$(python3 -c "
import json
fb=0
for line in open('data/1.5b/post_injection_runs.rejudged.jsonl'):
    if line.strip() and json.loads(line).get('judge_fallback'): fb+=1
print(fb)")
    echo "=== judge fallback pass $pass: $fb ==="
    if [ "$fb" = "0" ]; then break; fi
    python scripts/judge_batch.py --only-fallbacks \
      --input data/1.5b/post_injection_runs.rejudged.jsonl \
      --out data/1.5b/post_injection_runs.rejudged.jsonl
  done

  echo "=== STEP 6 evaluate ==="
  python scripts/evaluate.py \
    --input data/1.5b/post_injection_runs.rejudged.jsonl \
    --baselines data/baselines/1.5b/baseline_cots.jsonl \
    --output-dir results/1.5b \
    --require-judge-coverage

  echo "=== STEP 7 tag sanity ==="
  python3 -c "
import json, re
TAG = re.compile(r'</?\s*(?:reasoning|thinking|thought|think)\s*>', re.I)
TAG_ONLY = '</reasoning>'
dlg_tag=0; seg_tag_only=0; segs=0
for line in open('data/1.5b/complete_dialogues.jsonl'):
    d=json.loads(line)
    trace=d.get('accumulated_trace') or ''
    if TAG.search(trace): dlg_tag+=1
    for seg in d.get('trace_segments') or []:
        if seg.get('type')!='model' or seg.get('role')!='intermediate': continue
        segs+=1
        t=(seg.get('text') or '').strip()
        if t==TAG_ONLY: seg_tag_only+=1
print(f'dialogues_with_tag={dlg_tag} tag_only_intermediate={seg_tag_only}/{segs}')
"

  touch data/1.5b/PIPELINE_COMPLETE
  echo "=== TAGFIX_RERUN_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG" || EC=$?

if [[ "$EC" -ne 0 ]]; then
  echo "=== TAGFIX_RERUN_EXIT=$EC ===" | tee -a "$LOG"
  exit "$EC"
fi
