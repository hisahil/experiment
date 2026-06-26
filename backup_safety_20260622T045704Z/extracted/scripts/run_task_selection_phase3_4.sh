#!/usr/bin/env bash
# Task-selection Phase 3 (14B+32B) and Phase 4 (intersection) — sequential overnight run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

LOG_DIR="data/task_selection/logs"
META_DIR="data/task_selection/meta"
OUT_DIR="data/task_selection/output"
TIMINGS_JSON="${LOG_DIR}/run_timings.json"
OVERNIGHT_LOG="${LOG_DIR}/overnight.log"

mkdir -p "$LOG_DIR" data/task_selection/baselines/14b data/task_selection/baselines/32b "$META_DIR" "$OUT_DIR"

exec > >(tee -a "$OVERNIGHT_LOG") 2>&1

iso_now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
secs_since() { python3 -c "import time; print(round(time.time() - float('$1'), 1))"; }

RUN_T0=$(python3 -c "import time; print(time.time())")
STARTED_AT=$(iso_now)
echo "=== task_selection phase 3+4 START $STARTED_AT ==="

BASELINE_7B="data/task_selection/baselines/7b/baseline_cots.jsonl"
POOL="data/task_selection/pool/candidates.jsonl"
if [[ ! -f "$BASELINE_7B" ]]; then
  echo "[FAIL] missing $BASELINE_7B"
  exit 1
fi
N7B=$(wc -l < "$BASELINE_7B" | tr -d ' ')
if [[ "$N7B" -lt 304 ]]; then
  echo "[FAIL] 7B baselines incomplete: $N7B/304 lines — aborting"
  exit 1
fi
echo "Preflight OK: 7B baselines=$N7B lines"

T_CONTINGENT=$(python3 -c "import time; print(time.time())")
python scripts/build_contingent.py
CONTINGENT_N=$(wc -l < "$META_DIR/contingent.jsonl" | tr -d ' ')
CONTINGENT_ELAPSED=$(secs_since "$T_CONTINGENT")
echo "Contingent built: $CONTINGENT_N tasks in ${CONTINGENT_ELAPSED}s"

T_14B=$(python3 -c "import time; print(time.time())")
python scripts/generate_baselines.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --tasks "$META_DIR/contingent.jsonl" \
  --output data/task_selection/baselines/14b/baseline_cots.jsonl \
  2>&1 | tee "$LOG_DIR/14b.log"
N14B=$(wc -l < data/task_selection/baselines/14b/baseline_cots.jsonl | tr -d ' ')
ELAPSED_14B=$(secs_since "$T_14B")
echo "14B done: $N14B lines in ${ELAPSED_14B}s"

T_32B=$(python3 -c "import time; print(time.time())")
python scripts/generate_baselines.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --tasks "$META_DIR/contingent.jsonl" \
  --output data/task_selection/baselines/32b/baseline_cots.jsonl \
  2>&1 | tee "$LOG_DIR/32b.log"
N32B=$(wc -l < data/task_selection/baselines/32b/baseline_cots.jsonl | tr -d ' ')
ELAPSED_32B=$(secs_since "$T_32B")
echo "32B done: $N32B lines in ${ELAPSED_32B}s"

T_INTERSECT=$(python3 -c "import time; print(time.time())")
python scripts/build_intersection_tasks.py \
  --pool "$POOL" \
  --baselines "$BASELINE_7B" \
              data/task_selection/baselines/14b/baseline_cots.jsonl \
              data/task_selection/baselines/32b/baseline_cots.jsonl \
  --out "$OUT_DIR/experiment_tasks.jsonl" \
  2>&1 | tee "$LOG_DIR/intersection.log"
cp "$OUT_DIR/experiment_tasks.jsonl" data/experiment_tasks.jsonl
FINAL_N=$(wc -l < "$OUT_DIR/experiment_tasks.jsonl" | tr -d ' ')
ELAPSED_INTERSECT=$(secs_since "$T_INTERSECT")
echo "Intersection done: $FINAL_N final tasks in ${ELAPSED_INTERSECT}s"

TOTAL_ELAPSED=$(secs_since "$RUN_T0")
ENDED_AT=$(iso_now)

python3 - "$TIMINGS_JSON" <<PY
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
run_t0 = float("$RUN_T0")
total = float("$TOTAL_ELAPSED")

def fmt(seconds: float) -> str:
    total_i = int(seconds)
    h, rem = divmod(total_i, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

payload = {
    "phase": "task_selection_3_4",
    "started_at": "$STARTED_AT",
    "ended_at": "$ENDED_AT",
    "started_at_unix": run_t0,
    "contingent_tasks": int("$CONTINGENT_N"),
    "final_tasks": int("$FINAL_N"),
    "baseline_lines": {"7b": int("$N7B"), "14b": int("$N14B"), "32b": int("$N32B")},
    "elapsed_s": {
        "contingent": float("$CONTINGENT_ELAPSED"),
        "14b": float("$ELAPSED_14B"),
        "32b": float("$ELAPSED_32B"),
        "intersection": float("$ELAPSED_INTERSECT"),
        "total": total,
    },
    "elapsed_fmt": {
        "14b": fmt(float("$ELAPSED_14B")),
        "32b": fmt(float("$ELAPSED_32B")),
        "total": fmt(total),
    },
    "elapsed_h": {
        "14b": round(float("$ELAPSED_14B") / 3600, 3),
        "32b": round(float("$ELAPSED_32B") / 3600, 3),
        "total": round(total / 3600, 3),
    },
    "logs": {
        "overnight": "$OVERNIGHT_LOG",
        "14b": "$LOG_DIR/14b.log",
        "32b": "$LOG_DIR/32b.log",
        "intersection": "$LOG_DIR/intersection.log",
    },
    "outputs": {
        "contingent": "$META_DIR/contingent.jsonl",
        "experiment_tasks": "$OUT_DIR/experiment_tasks.jsonl",
        "experiment_tasks_root": "data/experiment_tasks.jsonl",
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Timings written to {path}")
PY

echo "=== task_selection phase 3+4 DONE $ENDED_AT | total ${TOTAL_ELAPSED}s ==="
echo "Final tasks: $FINAL_N -> data/experiment_tasks.jsonl"
echo "Timings: $TIMINGS_JSON"
