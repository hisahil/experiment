#!/usr/bin/env bash
# Overnight mini_run_3seed: Script 2 (batch) -> Script 3b -> evaluate.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
OVERNIGHT_LOG="${LOG_DIR}/mini_run_overnight.log"
TIMINGS_JSON="${LOG_DIR}/mini_run_run_timings.json"
INJECTIONS_OUT="data/injections_generated.jsonl"
DIALOGUES_OUT="data/post_injection_runs.jsonl"
EXPECTED_INJECTIONS=180
EXPECTED_DIALOGUES=540
SCOPE="mini_run_3seed"
BASELINE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PYTHON="${PYTHON:-.venv/bin/python}"
SCRIPT2_MODE="batch"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec > >(tee -a "$OVERNIGHT_LOG") 2>&1

iso_now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
secs_since() { python3 -c "import time; print(round(time.time() - float('$1'), 1))"; }

RUN_T0=$(python3 -c "import time; print(time.time())")
STARTED_AT=$(iso_now)
echo "=== mini_run overnight START $STARTED_AT ==="

T_PREFLIGHT=$(python3 -c "import time; print(time.time())")
fail() { echo "[FAIL] $*"; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || fail "nvidia-smi query failed"

[[ -x "$PYTHON" ]] || fail "missing $PYTHON — run scripts/setup_gpu.sh first"
"$PYTHON" -c "import anthropic, vllm" 2>/dev/null || fail "anthropic/vllm not importable in venv"

[[ -n "${ANTHROPIC_API_KEY:-}" ]] || fail "ANTHROPIC_API_KEY not set (.env or env)"

[[ -f data/experiment_tasks.jsonl ]] || fail "missing data/experiment_tasks.jsonl"
TASK_N=$(wc -l < data/experiment_tasks.jsonl | tr -d ' ')
[[ "$TASK_N" -ge 10 ]] || fail "experiment_tasks.jsonl has $TASK_N lines (need >=10)"

[[ -f data/baselines/7b/baseline_cots_continuation.jsonl ]] || fail "missing 7B continuation baselines"
BASE_N=$(wc -l < data/baselines/7b/baseline_cots_continuation.jsonl | tr -d ' ')
[[ "$BASE_N" -ge 10 ]] || fail "7B baselines have $BASE_N lines (need >=10)"

PREFLIGHT_ELAPSED=$(secs_since "$T_PREFLIGHT")
echo "Preflight OK: tasks=$TASK_N baselines=$BASE_N (${PREFLIGHT_ELAPSED}s)"

if [[ -f "$INJECTIONS_OUT" ]]; then
  BK="data/injections_generated_pre_mini_run_$(date +%Y%m%d_%H%M%S).jsonl"
  cp "$INJECTIONS_OUT" "$BK"
  echo "Backed up previous injections -> $BK"
fi

# Script 2 must target the EXACT same tasks Script 3b's scope-based select_tasks()
# will pick (seeded shuffle over eligible-baseline tasks) — NOT a naive first-N
# slice of experiment_tasks.jsonl. Those two selections can diverge (different
# task_ids) once more than `max_per_domain` tasks have verified baselines, which
# desyncs Script 2's injections from what Script 3b's preflight actually needs.
# Materialize the scope's task subset once, up front, and feed it to Script 2.
SCOPE_TASKS_FILE="${LOG_DIR}/mini_run_scope_tasks.jsonl"
"$PYTHON" - "$SCOPE" "$BASELINE_MODEL" "$SCOPE_TASKS_FILE" <<'PYEOF'
import sys
sys.path.insert(0, "src")
import json
import yaml
from experiment_scope import get_scope, select_tasks
from baseline_index import index_verified_baselines

scope_name, model_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open("configs/experiment.yaml"))
scope = get_scope(cfg, scope_name)
experiment_seed = int(cfg.get("experiment", {}).get("seed", 42))

baseline_records = [json.loads(l) for l in open("data/baselines/7b/baseline_cots_continuation.jsonl") if l.strip()]
baseline_index = index_verified_baselines(baseline_records, model_name)

all_tasks = [json.loads(l) for l in open("data/experiment_tasks.jsonl") if l.strip()]
all_tasks = [t for t in all_tasks if t.get("domain") == "math"]
task_by_id = {t["task_id"]: t for t in all_tasks}
eligible = [task_by_id[tid] for tid in sorted(baseline_index.keys()) if tid in task_by_id]

tasks = select_tasks(eligible, scope, experiment_seed)
with open(out_path, "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"[scope-tasks] wrote {len(tasks)} tasks for scope={scope_name!r} -> {out_path}")
PYEOF

T_S2=$(python3 -c "import time; print(time.time())")
S2_LOG="${LOG_DIR}/script2_batch.log"
set +e
"$PYTHON" scripts/generate_injections.py \
  --batch \
  --baselines data/baselines/7b/baseline_cots_continuation.jsonl \
  --baseline-model "$BASELINE_MODEL" \
  --tasks "$SCOPE_TASKS_FILE" \
  --output "$INJECTIONS_OUT" \
  --domain math \
  --k-values 1 2 3 \
  --error-types off_topic paraphrased_shift false_consensus \
  --length-levels short long \
  2>&1 | tee "$S2_LOG"
S2_RC=${PIPESTATUS[0]}
set -e

if [[ "$S2_RC" -ne 0 ]]; then
  echo "[WARN] batch failed (rc=$S2_RC) — falling back to synchronous independent path"
  SCRIPT2_MODE="sync_fallback"
  T_S2=$(python3 -c "import time; print(time.time())")
  "$PYTHON" scripts/generate_injections.py \
    --injection-strategy independent \
    --baselines data/baselines/7b/baseline_cots_continuation.jsonl \
    --baseline-model "$BASELINE_MODEL" \
    --tasks "$SCOPE_TASKS_FILE" \
    --output "$INJECTIONS_OUT" \
    --domain math \
    --k-values 1 2 3 \
    --error-types off_topic paraphrased_shift false_consensus \
    --length-levels short long \
    2>&1 | tee -a "$S2_LOG"
fi
S2_ELAPSED=$(secs_since "$T_S2")

N_INJ=$(wc -l < "$INJECTIONS_OUT" | tr -d ' ')
if [[ "$N_INJ" -ne "$EXPECTED_INJECTIONS" ]]; then
  fail "injections gate: $N_INJ != $EXPECTED_INJECTIONS"
fi
echo "Script 2 gate OK: $N_INJ injection records (${S2_ELAPSED}s)"

T_S3=$(python3 -c "import time; print(time.time())")
"$PYTHON" scripts/run_adaptive_dialogue.py \
  --scope "$SCOPE" \
  --generation-mode continuation \
  --domain math --limit 10 \
  --record-timings
S3_ELAPSED=$(secs_since "$T_S3")

N_DIAL=$(wc -l < "$DIALOGUES_OUT" | tr -d ' ')
if [[ "$N_DIAL" -ne "$EXPECTED_DIALOGUES" ]]; then
  fail "dialogues gate: $N_DIAL != $EXPECTED_DIALOGUES"
fi
echo "Script 3b gate OK: $N_DIAL dialogues (${S3_ELAPSED}s)"

T_S4=$(python3 -c "import time; print(time.time())")
"$PYTHON" scripts/evaluate.py --config configs/experiment.yaml
S4_ELAPSED=$(secs_since "$T_S4")
echo "Script 4 done (${S4_ELAPSED}s)"

TOTAL_ELAPSED=$(secs_since "$RUN_T0")
ENDED_AT=$(iso_now)

export TIMINGS_JSON STARTED_AT ENDED_AT RUN_T0 SCOPE BASELINE_MODEL SCRIPT2_MODE
export PREFLIGHT_ELAPSED S2_ELAPSED S3_ELAPSED S4_ELAPSED TOTAL_ELAPSED
export N_INJ N_DIAL EXPECTED_INJECTIONS EXPECTED_DIALOGUES S2_LOG OVERNIGHT_LOG
"$PYTHON" - <<'PY'
import json
import os
import re
from pathlib import Path

def fmt(seconds: float) -> str:
    total_i = int(seconds)
    h, rem = divmod(total_i, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

s2_log = Path(os.environ["S2_LOG"])
log_text = s2_log.read_text(encoding="utf-8") if s2_log.is_file() else ""
batch_id = None
m = re.search(r"Submitted injection batch (msgbatch_[A-Za-z0-9_-]+)", log_text)
if m:
    batch_id = m.group(1)
batch_failed = 0
m2 = re.search(r"'failed':\s*(\d+)", log_text)
if m2:
    batch_failed = int(m2.group(1))
batch_requests = 60 if "Submitted injection batch" in log_text else 0

script3_detail = {}
run_timings = Path("results/run_timings.json")
if run_timings.is_file():
    script3_detail = json.loads(run_timings.read_text(encoding="utf-8"))

payload = {
    "run": "mini_run_overnight",
    "scope": os.environ["SCOPE"],
    "model": os.environ["BASELINE_MODEL"],
    "started_at": os.environ["STARTED_AT"],
    "ended_at": os.environ["ENDED_AT"],
    "script2_mode": os.environ.get("SCRIPT2_MODE", "batch"),
    "script2_batch_id": batch_id,
    "elapsed_s": {
        "preflight": float(os.environ["PREFLIGHT_ELAPSED"]),
        "script2_injections": float(os.environ["S2_ELAPSED"]),
        "script3b_gpu": float(os.environ["S3_ELAPSED"]),
        "evaluate": float(os.environ["S4_ELAPSED"]),
        "total": float(os.environ["TOTAL_ELAPSED"]),
    },
    "elapsed_fmt": {
        "script2_injections": fmt(float(os.environ["S2_ELAPSED"])),
        "script3b_gpu": fmt(float(os.environ["S3_ELAPSED"])),
        "evaluate": fmt(float(os.environ["S4_ELAPSED"])),
        "total": fmt(float(os.environ["TOTAL_ELAPSED"])),
    },
    "elapsed_h": {
        "script3b_gpu": round(float(os.environ["S3_ELAPSED"]) / 3600, 3),
        "total": round(float(os.environ["TOTAL_ELAPSED"]) / 3600, 3),
    },
    "counts": {
        "injection_records": int(os.environ["N_INJ"]),
        "batch_requests": batch_requests,
        "batch_failed": batch_failed,
        "dialogues_planned": int(os.environ["EXPECTED_DIALOGUES"]),
        "dialogues_completed": int(os.environ["N_DIAL"]),
    },
    "logs": {
        "overnight": os.environ["OVERNIGHT_LOG"],
        "script2_batch": os.environ["S2_LOG"],
    },
    "script3_detail": script3_detail,
}
path = Path(os.environ["TIMINGS_JSON"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Timings written to {path}")
PY

echo "=== mini_run DONE | total ${TOTAL_ELAPSED}s | timings: $TIMINGS_JSON ==="
