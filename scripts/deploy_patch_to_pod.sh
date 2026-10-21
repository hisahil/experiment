#!/usr/bin/env bash
# Deploy a small patch tarball to RunPod via base64 in a single ssh -tt heredoc.
# scp/rsync and `ssh -tt host "cmd"` both fail on the RunPod proxy; heredoc stdin works.
set -euo pipefail

POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="${1:-/tmp/deploy_patch.tar.gz}"

echo "==> Building patch if needed..."
if [[ ! -f "$PATCH" ]]; then
  tar czf "$PATCH" -C "$ROOT" \
    src/answer_judge.py src/adaptive_dialogue.py src/evaluator.py \
    src/grid_validation.py src/run_timings.py \
    src/baseline_index.py src/experiment_scope.py src/model_family.py \
    src/cot_segmentation.py src/derived_metrics.py src/math_utils.py \
    src/experiment_design.py src/llm_backend.py \
    scripts/run_adaptive_dialogue.py scripts/evaluate.py scripts/run_perturbed.py \
    scripts/convert_baselines_to_continuation.py \
    scripts/rejudge_fallbacks.py scripts/merge_runs.py \
    scripts/audit_transcripts.py scripts/sync_pod_live.sh scripts/validate_grid.py \
    scripts/smoke_judge.py scripts/run_pipeline.py scripts/smoke_cross_model.py \
    scripts/prepare_tasks.py scripts/generate_baselines.py \
    scripts/build_intersection_tasks.py scripts/build_contingent.py \
    scripts/run_task_selection_phase3_4.sh scripts/setup_gpu.sh \
    scripts/backfill_injection_fields.py scripts/smoke_intermediate_segments.py \
    scripts/stop_mac_sidecars.sh scripts/start_mac_sidecars.sh \
    scripts/generate_injections.py scripts/judge_batch.py \
    configs/injection_generation.yaml configs/off_topic_themes.yaml \
    configs/model_families.yaml prompts/baseline_generation_gemma.txt \
    scripts/run_mini_run_overnight.sh scripts/hourly_pod_snapshot.sh \
    scripts/deploy_mini_run_to_pod.sh scripts/run_cross_smokes_only.sh \
    configs/experiment.yaml configs/datasets.yaml
fi

B64="$(base64 < "$PATCH")"
echo "==> Uploading $(wc -c < "$PATCH" | tr -d ' ') bytes via ssh heredoc..."

ssh -tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=60 "$POD" <<EOF
base64 -d > /tmp/deploy_patch.tar.gz << 'B64END'
${B64}
B64END
tar xzf /tmp/deploy_patch.tar.gz -C /root/msc_thesis_experiment
rm -f /tmp/deploy_patch.tar.gz
grep stop_max_new_tokens /root/msc_thesis_experiment/configs/experiment.yaml | head -1
grep -q JudgeCreditError /root/msc_thesis_experiment/src/answer_judge.py && echo DEPLOY_OK
exit
EOF

echo "==> Done."
