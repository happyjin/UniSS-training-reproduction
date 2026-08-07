#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="${EVAL_ROOT}/../scripts"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

RUN_ID="${RUN_ID:-fixed_chunk_stage_a_v2_vs_stage_b_v3_v1}"
LOG_ROOT="${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v6/${RUN_ID}"
REPORT_ROOT="${REPO_ROOT}/reports/uniss_phase3_whisper_streamspeech_joint_v6/${RUN_ID}"
STAGE_A_CHECKPOINT="${STAGE_A_CHECKPOINT:-${REPO_ROOT}/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/phase3_joint_v6_stage_a_heads_only_15shard_v2}"
STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:-${REPO_ROOT}/checkpoints/uniss_phase3_whisper_streamspeech_joint_v6/phase3_joint_v6_stage_b_guarded_joint_15shard_v3}"
CHUNKS=(320 640 960 1280 offline)

refuse_existing "${LOG_ROOT}" "${REPORT_ROOT}"
require_dir "${STAGE_A_CHECKPOINT}"
require_dir "${STAGE_B_CHECKPOINT}"
mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"

for model_label in stage_a stage_b; do
  if [[ "${model_label}" == "stage_a" ]]; then
    checkpoint="${STAGE_A_CHECKPOINT}"
  else
    checkpoint="${STAGE_B_CHECKPOINT}"
  fi
  for chunk in "${CHUNKS[@]}"; do
    echo "Evaluating ${model_label} at fixed chunk=${chunk}"
    CHECKPOINT_DIR="${checkpoint}" \
    MODEL_LABEL="${model_label}" \
    CHUNK="${chunk}" \
    OUTPUT_LOG="${LOG_ROOT}/${model_label}_${chunk}.log" \
    MASTER_PORT="${MASTER_PORT:-29765}" \
      bash "${EVAL_ROOT}/run_fixed_chunk_eval_8gpu.sh"
  done
done

"${PYTHON}" -m \
  experiments.uniss_phase3_whisper_streamspeech_joint_v6.evaluation.summarize_fixed_chunk_eval \
  --log-root "${LOG_ROOT}" \
  --stage-a-checkpoint "${STAGE_A_CHECKPOINT}" \
  --stage-b-checkpoint "${STAGE_B_CHECKPOINT}" \
  --output-json "${REPORT_ROOT}/metrics.json" \
  --output-md "${REPORT_ROOT}/report.md"

echo "Fixed-chunk report: ${REPORT_ROOT}/report.md"
