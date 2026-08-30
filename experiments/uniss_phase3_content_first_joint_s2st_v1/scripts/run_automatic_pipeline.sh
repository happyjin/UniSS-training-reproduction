#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"
FORMAL_RUN=uniss_phase3_content_first_joint_s2st_v1_formal1e_v1
SFT_ROOT=${REPO_ROOT}/checkpoints/${FORMAL_RUN}
SFT_ITER=${SFT_ROOT}/iter_0000717
FREE_EVAL_SESSION=${FORMAL_RUN}_free_eval
FREE_EVAL_COMPLETE=${REPO_ROOT}/reports/${FORMAL_RUN}/checkpoint_evaluation/free_running_content_first_v1/iter_0000717/complete.json
PIPELINE_ROOT=${REPO_ROOT}/eval_outputs/uniss_phase3_content_first_joint_s2st_v1
REPORT_DIR=${REPO_ROOT}/reports/uniss_phase3_content_first_joint_s2st_v1/coverage_grpo_final_v1
mkdir -p "${PIPELINE_ROOT}" "${REPO_ROOT}/logs/uniss_phase3_content_first_joint_s2st_v1"
exec 9>"${PIPELINE_ROOT}/automatic_pipeline.lock"
flock -n 9 || { echo "automatic pipeline already owns the lock" >&2; exit 2; }

pipeline_complete=0
restore_holder_after_exit() {
  local status=$?
  if (( pipeline_complete == 0 )); then
    echo "automatic pipeline stopped before completion (exit=${status}); scheduling GPU holder recovery" >&2
    bash "${HERE}/scripts/schedule_gpu_holder.sh" || true
  fi
}
trap restore_holder_after_exit EXIT

while tmux has-session -t "${FREE_EVAL_SESSION}" 2>/dev/null; do sleep 10; done
[[ -f "${FREE_EVAL_COMPLETE}" ]] || { echo "free-running evaluation ended without complete.json" >&2; exit 3; }

run_rollout() {
  local rollout_id=$1 adapter=$2 round=$3
  local merged=${PIPELINE_ROOT}/${rollout_id}/ROLLOUT_MERGED.json
  [[ -f "${merged}" ]] && return 0
  [[ ! -e "${PIPELINE_ROOT}/${rollout_id}" ]] || { echo "partial rollout exists: ${rollout_id}" >&2; exit 4; }
  bash "${HERE}/scripts/run_fresh_coverage_rollout_8gpu.sh" "${rollout_id}" "${adapter}" "${round}"
}
run_pack() {
  local rollout_id=$1 pack_id=$2
  local audit=${REPO_ROOT}/data/processed/uniss_phase3_content_first_joint_s2st_v1/${pack_id}/AUDIT.json
  [[ -f "${audit}" ]] && return 0
  bash "${HERE}/scripts/pack_coverage_round.sh" "${rollout_id}" "${pack_id}"
}
run_train() {
  local run_id=$1 pack_id=$2 load_root=$3
  local root=${REPO_ROOT}/checkpoints/uniss_phase3_content_first_joint_s2st_v1/${run_id}
  [[ -f "${root}/latest_checkpointed_iteration.txt" ]] && return 0
  bash "${HERE}/scripts/run_coverage_grpo_8gpu.sh" "${run_id}" "${pack_id}" "${load_root}"
}
latest_iter() {
  local root=$1 iteration
  iteration=$(tr -d '[:space:]' < "${root}/latest_checkpointed_iteration.txt")
  printf '%s/iter_%07d' "${root}" "$((10#${iteration}))"
}

PRE=content_first_pre_grpo_g4_w64_v1
PACK1=content_first_coverage_round1_pack_v1
R1=content_first_coverage_grpo_round1_v1
MID=content_first_post_round1_g4_w64_v1
PACK2=content_first_coverage_round2_pack_v1
R2=content_first_coverage_grpo_round2_v1
POST=content_first_post_round2_g4_w64_v1

run_rollout "${PRE}" "${SFT_ITER}" 1
run_pack "${PRE}" "${PACK1}"
SMOKE_ROOT=${REPO_ROOT}/checkpoints/uniss_phase3_content_first_joint_s2st_v1/content_first_coverage_grpo_smoke2_v1
if [[ ! -f "${SMOKE_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  EVENT_SMOKE=1 TRAIN_ITERS=2 bash "${HERE}/scripts/run_coverage_grpo_8gpu.sh" \
    content_first_coverage_grpo_smoke2_v1 "${PACK1}" "${SFT_ROOT}"
fi
run_train "${R1}" "${PACK1}" "${SFT_ROOT}"
R1_ROOT=${REPO_ROOT}/checkpoints/uniss_phase3_content_first_joint_s2st_v1/${R1}
R1_ITER=$(latest_iter "${R1_ROOT}")
run_rollout "${MID}" "${R1_ITER}" 2
run_pack "${MID}" "${PACK2}"
run_train "${R2}" "${PACK2}" "${R1_ROOT}"
R2_ROOT=${REPO_ROOT}/checkpoints/uniss_phase3_content_first_joint_s2st_v1/${R2}
R2_ITER=$(latest_iter "${R2_ROOT}")
run_rollout "${POST}" "${R2_ITER}" 3

if [[ ! -f "${REPORT_DIR}/REPORT.zh-CN.md" ]]; then
  "${PYTHON}" -m experiments.uniss_phasea_coverage_constrained_grpo_v3.evaluation.write_report \
    --arm "pre_GRPO=${PIPELINE_ROOT}/${PRE}/ROLLOUT_MERGED.json" \
    --arm "round1=${PIPELINE_ROOT}/${MID}/ROLLOUT_MERGED.json" \
    --arm "round2=${PIPELINE_ROOT}/${POST}/ROLLOUT_MERGED.json" \
    --training-log "round1=${REPO_ROOT}/logs/uniss_phase3_content_first_joint_s2st_v1/${R1}.log" \
    --training-log "round2=${REPO_ROOT}/logs/uniss_phase3_content_first_joint_s2st_v1/${R2}.log" \
    --output-dir "${REPORT_DIR}"
fi
[[ -f "${REPORT_DIR}/REPORT.zh-CN.md" ]] || { echo "final report missing" >&2; exit 5; }
bash "${HERE}/scripts/start_gpu_holder.sh"
pipeline_complete=1
echo "PIPELINE_STATUS=complete"
echo "REPORT=${REPORT_DIR}/REPORT.zh-CN.md"
