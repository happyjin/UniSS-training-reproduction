#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"

TRAINING_RUN=${1:-episode_grpo_formal_8gpu_v1}
CHECKPOINT_ROOT=${REPO_ROOT}/checkpoints/uniss_phasea_stateful_longepisode_rl_v1/${TRAINING_RUN}
TRAINING_LOG=${REPO_ROOT}/logs/uniss_phasea_stateful_longepisode_rl_v1/${TRAINING_RUN}.log
COMPARISON_ROOT=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1
REPORT_ROOT=${REPO_ROOT}/reports/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1
SELECTION=${COMPARISON_ROOT}/CHECKPOINT_SELECTION.json
C0=${REPO_ROOT}/eval_outputs/uniss_stagea_quality_first_joint_grpo_v1/formal_complete_v1/bounded_longform_chunk640_recovery2/stage_a_iter381_merged/results.json
C1=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/phasea_iter381_runtime_v2/results.json
C2_CHECKPOINT=${REPO_ROOT}/checkpoints/uniss_stagea_quality_first_joint_grpo_v1/a3_g8_full_recovery1/iter_0002510
ATTRIBUTION=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/reference_attribution_valid16_v1/ATTRIBUTION_MERGED.json
TRAIN_ROLLOUT=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/ROLLOUT_MERGED.json
VALID_ROLLOUT=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_valid16_g4_v1/ROLLOUT_MERGED.json
FINAL_REPORT=${REPORT_ROOT}/REPORT.zh-CN.md

mkdir -p "${COMPARISON_ROOT}" "${REPORT_ROOT}/stages"
exec 9>"${COMPARISON_ROOT}/post_train_evaluation.lock"
flock 9
if [[ -f "${FINAL_REPORT}" ]]; then
  echo "REPORT=${FINAL_REPORT}"
  exit 0
fi

for path in "${TRAINING_LOG}" "${C0}" "${C1}" "${C2_CHECKPOINT}/.metadata" "${ATTRIBUTION}" "${TRAIN_ROLLOUT}" "${VALID_ROLLOUT}"; do
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
rg -q '\[after training is done\]' "${TRAINING_LOG}" || {
  echo "formal training has not finished: ${TRAINING_LOG}" >&2
  exit 2
}

if [[ ! -f "${SELECTION}" ]]; then
  "${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/select_rl_checkpoint.py" \
    --log "${TRAINING_LOG}" --checkpoint-root "${CHECKPOINT_ROOT}" \
    --output "${SELECTION}"
fi

mapfile -t EPOCH_CHECKPOINTS < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' -printf '%p\n' | sort)
[[ ${#EPOCH_CHECKPOINTS[@]} -eq 3 ]] || {
  echo "expected exactly three epoch checkpoints, found ${#EPOCH_CHECKPOINTS[@]}" >&2
  exit 3
}

run_arm() {
  local run_id=$1 checkpoint=$2 output=$3 gpu0=$4 gpu1=$5 gpu2=$6 gpu3=$7
  if [[ ! -f "${output}/results.json" ]]; then
    bash "${SCRIPT_DIR}/run_stateful_longform_4gpu.sh" \
      "${run_id}" "${output}" "${checkpoint}" "${gpu0}" "${gpu1}" "${gpu2}" "${gpu3}"
  fi
  local stage_report=${REPORT_ROOT}/stages/${run_id}.zh-CN.md
  if [[ ! -f "${stage_report}" ]]; then
    "${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/write_stage_report.py" \
      --stage-name "${run_id}" --old-results "${C0}" \
      --new-results "${output}/results.json" --output "${stage_report}"
  fi
}

C2_OUTPUT=${COMPARISON_ROOT}/c2_old_a3_runtime_v2
EPOCH0_OUTPUT=${COMPARISON_ROOT}/rl_epoch1_runtime_v2
run_arm c2_old_a3_runtime_v2 "${C2_CHECKPOINT}" "${C2_OUTPUT}" 0 1 2 3 &
pid0=$!
run_arm rl_epoch1_runtime_v2 "${EPOCH_CHECKPOINTS[0]}" "${EPOCH0_OUTPUT}" 4 5 6 7 &
pid1=$!
status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
[[ ${status} -eq 0 ]] || { echo "first comparison wave failed" >&2; exit 4; }

EPOCH1_OUTPUT=${COMPARISON_ROOT}/rl_epoch2_runtime_v2
EPOCH2_OUTPUT=${COMPARISON_ROOT}/rl_epoch3_runtime_v2
run_arm rl_epoch2_runtime_v2 "${EPOCH_CHECKPOINTS[1]}" "${EPOCH1_OUTPUT}" 0 1 2 3 &
pid0=$!
run_arm rl_epoch3_runtime_v2 "${EPOCH_CHECKPOINTS[2]}" "${EPOCH2_OUTPUT}" 4 5 6 7 &
pid1=$!
status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
[[ ${status} -eq 0 ]] || { echo "second comparison wave failed" >&2; exit 5; }

if [[ ! -f "${FINAL_REPORT}" ]]; then
  "${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/write_final_report.py" \
    --runtime-v1 "${C0}" --runtime-v2 "${C1}" \
    --a3-v2 "${C2_OUTPUT}/results.json" --selection "${SELECTION}" \
    --epoch-result "${EPOCH0_OUTPUT}/results.json" \
    --epoch-result "${EPOCH1_OUTPUT}/results.json" \
    --epoch-result "${EPOCH2_OUTPUT}/results.json" \
    --attribution "${ATTRIBUTION}" --train-rollout "${TRAIN_ROLLOUT}" \
    --valid-rollout "${VALID_ROLLOUT}" --output "${FINAL_REPORT}"
fi
echo "SELECTION=${SELECTION}"
echo "REPORT=${FINAL_REPORT}"
bash "${SCRIPT_DIR}/start_gpu_holder_after_completion.sh"
