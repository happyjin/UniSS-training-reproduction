#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${TASK_POOL_RUN_ID:?set the immutable formal task-pool run ID}"
: "${TEACHER_FORMAL_RUN_ID:?set the immutable teacher-cache run ID}"
: "${CANARY_RUN_ID:?set the immutable post-task-pool canary run ID}"

TASK_POOL_SESSION=${TASK_POOL_SESSION:-uniss_e2e_task_pools_after_teacher}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-30}
CANARY_MBS=${CANARY_MBS:-2}
CANARY_GBS=${CANARY_GBS:-128}
CANARY_NUM_WORKERS=${CANARY_NUM_WORKERS:-8}
CANARY_MASTER_PORT_BASE=${CANARY_MASTER_PORT_BASE:-29810}
if (( WAIT_INTERVAL_SECONDS < 1 )); then
  echo "WAIT_INTERVAL_SECONDS must be positive" >&2
  exit 2
fi
if [[ "${CANARY_MBS}" != "1" && "${CANARY_MBS}" != "2" ]]; then
  echo "CANARY_MBS must be 1 or 2" >&2
  exit 2
fi

TRAIN_REPORT=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_train/BUILD_COMPLETE.json
VALID_REPORT=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_valid/BUILD_COMPLETE.json
V1_TRAIN_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_FORMAL_RUN_ID}_v1_train/AUDIT.json
PHASE3_TRAIN_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_FORMAL_RUN_ID}_phase3_train/AUDIT.json
V1_VALID_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_FORMAL_RUN_ID}_v1_valid/AUDIT.json
PHASE3_VALID_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_FORMAL_RUN_ID}_phase3_valid/AUDIT.json
GOLD_GATE=${REPORT_ROOT}/GOLD_TRAJECTORY_GATE.json
CANARY_REPORT_ROOT=${REPORT_ROOT}/post_task_pool_canaries/${CANARY_RUN_ID}
PREFLIGHT=${CANARY_REPORT_ROOT}/PREFLIGHT.json
RESULTS=${CANARY_REPORT_ROOT}/RUN_RESULTS.jsonl
FINAL_REPORT=${CANARY_REPORT_ROOT}/CANARY_REPORT.json
RUN_LOG_ROOT=${LOG_ROOT}/post_task_pool_canaries/${CANARY_RUN_ID}
RUN_SAVE_ROOT=${CHECKPOINT_ROOT}/post_task_pool_canaries/${CANARY_RUN_ID}
RUN_TENSORBOARD_ROOT=${TENSORBOARD_ROOT}/post_task_pool_canaries/${CANARY_RUN_ID}
GPU_LOCK=${USER_ROOT}/.locks/uniss_e2e_post_task_pool_canary_gpu.lock

prerequisites=(
  "${TRAIN_REPORT}"
  "${VALID_REPORT}"
  "${V1_TRAIN_AUDIT}"
  "${PHASE3_TRAIN_AUDIT}"
  "${V1_VALID_AUDIT}"
  "${PHASE3_VALID_AUDIT}"
  "${GOLD_GATE}"
)
while true; do
  missing=0
  for path in "${prerequisites[@]}"; do
    [[ -f "${path}" ]] || missing=$((missing + 1))
  done
  (( missing > 0 )) || break
  if ! tmux has-session -t "${TASK_POOL_SESSION}" 2>/dev/null; then
    echo "task-pool session ended with ${missing} canary prerequisite(s) missing" >&2
    printf 'missing_or_pending=%s\n' "${prerequisites[@]}" >&2
    exit 3
  fi
  printf '%s waiting for formal task pools and teacher audits (%d missing)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${missing}"
  sleep "${WAIT_INTERVAL_SECONDS}"
done

for path in "${CANARY_REPORT_ROOT}" "${RUN_LOG_ROOT}" "${RUN_SAVE_ROOT}" \
  "${RUN_TENSORBOARD_ROOT}"; do
  [[ ! -e "${path}" ]] || {
    echo "refusing to overwrite post-task-pool canary output: ${path}" >&2
    exit 4
  }
done
mkdir -p "${CANARY_REPORT_ROOT}" "$(dirname -- "${GPU_LOCK}")"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.canary_gate \
  preflight \
  --data-run-id "${DATA_RUN_ID}" \
  --task-pool-run-id "${TASK_POOL_RUN_ID}" \
  --teacher-run-id "${TEACHER_FORMAL_RUN_ID}" \
  --train-report "${TRAIN_REPORT}" \
  --valid-report "${VALID_REPORT}" \
  --v1-train-audit "${V1_TRAIN_AUDIT}" \
  --phase3-train-audit "${PHASE3_TRAIN_AUDIT}" \
  --v1-valid-audit "${V1_VALID_AUDIT}" \
  --phase3-valid-audit "${PHASE3_VALID_AUDIT}" \
  --gold-gate "${GOLD_GATE}" \
  --output "${PREFLIGHT}" >/dev/null

exec 9>"${GPU_LOCK}"
flock 9
while true; do
  mapfile -t gpu_pids < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' | sort -u
  )
  if (( ${#gpu_pids[@]} == 0 )); then
    break
  fi
  printf '%s waiting for GPUs to become free; refusing to interrupt PIDs: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${gpu_pids[*]}"
  ps -o pid=,user=,etimes=,args= -p "$(IFS=,; echo "${gpu_pids[*]}")" 2>/dev/null || true
  sleep "${WAIT_INTERVAL_SECONDS}"
done

run_one() {
  local name=$1
  local family=$2
  local train_iters=$3
  local port=$4
  local run_id=${CANARY_RUN_ID}_${name}
  local log=${RUN_LOG_ROOT}/${name}.log
  local save_dir=${RUN_SAVE_ROOT}/${name}
  local tensorboard_dir=${RUN_TENSORBOARD_ROOT}/${name}
  local geometry=${CANARY_REPORT_ROOT}/geometry/${name}.json
  local started_at ended_at exit_code
  local -a family_env=()
  if [[ -n "${family}" ]]; then
    family_env=(RUN_SMOKE_FAMILY="${family}")
  fi
  started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  set +e
  env \
    DATA_RUN_ID="${DATA_RUN_ID}" \
    RUN_ID="${run_id}" \
    RUN_TRAIN_BUILD_REPORT="${TRAIN_REPORT}" \
    RUN_VALID_BUILD_REPORT="${VALID_REPORT}" \
    RUN_V1_TRAIN_CACHE_AUDIT="${V1_TRAIN_AUDIT}" \
    RUN_PHASE3_TRAIN_CACHE_AUDIT="${PHASE3_TRAIN_AUDIT}" \
    RUN_V1_VALID_CACHE_AUDIT="${V1_VALID_AUDIT}" \
    RUN_PHASE3_VALID_CACHE_AUDIT="${PHASE3_VALID_AUDIT}" \
    RUN_SAVE_DIR="${save_dir}" \
    RUN_TENSORBOARD_DIR="${tensorboard_dir}" \
    RUN_LOG="${log}" \
    RUN_GEOMETRY="${geometry}" \
    RUN_LOAD="$(dirname -- "${V1_CHECKPOINT}")" \
    RUN_NPROC=8 \
    RUN_MBS="${CANARY_MBS}" \
    RUN_GBS="${CANARY_GBS}" \
    RUN_NUM_WORKERS="${CANARY_NUM_WORKERS}" \
    RUN_MASTER_PORT="${port}" \
    RUN_SAVE_INTERVAL=1 \
    RUN_EVAL_ITERS=0 \
    RUN_LOG_INTERVAL=1 \
    RUN_SMOKE=1 \
    RUN_TRAIN_ITERS="${train_iters}" \
    RUN_WARMUP_ITERS=0 \
    RUN_AUDIT_GRADIENTS=1 \
    RUN_VERIFY_DATASET_SHA256=0 \
    RUN_VERIFY_CACHE_SHA256=0 \
    "${family_env[@]}" \
    "${SCRIPT_DIR}/run_e2e_megatron.sh"
  exit_code=$?
  set -e
  ended_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  jq -cn \
    --arg name "${name}" \
    --arg family "${family}" \
    --arg started_at "${started_at}" \
    --arg ended_at "${ended_at}" \
    --arg log "${log}" \
    --arg gpu_csv "${log%.log}.gpu.csv" \
    --arg save_dir "${save_dir}" \
    --arg tensorboard_dir "${tensorboard_dir}" \
    --argjson train_iters "${train_iters}" \
    --argjson exit_code "${exit_code}" \
    '{name:$name,family:(if $family == "" then null else $family end),train_iters:$train_iters,exit_code:$exit_code,started_at:$started_at,ended_at:$ended_at,log:$log,gpu_csv:$gpu_csv,save_dir:$save_dir,tensorboard_dir:$tensorboard_dir}' \
    >> "${RESULTS}"
  if (( exit_code != 0 )); then
    echo "post-task-pool canary failed: ${name} (exit ${exit_code})" >&2
    return "${exit_code}"
  fi
}

run_one structural "" 2 "$((CANARY_MASTER_PORT_BASE + 0))"
families=(
  streaming_asr_event
  incremental_mt_event
  interleaved_e2e_s2st
  phase3_quality_replay
  phase3_performance_replay
)
for index in "${!families[@]}"; do
  family=${families[index]}
  run_one "${family}" "${family}" 1 "$((CANARY_MASTER_PORT_BASE + index + 1))"
done

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.canary_gate \
  finalize \
  --preflight "${PREFLIGHT}" \
  --results "${RESULTS}" \
  --output "${FINAL_REPORT}" >/dev/null

echo "post_task_pool_canary_status=passed"
echo "report=${FINAL_REPORT}"
echo "formal_training_authorized=false"
echo "next_gate=free_running_validation_and_frozen_parameter_bitwise_audit"
