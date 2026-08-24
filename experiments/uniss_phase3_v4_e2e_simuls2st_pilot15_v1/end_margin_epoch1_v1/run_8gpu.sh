#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?set a fresh immutable RUN_ID}"

TASK_POOL_RUN_ID=${TASK_POOL_RUN_ID:-task_pool_formal_p4_20260820T154500Z}
TEACHER_RUN_ID=${TEACHER_RUN_ID:-teacher_cache_formal_p4_20260820T154500Z}
STRUCTURAL_CANARY_RUN_ID=${STRUCTURAL_CANARY_RUN_ID:-post_task_pool_canary_p4_replayfix_w0_20260821T085848Z}
MASTER_PORT=${MASTER_PORT:-29924}

TRAIN_REPORT=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_train/BUILD_COMPLETE.json
VALID_REPORT=${PROCESSED_ROOT}/task_pools/${TASK_POOL_RUN_ID}_valid/BUILD_COMPLETE.json
V1_TRAIN_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}_v1_train/AUDIT.json
PHASE3_TRAIN_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_RUN_ID}_phase3_train/AUDIT.json
V1_VALID_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}_v1_valid/AUDIT.json
PHASE3_VALID_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_RUN_ID}_phase3_valid/AUDIT.json
CANARY_REPORT=${REPORT_ROOT}/post_task_pool_canaries/${STRUCTURAL_CANARY_RUN_ID}/CANARY_REPORT.json

RUN_REPORT_ROOT=${REPORT_ROOT}/extended_canaries/${RUN_ID}
RUN_LOG=${LOG_ROOT}/extended_canaries/${RUN_ID}.log
RUN_SAVE_DIR=${CHECKPOINT_ROOT}/extended_canaries/${RUN_ID}
RUN_TENSORBOARD_DIR=${TENSORBOARD_ROOT}/extended_canaries/${RUN_ID}
RUN_GEOMETRY=${RUN_REPORT_ROOT}/TRAINING_GEOMETRY.json
FROZEN_AUDIT=${RUN_REPORT_ROOT}/FROZEN_STAGE_A_BITWISE_AUDIT.json
RUN_SUMMARY=${RUN_REPORT_ROOT}/EXTENDED_CANARY.json
GPU_LOCK=${USER_ROOT}/.locks/uniss_e2e_learning_canary_gpu.lock

required=(
  "${TRAIN_REPORT}"
  "${VALID_REPORT}"
  "${V1_TRAIN_AUDIT}"
  "${PHASE3_TRAIN_AUDIT}"
  "${V1_VALID_AUDIT}"
  "${PHASE3_VALID_AUDIT}"
  "${CANARY_REPORT}"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "missing extended-canary input: ${path}" >&2; exit 3; }
done
for path in "${RUN_REPORT_ROOT}" "${RUN_LOG}" "${RUN_SAVE_DIR}" "${RUN_TENSORBOARD_DIR}"; do
  [[ ! -e "${path}" ]] || {
    echo "refusing to overwrite extended-canary output: ${path}" >&2
    exit 4
  }
done

mkdir -p "$(dirname -- "${GPU_LOCK}")" "${RUN_REPORT_ROOT}"
exec 9>"${GPU_LOCK}"
flock -n 9 || { echo "another E2E research run owns the GPU lock" >&2; exit 5; }
mapfile -t active_gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF && $1 != "[N/A]" {print $1}' | sort -u
)
if (( ${#active_gpu_pids[@]} > 0 )); then
  printf 'GPUs are busy; refusing to interfere with PIDs: %s\n' "${active_gpu_pids[*]}" >&2
  exit 6
fi

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.compute_geometry \
  --task-pool-report "${TRAIN_REPORT}" \
  --global-batch-size 128 \
  --coverage-epochs 1 \
  --seed 20260818 \
  --output "${RUN_GEOMETRY}" >/dev/null

read -r train_iters warmup_iters < <(
  "${PYTHON_BIN}" - "${RUN_GEOMETRY}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["train_iters"], value["warmup_updates"])
PY
)
[[ "${train_iters}" == "1132" && "${warmup_iters}" == "34" ]] || {
  echo "unexpected one-coverage geometry: ${train_iters}/${warmup_iters}" >&2
  exit 7
}

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
env \
  DATA_RUN_ID="${DATA_RUN_ID}" \
  RUN_ID="${RUN_ID}" \
  RUN_TRAIN_BUILD_REPORT="${TRAIN_REPORT}" \
  RUN_VALID_BUILD_REPORT="${VALID_REPORT}" \
  RUN_V1_TRAIN_CACHE_AUDIT="${V1_TRAIN_AUDIT}" \
  RUN_PHASE3_TRAIN_CACHE_AUDIT="${PHASE3_TRAIN_AUDIT}" \
  RUN_V1_VALID_CACHE_AUDIT="${V1_VALID_AUDIT}" \
  RUN_PHASE3_VALID_CACHE_AUDIT="${PHASE3_VALID_AUDIT}" \
  RUN_SAVE_DIR="${RUN_SAVE_DIR}" \
  RUN_TENSORBOARD_DIR="${RUN_TENSORBOARD_DIR}" \
  RUN_LOG="${RUN_LOG}" \
  RUN_GEOMETRY="${RUN_GEOMETRY}" \
  RUN_LOAD="$(dirname -- "${V1_CHECKPOINT}")" \
  RUN_NPROC=8 \
  RUN_MBS=2 \
  RUN_GBS=128 \
  RUN_COVERAGE_EPOCHS=1 \
  RUN_NUM_WORKERS=0 \
  RUN_MASTER_PORT="${MASTER_PORT}" \
  RUN_SAVE_INTERVAL=200 \
  RUN_EVAL_ITERS=0 \
  RUN_EVAL_INTERVAL=200 \
  RUN_LOG_INTERVAL=1 \
  RUN_EXTENDED_CANARY=1 \
  RUN_CANARY_REPORT="${CANARY_REPORT}" \
  RUN_TRAIN_ITERS="${train_iters}" \
  RUN_WARMUP_ITERS="${warmup_iters}" \
  RUN_AUDIT_GRADIENTS=1 \
  RUN_CONTENT_END_WEIGHT=0.0 \
  RUN_SEMANTIC_END_WEIGHT=0.5 \
  RUN_SEMANTIC_END_MARGIN_WEIGHT=0.25 \
  RUN_SEMANTIC_END_LOGIT_MARGIN=2.0 \
  RUN_SEMANTIC_ROLLIN_END_WEIGHT=0.0 \
  RUN_SEMANTIC_ROLLIN_END_MARGIN_WEIGHT=0.0 \
  RUN_SEMANTIC_ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT=0.0 \
  RUN_SEMANTIC_ROLLIN_CONTINUE_MARGIN_WEIGHT=0.0 \
  RUN_SEMANTIC_CONTINUE_MARGIN_WEIGHT=0.0 \
  RUN_SEMANTIC_BOUNDARY_BINARY_WEIGHT=0.0 \
  RUN_SEMANTIC_PREFIX_CORRUPTION_RATE=0.0 \
  RUN_SEMANTIC_BOUNDARY_ROLLIN_RATE=0.0 \
  RUN_VERIFY_DATASET_SHA256=0 \
  RUN_VERIFY_CACHE_SHA256=0 \
  "${BASE_EXPERIMENT}/scripts/run_e2e_megatron.sh"

printf -v iter_tag 'iter_%07d' "$((10#${train_iters}))"
CANDIDATE_CHECKPOINT=${RUN_SAVE_DIR}/${iter_tag}
"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.audit_frozen_stage_a \
  --reference "${V1_CHECKPOINT}" \
  --candidate "extended_canary=${CANDIDATE_CHECKPOINT}" \
  --output "${FROZEN_AUDIT}" >/dev/null
ended_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

jq -n \
  --arg started_at "${started_at}" \
  --arg ended_at "${ended_at}" \
  --arg run_id "${RUN_ID}" \
  --arg checkpoint "${CANDIDATE_CHECKPOINT}" \
  --arg tensorboard "${RUN_TENSORBOARD_DIR}" \
  --arg log "${RUN_LOG}" \
  --arg gpu_csv "${RUN_LOG%.log}.gpu.csv" \
  --arg frozen_audit "${FROZEN_AUDIT}" \
  --arg structural_canary "${CANARY_REPORT}" \
  --argjson train_iters "${train_iters}" \
  '{schema_version:"uniss_e2e_extended_canary_v1",status:"complete",formal_training_authorized:false,started_at:$started_at,ended_at:$ended_at,run_id:$run_id,coverage_epochs:1,train_iters:$train_iters,checkpoint:$checkpoint,tensorboard:$tensorboard,log:$log,gpu_csv:$gpu_csv,frozen_stage_a_audit:$frozen_audit,structural_canary:$structural_canary,next_required_gate:"fixed_16_sample_free_running_validation"}' \
  > "${RUN_SUMMARY}"

echo "extended_canary_status=complete"
echo "checkpoint=${CANDIDATE_CHECKPOINT}"
echo "report=${RUN_SUMMARY}"
echo "tensorboard=${RUN_TENSORBOARD_DIR}"
echo "next_gate=fixed_16_sample_free_running_validation"
