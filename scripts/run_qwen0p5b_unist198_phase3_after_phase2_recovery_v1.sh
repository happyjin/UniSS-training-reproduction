#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen0p5b_unist198_phase3_after_phase2_recovery_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

PHASE3_LR="${PHASE3_LR:-}"
PHASE3_MIN_LR="${PHASE3_MIN_LR:-}"

PHASE2_TRACKER="${PHASE2_SAVE_DIR}/latest_checkpointed_iteration.txt"
PHASE2_COMPLETE_MARKER="${PHASE2_RUN_DIR}/TRAINING_COMPLETE"
PHASE2_GATE_OUTPUT="${PHASE3_RUN_DIR}/PHASE2_FINAL_GATE.json"
PHASE2_GATE_MARKER="${PHASE3_RUN_DIR}/PHASE2_FINAL_GATE_PASSED"
PHASE3_TRACKER="${PHASE3_SAVE_DIR}/latest_checkpointed_iteration.txt"
PHASE3_COMPLETE_MARKER="${PHASE3_RUN_DIR}/TRAINING_COMPLETE"
WAIT_LOG="${PHASE3_RUN_DIR}/wait_and_train.log"

[[ "${NPROC_PER_NODE}" == "8" ]] || { echo "Phase3 requires 8 processes" >&2; exit 1; }
[[ "${GLOBAL_BATCH_SIZE}" == "128" ]] || { echo "Phase3 requires global batch size 128" >&2; exit 1; }
[[ "${DATALOADER_TYPE}" == "cyclic" ]] || { echo "Phase3 requires cyclic shuffled loading" >&2; exit 1; }
[[ "$((PHASE2_SOURCE_ITERATION + PHASE2_TRAIN_ITERS))" == "${PHASE2_EFFECTIVE_FINAL_ITERATION}" ]] || {
  echo "Phase2 source ${PHASE2_SOURCE_ITERATION} + local target ${PHASE2_TRAIN_ITERS} does not equal effective target ${PHASE2_EFFECTIVE_FINAL_ITERATION}" >&2
  exit 1
}
[[ "${PHASE3_TRAIN_ITERS}" == "9075" ]] || { echo "Unexpected Phase3 target: ${PHASE3_TRAIN_ITERS}" >&2; exit 1; }
[[ "${NO_DATA_SHARDING}" == "0" || "${NO_DATA_SHARDING}" == "1" ]] || {
  echo "NO_DATA_SHARDING must be 0 or 1" >&2
  exit 1
}
[[ "${FULL_VALIDATION}" == "0" || "${FULL_VALIDATION}" == "1" ]] || {
  echo "FULL_VALIDATION must be 0 or 1" >&2
  exit 1
}
if [[ "${FULL_VALIDATION}" == "1" ]]; then
  [[ "${EVAL_MICRO_BATCH_SIZE}" == "1" ]] || {
    echo "Full validation requires EVAL_MICRO_BATCH_SIZE=1" >&2
    exit 1
  }
  [[ "${EVAL_GLOBAL_BATCH_SIZE}" == "${NPROC_PER_NODE}" ]] || {
    echo "Full validation requires one sample per data-parallel rank: EVAL_GLOBAL_BATCH_SIZE=${NPROC_PER_NODE}" >&2
    exit 1
  }
fi
if [[ -n "${PHASE3_LR}" || -n "${PHASE3_MIN_LR}" ]]; then
  [[ -n "${PHASE3_LR}" && -n "${PHASE3_MIN_LR}" ]] || {
    echo "PHASE3_LR and PHASE3_MIN_LR must be set together" >&2
    exit 1
  }
fi

tracker_iteration() {
  local tracker="$1"
  if [[ ! -s "${tracker}" ]]; then
    echo -1
    return
  fi
  local value
  value="$(tr -d '[:space:]' < "${tracker}")"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Invalid checkpoint tracker ${tracker}: ${value}" >&2; exit 1; }
  echo "${value}"
}

phase2_pipeline_is_alive() {
  pgrep -af "${PHASE2_PIPELINE_PATTERN}" >/dev/null 2>&1
}

phase3_training_is_alive() {
  pgrep -af 'training/pretrain_uniss_megatron.py' 2>/dev/null \
    | rg -F -- "--save ${PHASE3_SAVE_DIR}" >/dev/null 2>&1
}

configure_python_nvidia_libraries() {
  local library_dirs=()
  local cuda_root directory site_packages joined
  if command -v nvcc >/dev/null 2>&1; then
    cuda_root="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)"
    for directory in \
      "${cuda_root}/lib" \
      "${cuda_root}/lib64" \
      "${cuda_root}/targets/x86_64-linux/lib"; do
      [[ -d "${directory}" ]] && library_dirs+=("${directory}")
    done
  fi
  site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
  shopt -s nullglob
  for directory in "${site_packages}"/nvidia/*/lib; do
    [[ -d "${directory}" ]] && library_dirs+=("${directory}")
  done
  shopt -u nullglob
  (( ${#library_dirs[@]} > 0 )) || { echo "No CUDA or pip NVIDIA library directories found" >&2; return 1; }
  joined="$(IFS=:; echo "${library_dirs[*]}")"
  export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

phase3_cmd=(env
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "NPROC_PER_NODE=${NPROC_PER_NODE}"
  "TP=${TP}" "PP=${PP}" "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
  "TRAIN_DATA=${PHASE3_TRAIN}" "VALID_DATA=${PHASE3_VALID}"
  "SAVE_DIR=${PHASE3_SAVE_DIR}" "TRAIN_ITERS=${PHASE3_TRAIN_ITERS}"
  "LR=${PHASE3_LR}" "MIN_LR=${PHASE3_MIN_LR}"
  "LR_WARMUP_ITERS=${PHASE3_LR_WARMUP_ITERS}"
  "DATALOADER_TYPE=${DATALOADER_TYPE}"
  "SAVE_INTERVAL=${SAVE_INTERVAL}" "EVAL_INTERVAL=${EVAL_INTERVAL}"
  "EVAL_ITERS=${EVAL_ITERS}" "LOG_INTERVAL=${LOG_INTERVAL}"
  "MASTER_PORT=${PHASE3_MASTER_PORT}"
  "LOAD_CHECKPOINT=${PHASE2_SAVE_DIR}"
  "FINETUNE=1" "LOAD_OPTIM=0" "LOAD_RNG=0"
  "${REPO_ROOT}/scripts/train_phase3_qwen0p5b.sh")

phase3_args=(
  --lr-decay-iters "${PHASE3_TRAIN_ITERS}"
  --clip-grad "${CLIP_GRAD}"
  --seed "${SEED}"
  --attention-backend fused
  --tensorboard-dir "${PHASE3_TENSORBOARD_DIR}"
  --tensorboard-log-interval "${TENSORBOARD_LOG_INTERVAL}"
  --log-timers-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard
  --log-memory-interval "${TENSORBOARD_MEMORY_INTERVAL}"
  --log-world-size-to-tensorboard
  --log-throughput
)
[[ "${NO_DATA_SHARDING}" == "1" ]] && phase3_args+=(--no-data-sharding)
[[ "${FULL_VALIDATION}" == "1" ]] && phase3_args+=(--full-validation)
if [[ -n "${EVAL_MICRO_BATCH_SIZE}" ]]; then
  phase3_args+=(--eval-micro-batch-size "${EVAL_MICRO_BATCH_SIZE}")
fi
if [[ -n "${EVAL_GLOBAL_BATCH_SIZE}" ]]; then
  phase3_args+=(--eval-global-batch-size "${EVAL_GLOBAL_BATCH_SIZE}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] wait for Phase2 local checkpoint ${PHASE2_TRAIN_ITERS} (source=${PHASE2_SOURCE_ITERATION}, effective=${PHASE2_EFFECTIVE_FINAL_ITERATION}) and marker ${PHASE2_COMPLETE_MARKER}"
  echo "[dry-run] validate final Phase2 TensorBoard/log before allocating GPUs"
  echo "[dry-run] Phase3 packed count=1161587 target=${PHASE3_TRAIN_ITERS} dataloader=${DATALOADER_TYPE} seed=${SEED}"
  echo "[dry-run] Phase3 TensorBoard=${PHASE3_TENSORBOARD_DIR} port=${PHASE3_TENSORBOARD_PORT}"
  printf '[dry-run] '
  printf '%q ' "${phase3_cmd[@]}" --dry-run "${phase3_args[@]}"
  printf '\n'
  "${phase3_cmd[@]}" --dry-run "${phase3_args[@]}"
  exit 0
fi

if [[ "${FULL_VALIDATION}" == "1" ]]; then
  "${REPO_ROOT}/scripts/apply_megatron_full_validation_patch.sh"
fi

for required in "${ACTIVATE_SCRIPT}" "${PHASE3_TRAIN}" "${PHASE3_VALID}" "${PHASE3_TRAIN}.count"; do
  [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done
[[ "$(tr -d '[:space:]' < "${PHASE3_TRAIN}.count")" == "1161587" ]] || {
  echo "Unexpected Phase3 packed count" >&2
  exit 1
}

mkdir -p "${PHASE3_RUN_DIR}" "$(dirname "${PHASE3_LOG_PATH}")"
log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${WAIT_LOG}"
}

last_reported=-2
while true; do
  phase2_iteration="$(tracker_iteration "${PHASE2_TRACKER}")"
  if [[ "${phase2_iteration}" != "${last_reported}" ]]; then
    log "waiting for Phase2: checkpoint=${phase2_iteration}/${PHASE2_TRAIN_ITERS}"
    last_reported="${phase2_iteration}"
  fi
  if (( phase2_iteration == PHASE2_TRAIN_ITERS )) && [[ -f "${PHASE2_COMPLETE_MARKER}" ]]; then
    break
  fi
  if (( phase2_iteration > PHASE2_TRAIN_ITERS )); then
    log "ERROR Phase2 checkpoint exceeds expected target"
    exit 1
  fi
  if ! phase2_pipeline_is_alive; then
    log "ERROR Phase2 pipeline exited without its final checkpoint and completion marker"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

log "Phase2 final checkpoint and completion marker confirmed; running final health gate"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
phase2_gate_args=(
  --tensorboard-dir "${PHASE2_TENSORBOARD_DIR}" \
  --log "${PHASE2_LOG_PATH}" \
  --required-step "${PHASE2_TRAIN_ITERS}" \
  --max-valid-loss "${PHASE2_MAX_VALID_LOSS}" \
  --max-last-valid-loss "${PHASE2_MAX_LAST_VALID_LOSS}" \
  --grad-spike-threshold "${PHASE2_GRAD_SPIKE_THRESHOLD}" \
  --max-consecutive-grad-spikes "${PHASE2_MAX_CONSECUTIVE_GRAD_SPIKES}" \
  --output "${PHASE2_GATE_OUTPUT}"
)
[[ -n "${PHASE2_ABSOLUTE_MAX_GRAD_NORM}" ]] && \
  phase2_gate_args+=(--absolute-max-grad-norm "${PHASE2_ABSOLUTE_MAX_GRAD_NORM}")
"${ENV_ROOT}/bin/python" -m training.validate_phase2_recovery "${phase2_gate_args[@]}"
printf 'passed_at=%s\n' "$(date -u +%FT%TZ)" > "${PHASE2_GATE_MARKER}"

if phase3_training_is_alive; then
  log "ERROR Phase3 is already running for ${PHASE3_SAVE_DIR}"
  exit 1
fi

current_phase3_iteration="$(tracker_iteration "${PHASE3_TRACKER}")"
if (( current_phase3_iteration == PHASE3_TRAIN_ITERS )); then
  log "Phase3 is already complete at iteration ${PHASE3_TRAIN_ITERS}"
  exit 0
elif (( current_phase3_iteration >= 0 && current_phase3_iteration < PHASE3_TRAIN_ITERS )); then
  phase3_cmd=(env
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "NPROC_PER_NODE=${NPROC_PER_NODE}"
    "TP=${TP}" "PP=${PP}" "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
    "TRAIN_DATA=${PHASE3_TRAIN}" "VALID_DATA=${PHASE3_VALID}"
    "SAVE_DIR=${PHASE3_SAVE_DIR}" "TRAIN_ITERS=${PHASE3_TRAIN_ITERS}"
    "LR=${PHASE3_LR}" "MIN_LR=${PHASE3_MIN_LR}"
    "LR_WARMUP_ITERS=${PHASE3_LR_WARMUP_ITERS}"
    "DATALOADER_TYPE=${DATALOADER_TYPE}"
    "SAVE_INTERVAL=${SAVE_INTERVAL}" "EVAL_INTERVAL=${EVAL_INTERVAL}"
    "EVAL_ITERS=${EVAL_ITERS}" "LOG_INTERVAL=${LOG_INTERVAL}"
    "MASTER_PORT=${PHASE3_MASTER_PORT}"
    "LOAD_CHECKPOINT=${PHASE3_SAVE_DIR}"
    "FINETUNE=0" "LOAD_OPTIM=1" "LOAD_RNG=1"
    "${REPO_ROOT}/scripts/train_phase3_qwen0p5b.sh")
  run_kind=resume
else
  if [[ -e "${PHASE3_SAVE_DIR}" ]]; then
    log "ERROR refusing fresh Phase3 run into existing directory without a tracker: ${PHASE3_SAVE_DIR}"
    exit 1
  fi
  run_kind=fresh
fi

configure_python_nvidia_libraries
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES
gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
[[ "${gpu_count}" == "8" ]] || { log "ERROR expected 8 visible GPUs, found ${gpu_count}"; exit 1; }

mkdir -p "${PHASE3_SAVE_DIR}" "${PHASE3_TENSORBOARD_DIR}"
if [[ "${run_kind}" == "fresh" ]]; then
  {
    echo "experiment=${PHASE3_NAME}"
    echo "created_at=$(date -u +%FT%TZ)"
    echo "repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "source_phase2=${PHASE2_SAVE_DIR}"
    echo "source_phase2_iteration=${PHASE2_TRAIN_ITERS}"
    echo "source_phase2_base_iteration=${PHASE2_SOURCE_ITERATION}"
    echo "source_phase2_effective_iteration=${PHASE2_EFFECTIVE_FINAL_ITERATION}"
    echo "train_data=${PHASE3_TRAIN}"
    echo "valid_data=${PHASE3_VALID}"
    echo "train_iters=${PHASE3_TRAIN_ITERS}"
    echo "lr=${PHASE3_LR:-default}"
    echo "min_lr=${PHASE3_MIN_LR:-default}"
    echo "lr_warmup_iters=${PHASE3_LR_WARMUP_ITERS}"
    echo "dataloader_type=${DATALOADER_TYPE}"
    echo "no_data_sharding=${NO_DATA_SHARDING}"
    echo "full_validation=${FULL_VALIDATION}"
    echo "eval_micro_batch_size=${EVAL_MICRO_BATCH_SIZE}"
    echo "eval_global_batch_size=${EVAL_GLOBAL_BATCH_SIZE}"
    echo "seed=${SEED}"
    echo "nproc_per_node=${NPROC_PER_NODE}"
    echo "micro_batch_size=${MICRO_BATCH_SIZE}"
    echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
    echo "tensorboard_dir=${PHASE3_TENSORBOARD_DIR}"
  } > "${PHASE3_RUN_DIR}/manifest.txt"
fi

log "starting isolated 8-GPU Phase3 (${run_kind}); target=${PHASE3_TRAIN_ITERS}; shuffle=${DATALOADER_TYPE}"
set +e
"${phase3_cmd[@]}" "${phase3_args[@]}" 2>&1 | tee -a "${PHASE3_LOG_PATH}"
train_status=${PIPESTATUS[0]}
set -e
if (( train_status != 0 )); then
  log "ERROR Phase3 failed with status ${train_status}"
  exit "${train_status}"
fi

actual_iteration="$(tracker_iteration "${PHASE3_TRACKER}")"
[[ "${actual_iteration}" == "${PHASE3_TRAIN_ITERS}" ]] || {
  log "ERROR Phase3 ended at ${actual_iteration}, expected ${PHASE3_TRAIN_ITERS}"
  exit 1
}
printf 'completed_at=%s\niteration=%s\n' \
  "$(date -u +%FT%TZ)" "${actual_iteration}" > "${PHASE3_COMPLETE_MARKER}"
log "completed Phase3 at iteration ${actual_iteration}"
