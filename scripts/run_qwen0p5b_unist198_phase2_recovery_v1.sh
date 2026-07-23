#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
MODE="pilot"
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${MODE}" in
  pilot|full) ;;
  *) echo "MODE must be pilot or full" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen0p5b_unist198_phase2_recovery_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

if [[ "${MODE}" == "pilot" ]]; then
  EXPERIMENT_NAME="${RECOVERY_NAME}_pilot"
  SAVE_DIR="${PILOT_SAVE_DIR}"
  RUN_DIR="${PILOT_RUN_DIR}"
  TENSORBOARD_DIR="${PILOT_TENSORBOARD_DIR}"
  LOG_PATH="${PILOT_LOG_PATH}"
  TRAIN_ITERS="${PILOT_TRAIN_ITERS}"
  LR_WARMUP_ITERS="${PILOT_LR_WARMUP_ITERS}"
  EVAL_INTERVAL="${PILOT_EVAL_INTERVAL}"
  MASTER_PORT="${PILOT_MASTER_PORT}"
else
  EXPERIMENT_NAME="${RECOVERY_NAME}_full"
  SAVE_DIR="${FULL_SAVE_DIR}"
  RUN_DIR="${FULL_RUN_DIR}"
  TENSORBOARD_DIR="${FULL_TENSORBOARD_DIR}"
  LOG_PATH="${FULL_LOG_PATH}"
  TRAIN_ITERS="${FULL_TRAIN_ITERS}"
  LR_WARMUP_ITERS="${FULL_LR_WARMUP_ITERS}"
  EVAL_INTERVAL="${FULL_EVAL_INTERVAL}"
  MASTER_PORT="${FULL_MASTER_PORT}"
fi
LR_DECAY_ITERS="${TRAIN_ITERS}"

[[ "${NPROC_PER_NODE}" == "8" ]] || { echo "Recovery requires 8 GPUs" >&2; exit 1; }
[[ "${GLOBAL_BATCH_SIZE}" == "128" ]] || { echo "Recovery requires global batch size 128" >&2; exit 1; }
[[ "${SOURCE_ITERATION}" == "${EXPECTED_SOURCE_ITERATION}" ]] || {
  echo "Recovery source ${SOURCE_ITERATION} does not match expected ${EXPECTED_SOURCE_ITERATION}" >&2
  exit 1
}
[[ "${DATALOADER_TYPE}" == "cyclic" ]] || { echo "Recovery requires cyclic dataloader" >&2; exit 1; }
[[ "${NO_DATA_SHARDING}" == "0" || "${NO_DATA_SHARDING}" == "1" ]] || {
  echo "NO_DATA_SHARDING must be 0 or 1" >&2
  exit 1
}
[[ "${FULL_VALIDATION}" == "0" || "${FULL_VALIDATION}" == "1" ]] || {
  echo "FULL_VALIDATION must be 0 or 1" >&2
  exit 1
}

SOURCE_ITER_DIR="${SOURCE_CHECKPOINT_ROOT}/iter_$(printf '%07d' "${SOURCE_ITERATION}")"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

tracker_iteration() {
  local root="$1" tracker value
  tracker="${root}/latest_checkpointed_iteration.txt"
  if [[ ! -s "${tracker}" ]]; then
    echo -1
    return
  fi
  value="$(tr -d '[:space:]' < "${tracker}")"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Invalid tracker: ${tracker}" >&2; exit 1; }
  echo "${value}"
}

prepare_candidate() {
  local candidate_tmp shard_count
  require_file "${SOURCE_ITER_DIR}/metadata.json"
  require_file "${SOURCE_ITER_DIR}/.metadata"
  shard_count="$(find "${SOURCE_ITER_DIR}" -maxdepth 1 -type f -name '__*_0.distcp' | wc -l)"
  [[ "${shard_count}" == "8" ]] || { echo "Expected 8 checkpoint shards, found ${shard_count}" >&2; exit 1; }
  if [[ -e "${CANDIDATE_LOAD_ROOT}" ]]; then
    [[ "$(tracker_iteration "${CANDIDATE_LOAD_ROOT}")" == "${SOURCE_ITERATION}" ]] || {
      echo "Candidate does not point to iteration ${SOURCE_ITERATION}" >&2
      exit 1
    }
    require_file "${CANDIDATE_LOAD_ROOT}/$(basename "${SOURCE_ITER_DIR}")/metadata.json"
    return
  fi
  mkdir -p "$(dirname "${CANDIDATE_LOAD_ROOT}")"
  candidate_tmp="${CANDIDATE_LOAD_ROOT}.tmp.$$"
  rm -rf "${candidate_tmp}"
  mkdir -p "${candidate_tmp}"
  cp -al "${SOURCE_ITER_DIR}" "${candidate_tmp}/$(basename "${SOURCE_ITER_DIR}")"
  printf '%s\n' "${SOURCE_ITERATION}" > "${candidate_tmp}/latest_checkpointed_iteration.txt"
  mv "${candidate_tmp}" "${CANDIDATE_LOAD_ROOT}"
}

configure_python_nvidia_libraries() {
  local library_dirs=() cuda_root directory site_packages joined
  if command -v nvcc >/dev/null 2>&1; then
    cuda_root="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)"
    for directory in "${cuda_root}/lib" "${cuda_root}/lib64" "${cuda_root}/targets/x86_64-linux/lib"; do
      [[ -d "${directory}" ]] && library_dirs+=("${directory}")
    done
  fi
  site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
  shopt -s nullglob
  for directory in "${site_packages}"/nvidia/*/lib; do
    [[ -d "${directory}" ]] && library_dirs+=("${directory}")
  done
  shopt -u nullglob
  (( ${#library_dirs[@]} > 0 )) || { echo "No CUDA library directories found" >&2; exit 1; }
  joined="$(IFS=:; echo "${library_dirs[*]}")"
  export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

current_iteration="$(tracker_iteration "${SAVE_DIR}")"
if [[ "${current_iteration}" == "${TRAIN_ITERS}" ]]; then
  echo "${EXPERIMENT_NAME} already complete at iteration ${TRAIN_ITERS}"
  exit 0
elif (( current_iteration >= 0 && current_iteration < TRAIN_ITERS )); then
  LOAD_CHECKPOINT="${SAVE_DIR}"
  FINETUNE=0
  LOAD_OPTIM=1
  LOAD_RNG=1
  RUN_KIND=resume
elif (( current_iteration > TRAIN_ITERS )); then
  echo "Existing checkpoint ${current_iteration} exceeds target ${TRAIN_ITERS}" >&2
  exit 1
else
  LOAD_CHECKPOINT="${CANDIDATE_LOAD_ROOT}"
  FINETUNE=1
  LOAD_OPTIM=0
  LOAD_RNG=0
  RUN_KIND=fresh
fi

if [[ "${DRY_RUN}" == "0" ]]; then
  if [[ "${FULL_VALIDATION}" == "1" ]]; then
    "${REPO_ROOT}/scripts/apply_megatron_full_validation_patch.sh"
  fi
  require_file "${ACTIVATE_SCRIPT}"
  require_file "${TRAIN_DATA}"
  require_file "${VALID_DATA}"
  prepare_candidate
  if [[ "${RUN_KIND}" == "fresh" && -e "${SAVE_DIR}" ]]; then
    echo "Refusing fresh run into existing save directory: ${SAVE_DIR}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${ACTIVATE_SCRIPT}"
  configure_python_nvidia_libraries
  python - <<'PY'
import ctypes
import torch

ctypes.CDLL("libcudnn_graph.so.9")
import transformer_engine.pytorch  # noqa: F401,E402

if torch.cuda.device_count() != 8:
    raise SystemExit(f"Expected 8 visible GPUs, found {torch.cuda.device_count()}")
PY
fi

base_args=()
[[ "${DRY_RUN}" == "1" ]] && base_args+=(--dry-run)
train_extra_args=()
[[ "${NO_DATA_SHARDING}" == "1" ]] && train_extra_args+=(--no-data-sharding)
[[ "${FULL_VALIDATION}" == "1" ]] && train_extra_args+=(--full-validation)

cmd=(env
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "NPROC_PER_NODE=${NPROC_PER_NODE}"
  "TP=${TP}" "PP=${PP}" "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
  "TRAIN_DATA=${TRAIN_DATA}" "VALID_DATA=${VALID_DATA}"
  "LOAD_CHECKPOINT=${LOAD_CHECKPOINT}" "SAVE_DIR=${SAVE_DIR}"
  "TRAIN_ITERS=${TRAIN_ITERS}" "LR_WARMUP_ITERS=${LR_WARMUP_ITERS}"
  "LR=${LR}" "MIN_LR=${MIN_LR}" "LR_DECAY_STYLE=${LR_DECAY_STYLE}"
  "LR_DECAY_ITERS=${LR_DECAY_ITERS}" "DATALOADER_TYPE=${DATALOADER_TYPE}"
  "CLIP_GRAD=${CLIP_GRAD}" "SAVE_INTERVAL=${SAVE_INTERVAL}"
  "EVAL_INTERVAL=${EVAL_INTERVAL}" "EVAL_ITERS=${EVAL_ITERS}"
  "LOG_INTERVAL=${LOG_INTERVAL}" "MASTER_PORT=${MASTER_PORT}"
  "FINETUNE=${FINETUNE}" "LOAD_OPTIM=${LOAD_OPTIM}" "LOAD_RNG=${LOAD_RNG}"
  "${REPO_ROOT}/scripts/train_phase2_qwen0p5b.sh"
  "${base_args[@]}"
  "${train_extra_args[@]}"
  --seed "${SEED}"
  --attention-backend fused
  --tensorboard-dir "${TENSORBOARD_DIR}"
  --tensorboard-log-interval "${TENSORBOARD_LOG_INTERVAL}"
  --log-timers-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --log-memory-to-tensorboard
  --log-memory-interval "${TENSORBOARD_MEMORY_INTERVAL}"
  --log-world-size-to-tensorboard
  --log-throughput)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run] mode=%s source_iteration=%s run_kind=%s\n' "${MODE}" "${SOURCE_ITERATION}" "${RUN_KIND}"
  printf '[dry-run] candidate=%q save=%q tensorboard=%q\n' "${CANDIDATE_LOAD_ROOT}" "${SAVE_DIR}" "${TENSORBOARD_DIR}"
  "${cmd[@]}"
  exit 0
fi

mkdir -p "${SAVE_DIR}" "${RUN_DIR}" "${TENSORBOARD_DIR}" "$(dirname "${LOG_PATH}")"
if [[ "${RUN_KIND}" == "fresh" ]]; then
  manifest="${RUN_DIR}/manifest.txt"
  {
    echo "experiment=${EXPERIMENT_NAME}"
    echo "created_at=$(date -u +%FT%TZ)"
    echo "repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "source_checkpoint=${SOURCE_ITER_DIR}"
    echo "candidate_load_root=${CANDIDATE_LOAD_ROOT}"
    echo "train_data=${TRAIN_DATA}"
    echo "valid_data=${VALID_DATA}"
    echo "train_iters=${TRAIN_ITERS}"
    echo "lr=${LR}"
    echo "min_lr=${MIN_LR}"
    echo "lr_warmup_iters=${LR_WARMUP_ITERS}"
    echo "lr_decay_style=${LR_DECAY_STYLE}"
    echo "lr_decay_iters=${LR_DECAY_ITERS}"
    echo "dataloader_type=${DATALOADER_TYPE}"
    echo "no_data_sharding=${NO_DATA_SHARDING}"
    echo "full_validation=${FULL_VALIDATION}"
    echo "clip_grad=${CLIP_GRAD}"
    echo "finetune=1"
    echo "load_optim=0"
    echo "load_rng=0"
    echo "seed=${SEED}"
    echo "tensorboard_dir=${TENSORBOARD_DIR}"
  } > "${manifest}"
fi

echo "[$(date -u +%FT%TZ)] starting ${EXPERIMENT_NAME} (${RUN_KIND})" | tee -a "${LOG_PATH}"
set +e
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
train_status=${PIPESTATUS[0]}
set -e
if (( train_status != 0 )); then
  echo "${EXPERIMENT_NAME} failed with status ${train_status}" | tee -a "${LOG_PATH}" >&2
  exit "${train_status}"
fi

actual_iteration="$(tracker_iteration "${SAVE_DIR}")"
[[ "${actual_iteration}" == "${TRAIN_ITERS}" ]] || {
  echo "Ended at ${actual_iteration}, expected ${TRAIN_ITERS}" | tee -a "${LOG_PATH}" >&2
  exit 1
}
printf 'completed_at=%s\niteration=%s\n' "$(date -u +%FT%TZ)" "${actual_iteration}" > "${RUN_DIR}/TRAINING_COMPLETE"
echo "[$(date -u +%FT%TZ)] completed ${EXPERIMENT_NAME}" | tee -a "${LOG_PATH}"
