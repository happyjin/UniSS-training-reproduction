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
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen0p5b_unist198_phase2_from_phase1_v4.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

[[ "${NPROC_PER_NODE}" == "8" ]] || { echo "Phase2 v4 requires 8 GPUs" >&2; exit 1; }
[[ "${GLOBAL_BATCH_SIZE}" == "128" ]] || { echo "Phase2 v4 requires global batch size 128" >&2; exit 1; }
[[ "${SOURCE_ITERATION}" == "${EXPECTED_SOURCE_ITERATION}" ]] || {
  echo "Source ${SOURCE_ITERATION} does not match expected Phase1 iteration ${EXPECTED_SOURCE_ITERATION}" >&2
  exit 1
}
[[ "${TRAIN_ITERS}" == "15381" ]] || { echo "Phase2 v4 must cover one full 15381-step epoch" >&2; exit 1; }
(( PILOT_EXIT_ITERATION > 0 && PILOT_EXIT_ITERATION < TRAIN_ITERS )) || {
  echo "PILOT_EXIT_ITERATION must be between 1 and TRAIN_ITERS" >&2
  exit 1
}
(( LR_DECAY_ITERS <= PILOT_EXIT_ITERATION )) || {
  echo "LR_DECAY_ITERS must reach the low-LR regime by the pilot boundary" >&2
  exit 1
}
[[ "${DATALOADER_TYPE}" == "cyclic" ]] || { echo "Phase2 v4 requires cyclic global shuffle" >&2; exit 1; }
[[ "${NO_DATA_SHARDING}" == "1" ]] || { echo "Phase2 v4 requires --no-data-sharding" >&2; exit 1; }
[[ "${FULL_VALIDATION}" == "1" ]] || { echo "Phase2 v4 requires exact full validation" >&2; exit 1; }
[[ "${EVAL_MICRO_BATCH_SIZE}" == "1" ]] || { echo "Full validation requires eval micro batch 1" >&2; exit 1; }
[[ "${EVAL_GLOBAL_BATCH_SIZE}" == "${NPROC_PER_NODE}" ]] || {
  echo "Full validation requires eval global batch ${NPROC_PER_NODE}" >&2
  exit 1
}

SOURCE_ITER_DIR="${SOURCE_CHECKPOINT_ROOT}/iter_$(printf '%07d' "${SOURCE_ITERATION}")"
PILOT_GATE_OUTPUT="${RUN_DIR}/PILOT_GATE.json"
PILOT_GATE_MARKER="${RUN_DIR}/PILOT_GATE_PASSED"
FINAL_GATE_OUTPUT="${RUN_DIR}/FINAL_GATE.json"
FINAL_GATE_MARKER="${RUN_DIR}/FINAL_GATE_PASSED"
TRAINING_COMPLETE="${RUN_DIR}/TRAINING_COMPLETE"

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
  [[ "${shard_count}" == "8" ]] || { echo "Expected 8 Phase1 checkpoint shards, found ${shard_count}" >&2; exit 1; }
  if [[ -e "${CANDIDATE_LOAD_ROOT}" ]]; then
    [[ "$(tracker_iteration "${CANDIDATE_LOAD_ROOT}")" == "${SOURCE_ITERATION}" ]] || {
      echo "Candidate does not point to Phase1 iteration ${SOURCE_ITERATION}" >&2
      exit 1
    }
    require_file "${CANDIDATE_LOAD_ROOT}/$(basename "${SOURCE_ITER_DIR}")/metadata.json"
    return
  fi
  mkdir -p "$(dirname "${CANDIDATE_LOAD_ROOT}")"
  candidate_tmp="${CANDIDATE_LOAD_ROOT}.tmp.$$"
  [[ ! -e "${candidate_tmp}" ]] || { echo "Candidate temporary path already exists: ${candidate_tmp}" >&2; exit 1; }
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

build_train_command() {
  local load_checkpoint="$1" finetune="$2" load_optim="$3" load_rng="$4" use_pilot_exit="$5"
  local base_args=() extra_args=(
    --no-data-sharding
    --full-validation
    --eval-micro-batch-size "${EVAL_MICRO_BATCH_SIZE}"
    --eval-global-batch-size "${EVAL_GLOBAL_BATCH_SIZE}"
  )
  [[ "${DRY_RUN}" == "1" ]] && base_args+=(--dry-run)
  [[ "${use_pilot_exit}" == "1" ]] && extra_args+=(--exit-interval "${PILOT_EXIT_ITERATION}")

  TRAIN_COMMAND=(env
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "NPROC_PER_NODE=${NPROC_PER_NODE}"
    "TP=${TP}" "PP=${PP}" "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
    "TRAIN_DATA=${TRAIN_DATA}" "VALID_DATA=${VALID_DATA}"
    "LOAD_CHECKPOINT=${load_checkpoint}" "SAVE_DIR=${SAVE_DIR}"
    "TRAIN_ITERS=${TRAIN_ITERS}" "LR_WARMUP_ITERS=${LR_WARMUP_ITERS}"
    "LR=${LR}" "MIN_LR=${MIN_LR}" "LR_DECAY_STYLE=${LR_DECAY_STYLE}"
    "LR_DECAY_ITERS=${LR_DECAY_ITERS}" "DATALOADER_TYPE=${DATALOADER_TYPE}"
    "CLIP_GRAD=${CLIP_GRAD}" "SAVE_INTERVAL=${SAVE_INTERVAL}"
    "EVAL_INTERVAL=${EVAL_INTERVAL}" "EVAL_ITERS=${EVAL_ITERS}"
    "LOG_INTERVAL=${LOG_INTERVAL}" "MASTER_PORT=${MASTER_PORT}"
    "FINETUNE=${finetune}" "LOAD_OPTIM=${load_optim}" "LOAD_RNG=${load_rng}"
    "${REPO_ROOT}/scripts/train_phase2_qwen0p5b.sh"
    "${base_args[@]}"
    "${extra_args[@]}"
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
}

run_train_command() {
  local label="$1"
  echo "[$(date -u +%FT%TZ)] starting ${EXPERIMENT_NAME} ${label}" | tee -a "${LOG_PATH}"
  set +e
  "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${LOG_PATH}"
  local status=${PIPESTATUS[0]}
  set -e
  (( status == 0 )) || {
    echo "${EXPERIMENT_NAME} ${label} failed with status ${status}" | tee -a "${LOG_PATH}" >&2
    exit "${status}"
  }
}

run_gate() {
  local required_step="$1" max_valid="$2" max_last="$3" grad_threshold="$4"
  local max_consecutive="$5" absolute_max="$6" max_last_minus_best="$7" output="$8"
  "${ENV_ROOT}/bin/python" -m training.validate_phase2_recovery \
    --tensorboard-dir "${TENSORBOARD_DIR}" \
    --log "${LOG_PATH}" \
    --required-step "${required_step}" \
    --max-valid-loss "${max_valid}" \
    --max-last-valid-loss "${max_last}" \
    --max-last-minus-best "${max_last_minus_best}" \
    --grad-spike-threshold "${grad_threshold}" \
    --max-consecutive-grad-spikes "${max_consecutive}" \
    --absolute-max-grad-norm "${absolute_max}" \
    --output "${output}"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] clean Phase1 source iteration=${SOURCE_ITERATION}; no Phase2 recovery checkpoint"
  echo "[dry-run] one continuous Phase2 run target=${TRAIN_ITERS}; fixed gate=${PILOT_EXIT_ITERATION}; no online early stop"
  echo "[dry-run] initial FINETUNE=1 LOAD_OPTIM=0 LOAD_RNG=0"
  build_train_command "${CANDIDATE_LOAD_ROOT}" 1 0 0 1
  "${TRAIN_COMMAND[@]}"
  echo "[dry-run] validate step ${PILOT_EXIT_ITERATION}, then preserve optimizer/RNG/data cursor"
  echo "[dry-run] continuation FINETUNE=0 LOAD_OPTIM=1 LOAD_RNG=1"
  build_train_command "${SAVE_DIR}" 0 1 1 0
  "${TRAIN_COMMAND[@]}"
  echo "[dry-run] TensorBoard=${TENSORBOARD_DIR} port=${TENSORBOARD_PORT}"
  exit 0
fi

"${REPO_ROOT}/scripts/apply_megatron_full_validation_patch.sh"
require_file "${ACTIVATE_SCRIPT}"
require_file "${TRAIN_DATA}"
require_file "${TRAIN_DATA}.count"
require_file "${VALID_DATA}"
[[ "$(tr -d '[:space:]' < "${TRAIN_DATA}.count")" == "${EXPECTED_TRAIN_PACKED_COUNT}" ]] || {
  echo "Unexpected Phase2 packed count" >&2
  exit 1
}
prepare_candidate

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

mkdir -p "${SAVE_DIR}" "${RUN_DIR}" "${TENSORBOARD_DIR}" "$(dirname "${LOG_PATH}")"
current_iteration="$(tracker_iteration "${SAVE_DIR}")"
if (( current_iteration < 0 )); then
  if [[ -n "$(find "${SAVE_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing fresh run into non-empty save directory: ${SAVE_DIR}" >&2
    exit 1
  fi
  {
    echo "experiment=${EXPERIMENT_NAME}"
    echo "created_at=$(date -u +%FT%TZ)"
    echo "repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "source_checkpoint=${SOURCE_ITER_DIR}"
    echo "candidate_load_root=${CANDIDATE_LOAD_ROOT}"
    echo "train_data=${TRAIN_DATA}"
    echo "valid_data=${VALID_DATA}"
    echo "train_iters=${TRAIN_ITERS}"
    echo "pilot_exit_iteration=${PILOT_EXIT_ITERATION}"
    echo "lr=${LR}"
    echo "min_lr=${MIN_LR}"
    echo "lr_warmup_iters=${LR_WARMUP_ITERS}"
    echo "lr_decay_style=${LR_DECAY_STYLE}"
    echo "lr_decay_iters=${LR_DECAY_ITERS}"
    echo "dataloader_type=${DATALOADER_TYPE}"
    echo "no_data_sharding=${NO_DATA_SHARDING}"
    echo "full_validation=${FULL_VALIDATION}"
    echo "eval_micro_batch_size=${EVAL_MICRO_BATCH_SIZE}"
    echo "eval_global_batch_size=${EVAL_GLOBAL_BATCH_SIZE}"
    echo "clip_grad=${CLIP_GRAD}"
    echo "finetune=1"
    echo "load_optim=0"
    echo "load_rng=0"
    echo "seed=${SEED}"
    echo "tensorboard_dir=${TENSORBOARD_DIR}"
  } > "${RUN_DIR}/manifest.txt"
  build_train_command "${CANDIDATE_LOAD_ROOT}" 1 0 0 1
  run_train_command "fresh prefix 0-${PILOT_EXIT_ITERATION}"
  current_iteration="$(tracker_iteration "${SAVE_DIR}")"
elif (( current_iteration < PILOT_EXIT_ITERATION )); then
  build_train_command "${SAVE_DIR}" 0 1 1 1
  run_train_command "resume prefix ${current_iteration}-${PILOT_EXIT_ITERATION}"
  current_iteration="$(tracker_iteration "${SAVE_DIR}")"
fi

if (( current_iteration == PILOT_EXIT_ITERATION )) && [[ ! -f "${PILOT_GATE_MARKER}" ]]; then
  echo "[$(date -u +%FT%TZ)] running fixed ${PILOT_EXIT_ITERATION}-step health gate" | tee -a "${LOG_PATH}"
  run_gate "${PILOT_EXIT_ITERATION}" "${PILOT_MAX_VALID_LOSS}" "${PILOT_MAX_LAST_VALID_LOSS}" \
    "${PILOT_GRAD_SPIKE_THRESHOLD}" "${PILOT_MAX_CONSECUTIVE_GRAD_SPIKES}" \
    "${PILOT_ABSOLUTE_MAX_GRAD_NORM}" "${PILOT_MAX_LAST_MINUS_BEST}" "${PILOT_GATE_OUTPUT}"
  printf 'passed_at=%s\niteration=%s\n' "$(date -u +%FT%TZ)" "${PILOT_EXIT_ITERATION}" > "${PILOT_GATE_MARKER}"
fi

if (( current_iteration > PILOT_EXIT_ITERATION && current_iteration < TRAIN_ITERS )) && [[ ! -f "${PILOT_GATE_MARKER}" ]]; then
  echo "Checkpoint ${current_iteration} passed the pilot boundary without a gate marker" >&2
  exit 1
fi
if (( current_iteration == PILOT_EXIT_ITERATION )); then
  require_file "${PILOT_GATE_MARKER}"
  build_train_command "${SAVE_DIR}" 0 1 1 0
  run_train_command "continuous resume ${PILOT_EXIT_ITERATION}-${TRAIN_ITERS}"
  current_iteration="$(tracker_iteration "${SAVE_DIR}")"
elif (( current_iteration > PILOT_EXIT_ITERATION && current_iteration < TRAIN_ITERS )); then
  build_train_command "${SAVE_DIR}" 0 1 1 0
  run_train_command "resume ${current_iteration}-${TRAIN_ITERS}"
  current_iteration="$(tracker_iteration "${SAVE_DIR}")"
fi

[[ "${current_iteration}" == "${TRAIN_ITERS}" ]] || {
  echo "Phase2 v4 ended at ${current_iteration}, expected ${TRAIN_ITERS}" | tee -a "${LOG_PATH}" >&2
  exit 1
}
require_file "${PILOT_GATE_MARKER}"
if [[ ! -f "${FINAL_GATE_MARKER}" ]]; then
  echo "[$(date -u +%FT%TZ)] running final Phase2 health gate" | tee -a "${LOG_PATH}"
  run_gate "${TRAIN_ITERS}" "${FINAL_MAX_VALID_LOSS}" "${FINAL_MAX_LAST_VALID_LOSS}" \
    "${FINAL_GRAD_SPIKE_THRESHOLD}" "${FINAL_MAX_CONSECUTIVE_GRAD_SPIKES}" \
    "${FINAL_ABSOLUTE_MAX_GRAD_NORM}" "${FINAL_MAX_LAST_MINUS_BEST}" "${FINAL_GATE_OUTPUT}"
  printf 'passed_at=%s\niteration=%s\n' "$(date -u +%FT%TZ)" "${TRAIN_ITERS}" > "${FINAL_GATE_MARKER}"
fi
printf 'completed_at=%s\niteration=%s\nsource_phase1_iteration=%s\n' \
  "$(date -u +%FT%TZ)" "${TRAIN_ITERS}" "${SOURCE_ITERATION}" > "${TRAINING_COMPLETE}"
echo "[$(date -u +%FT%TZ)] completed healthy ${EXPERIMENT_NAME}" | tee -a "${LOG_PATH}"
