#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
START_PHASE="${START_PHASE:-phase1}"
END_PHASE="${END_PHASE:-phase3}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --start-phase) START_PHASE="$2"; shift 2 ;;
    --end-phase) END_PHASE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen0p5b_unist198_full_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

case "${START_PHASE}" in
  phase1|phase2|phase3) ;;
  *) echo "START_PHASE must be phase1, phase2, or phase3" >&2; exit 2 ;;
esac
case "${END_PHASE}" in
  phase1|phase2|phase3) ;;
  *) echo "END_PHASE must be phase1, phase2, or phase3" >&2; exit 2 ;;
esac

phase_rank() {
  case "$1" in phase1) echo 1 ;; phase2) echo 2 ;; phase3) echo 3 ;; esac
}

START_RANK="$(phase_rank "${START_PHASE}")"
END_RANK="$(phase_rank "${END_PHASE}")"
if (( START_RANK > END_RANK )); then
  echo "START_PHASE=${START_PHASE} cannot be after END_PHASE=${END_PHASE}" >&2
  exit 2
fi
if [[ "${GLOBAL_BATCH_SIZE}" != "128" ]]; then
  echo "GLOBAL_BATCH_SIZE must remain 128 for this reproduction, got ${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi
if [[ "${NPROC_PER_NODE}" != "8" ]]; then
  echo "NPROC_PER_NODE must be 8 for this experiment, got ${NPROC_PER_NODE}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${ACTIVATE_SCRIPT}" ]] || { echo "Missing activation script: ${ACTIVATE_SCRIPT}" >&2; exit 1; }
  # shellcheck source=/dev/null
  source "${ACTIVATE_SCRIPT}"
fi

configure_python_nvidia_libraries() {
  local library_dirs=()
  local directory
  shopt -s nullglob
  for directory in "${ENV_ROOT}"/lib/python*/site-packages/nvidia/*/lib; do
    [[ -d "${directory}" ]] && library_dirs+=("${directory}")
  done
  shopt -u nullglob
  if (( ${#library_dirs[@]} == 0 )); then
    echo "No pip NVIDIA library directories found under ${ENV_ROOT}" >&2
    return 1
  fi
  local joined
  joined="$(IFS=:; echo "${library_dirs[*]}")"
  export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

if [[ "${DRY_RUN}" == "0" ]]; then
  configure_python_nvidia_libraries
  python - <<'PY'
import ctypes

ctypes.CDLL("libcudnn_graph.so.9")
import transformer_engine.pytorch  # noqa: F401,E402
PY
fi

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE TP PP MICRO_BATCH_SIZE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ceil_div() {
  echo $(( ($1 + $2 - 1) / $2 ))
}

read_count() {
  local phase="$1"
  local packed="$2"
  local override_name="${phase^^}_PACKED_COUNT_OVERRIDE"
  local override="${!override_name:-}"
  if [[ -n "${override}" ]]; then
    [[ "${override}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid ${override_name}: ${override}" >&2; exit 1; }
    echo "${override}"
    return
  fi
  local count_file="${packed}.count"
  [[ -s "${count_file}" ]] || { echo "Missing packed count sidecar: ${count_file}" >&2; exit 1; }
  local value
  value="$(<"${count_file}")"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid packed count in ${count_file}: ${value}" >&2; exit 1; }
  echo "${value}"
}

P1_COUNT="$(read_count phase1 "${PHASE1_TRAIN}")"
P2_COUNT=0
P3_COUNT=0
if (( END_RANK >= 2 )); then
  P2_COUNT="$(read_count phase2 "${PHASE2_TRAIN}")"
fi
if (( END_RANK >= 3 )); then
  P3_COUNT="$(read_count phase3 "${PHASE3_TRAIN}")"
fi

P1_EPOCH_ITERS="$(ceil_div "${P1_COUNT}" "${GLOBAL_BATCH_SIZE}")"
P2_EPOCH_ITERS=0
P3_EPOCH_ITERS=0
if (( END_RANK >= 2 )); then
  P2_EPOCH_ITERS="$(ceil_div "${P2_COUNT}" "${GLOBAL_BATCH_SIZE}")"
fi
if (( END_RANK >= 3 )); then
  P3_EPOCH_ITERS="$(ceil_div "${P3_COUNT}" "${GLOBAL_BATCH_SIZE}")"
fi
PHASE1_TRAIN_ITERS="${PHASE1_TRAIN_ITERS:-$((3 * P1_EPOCH_ITERS))}"
PHASE1_LR_WARMUP_ITERS="${PHASE1_LR_WARMUP_ITERS:-${P1_EPOCH_ITERS}}"
PHASE2_TRAIN_ITERS="${PHASE2_TRAIN_ITERS:-${P2_EPOCH_ITERS}}"
PHASE2_LR_WARMUP_ITERS="${PHASE2_LR_WARMUP_ITERS:-$(( (P2_EPOCH_ITERS + 19) / 20 ))}"
PHASE3_TRAIN_ITERS="${PHASE3_TRAIN_ITERS:-${P3_EPOCH_ITERS}}"
PHASE3_LR_WARMUP_ITERS="${PHASE3_LR_WARMUP_ITERS:-0}"
PHASE3_NINETY_PERCENT_ITER="$((PHASE3_TRAIN_ITERS * 9 / 10))"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

tracker_iteration() {
  local checkpoint="$1"
  local tracker="${checkpoint}/latest_checkpointed_iteration.txt"
  if [[ ! -f "${tracker}" ]]; then
    echo -1
    return
  fi
  local value
  value="$(tr -d '[:space:]' < "${tracker}")"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Invalid checkpoint tracker: ${tracker}" >&2; exit 1; }
  echo "${value}"
}

require_completed_checkpoint() {
  local checkpoint="$1"
  local expected="$2"
  local actual
  actual="$(tracker_iteration "${checkpoint}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Checkpoint ${checkpoint} is at iteration ${actual}, expected ${expected}" >&2
    exit 1
  fi
}

write_manifest() {
  local timestamp manifest
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  manifest="${RUN_DIR}/manifest_${timestamp}_$$.txt"
  mkdir -p "${RUN_DIR}" "${TENSORBOARD_ROOT}" "${REPO_ROOT}/logs"
  {
    echo "experiment=${EXPERIMENT_NAME}"
    echo "created_at=$(date -u +%FT%TZ)"
    echo "semantic_scope=Full UniST-198 public-data proxy; not strict paper data composition"
    echo "repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "repo_status=$(git -C "${REPO_ROOT}" status --short | tr '\n' ';')"
    echo "megatron_lm_commit=$(git -C "${REPO_ROOT}/third_party/Megatron-LM" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "nproc_per_node=${NPROC_PER_NODE}"
    echo "tp=${TP}"
    echo "pp=${PP}"
    echo "micro_batch_size=${MICRO_BATCH_SIZE}"
    echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
    echo "seq_length=${SEQ_LENGTH}"
    echo "ld_library_path=${LD_LIBRARY_PATH:-}"
    echo "start_phase=${START_PHASE}"
    echo "end_phase=${END_PHASE}"
    echo "phase1_packed=${PHASE1_TRAIN} count=${P1_COUNT} train_iters=${PHASE1_TRAIN_ITERS} warmup_iters=${PHASE1_LR_WARMUP_ITERS}"
    if (( END_RANK >= 2 )); then
      echo "phase2_packed=${PHASE2_TRAIN} count=${P2_COUNT} train_iters=${PHASE2_TRAIN_ITERS} warmup_iters=${PHASE2_LR_WARMUP_ITERS}"
    fi
    if (( END_RANK >= 3 )); then
      echo "phase3_packed=${PHASE3_TRAIN} count=${P3_COUNT} train_iters=${PHASE3_TRAIN_ITERS} warmup_iters=${PHASE3_LR_WARMUP_ITERS} ninety_percent_iter=${PHASE3_NINETY_PERCENT_ITER}"
    fi
    python - <<'PY'
import platform
import torch

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cudnn={torch.backends.cudnn.version()}")
print(f"gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"gpu_{index}={torch.cuda.get_device_name(index)}")
PY
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
  } > "${manifest}"
  ln -sfn "$(basename "${manifest}")" "${RUN_DIR}/manifest_latest.txt"
  echo "Run manifest: ${manifest}"
}

if [[ "${DRY_RUN}" == "0" ]]; then
  if (( END_RANK >= 3 )); then
    require_file "${PACKING_COMPLETE_MARKER}"
  fi
  if (( START_RANK <= 1 && END_RANK >= 1 )); then
    require_file "${PHASE1_TRAIN}"
    require_file "${PHASE1_VALID}"
    require_file "${BASE_CHECKPOINT}/latest_checkpointed_iteration.txt"
  fi
  if (( START_RANK <= 2 && END_RANK >= 2 )); then
    require_file "${PHASE2_TRAIN}"
    require_file "${PHASE2_VALID}"
  fi
  if (( START_RANK <= 3 && END_RANK >= 3 )); then
    require_file "${PHASE3_TRAIN}"
    require_file "${PHASE3_VALID}"
  fi
  gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
  [[ "${gpu_count}" == "8" ]] || { echo "Expected 8 visible GPUs, found ${gpu_count}" >&2; exit 1; }
  write_manifest
else
  echo "[dry-run] ${EXPERIMENT_NAME}: START=${START_PHASE}, END=${END_PHASE}, P1=${P1_COUNT}, P2=${P2_COUNT}, P3=${P3_COUNT}"
  echo "[dry-run] schedule: phase1=${PHASE1_TRAIN_ITERS}/${PHASE1_LR_WARMUP_ITERS}, phase2=${PHASE2_TRAIN_ITERS}/${PHASE2_LR_WARMUP_ITERS}, phase3=${PHASE3_TRAIN_ITERS}/${PHASE3_LR_WARMUP_ITERS}"
fi

run_phase() {
  local phase="$1" script="$2" train_data="$3" valid_data="$4"
  local previous_checkpoint="$5" save_dir="$6" log_path="$7"
  local target_iters="$8" warmup_iters="$9" master_port="${10}"
  local tb_dir="${TENSORBOARD_ROOT}/${phase}"
  local rank
  rank="$(phase_rank "${phase}")"

  if (( rank > END_RANK )); then
    echo "${phase} skipped because END_PHASE=${END_PHASE}"
    return 0
  fi

  if (( rank < START_RANK )); then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] skip ${phase} because START_PHASE=${START_PHASE}"
      return 0
    fi
    require_completed_checkpoint "${save_dir}" "${target_iters}"
    echo "${phase} already required complete by START_PHASE=${START_PHASE}"
    return 0
  fi

  local current_iter load_checkpoint finetune load_optim load_rng
  current_iter="$(tracker_iteration "${save_dir}")"
  if [[ "${current_iter}" == "${target_iters}" ]]; then
    echo "${phase} already complete at iteration ${target_iters}; skipping"
    return 0
  elif (( current_iter >= 0 && current_iter < target_iters )); then
    load_checkpoint="${save_dir}"
    finetune=0
    load_optim=1
    load_rng=1
    echo "${phase} will resume from iteration ${current_iter}"
  elif (( current_iter > target_iters )); then
    echo "${phase} checkpoint iteration ${current_iter} exceeds target ${target_iters}" >&2
    exit 1
  else
    load_checkpoint="${previous_checkpoint}"
    finetune=1
    load_optim=0
    load_rng=0
    echo "${phase} will start from ${previous_checkpoint} with optimizer/RNG reset"
  fi

  local cmd=(env
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "NPROC_PER_NODE=${NPROC_PER_NODE}"
    "TP=${TP}" "PP=${PP}" "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
    "TRAIN_DATA=${train_data}" "VALID_DATA=${valid_data}"
    "LOAD_CHECKPOINT=${load_checkpoint}" "SAVE_DIR=${save_dir}"
    "TRAIN_ITERS=${target_iters}" "LR_WARMUP_ITERS=${warmup_iters}"
    "SAVE_INTERVAL=${SAVE_INTERVAL}" "EVAL_INTERVAL=${EVAL_INTERVAL}"
    "EVAL_ITERS=${EVAL_ITERS}" "LOG_INTERVAL=${LOG_INTERVAL}"
    "MASTER_PORT=${master_port}" "FINETUNE=${finetune}"
    "LOAD_OPTIM=${load_optim}" "LOAD_RNG=${load_rng}"
    "${script}"
    --attention-backend flash
    --tensorboard-dir "${tb_dir}"
    --tensorboard-log-interval "${TENSORBOARD_LOG_INTERVAL}"
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --log-memory-interval "${TENSORBOARD_MEMORY_INTERVAL}"
    --log-world-size-to-tensorboard
    --log-throughput)

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[%s] ' "${phase}"
    printf '%q ' "${cmd[@]}"
    printf '> %q 2>&1\n' "${log_path}"
    return 0
  fi

  mkdir -p "${save_dir}" "${tb_dir}" "$(dirname "${log_path}")"
  echo "[$(date -u +%FT%TZ)] starting ${phase}; target=${target_iters}" | tee -a "${log_path}"
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  require_completed_checkpoint "${save_dir}" "${target_iters}"
  echo "[$(date -u +%FT%TZ)] completed ${phase} at iteration ${target_iters}" | tee -a "${log_path}"
}

run_phase phase1 "${REPO_ROOT}/scripts/train_phase1_qwen0p5b.sh" \
  "${PHASE1_TRAIN}" "${PHASE1_VALID}" "${BASE_CHECKPOINT}" "${PHASE1_SAVE}" \
  "${PHASE1_LOG}" "${PHASE1_TRAIN_ITERS}" "${PHASE1_LR_WARMUP_ITERS}" "${PHASE1_MASTER_PORT}"

run_phase phase2 "${REPO_ROOT}/scripts/train_phase2_qwen0p5b.sh" \
  "${PHASE2_TRAIN}" "${PHASE2_VALID}" "${PHASE1_SAVE}" "${PHASE2_SAVE}" \
  "${PHASE2_LOG}" "${PHASE2_TRAIN_ITERS}" "${PHASE2_LR_WARMUP_ITERS}" "${PHASE2_MASTER_PORT}"

run_phase phase3 "${REPO_ROOT}/scripts/train_phase3_qwen0p5b.sh" \
  "${PHASE3_TRAIN}" "${PHASE3_VALID}" "${PHASE2_SAVE}" "${PHASE3_SAVE}" \
  "${PHASE3_LOG}" "${PHASE3_TRAIN_ITERS}" "${PHASE3_LR_WARMUP_ITERS}" "${PHASE3_MASTER_PORT}"

if [[ "${DRY_RUN}" == "0" && "${END_PHASE}" == "phase3" ]]; then
  completion_tmp="${RUN_DIR}/TRAINING_COMPLETE.tmp.$$"
  {
    echo "completed_at=$(date -u +%FT%TZ)"
    echo "phase1_iteration=${PHASE1_TRAIN_ITERS}"
    echo "phase2_iteration=${PHASE2_TRAIN_ITERS}"
    echo "phase3_iteration=${PHASE3_TRAIN_ITERS}"
  } > "${completion_tmp}"
  mv "${completion_tmp}" "${RUN_DIR}/TRAINING_COMPLETE"
  echo "All three phases complete: ${RUN_DIR}/TRAINING_COMPLETE"
elif [[ "${DRY_RUN}" == "0" ]]; then
  stage_marker="${RUN_DIR}/${END_PHASE^^}_TRAINING_COMPLETE"
  case "${END_PHASE}" in
    phase1) stage_save="${PHASE1_SAVE}" ;;
    phase2) stage_save="${PHASE2_SAVE}" ;;
  esac
  printf 'completed_at=%s\niteration=%s\n' \
    "$(date -u +%FT%TZ)" \
    "$(tracker_iteration "${stage_save}")" > "${stage_marker}.tmp.$$"
  mv "${stage_marker}.tmp.$$" "${stage_marker}"
  echo "Training through ${END_PHASE} complete: ${stage_marker}"
fi
