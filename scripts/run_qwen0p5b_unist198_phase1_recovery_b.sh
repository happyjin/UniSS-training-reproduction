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
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen0p5b_unist198_phase1_recovery_b1_v2.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

if [[ "${NPROC_PER_NODE}" != "8" ]]; then
  echo "Experiment B requires 8 GPUs, got NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi
if [[ "${GLOBAL_BATCH_SIZE}" != "128" ]]; then
  echo "Experiment B must preserve global batch size 128, got ${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi
if [[ "${SOURCE_ITERATION}" != "3300" ]]; then
  echo "Experiment B must start from iteration 3300, got ${SOURCE_ITERATION}" >&2
  exit 1
fi

SOURCE_ITER_DIR="${SOURCE_CHECKPOINT_ROOT}/iter_$(printf '%07d' "${SOURCE_ITERATION}")"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

tracker_iteration() {
  local checkpoint="$1" tracker value
  tracker="${checkpoint}/latest_checkpointed_iteration.txt"
  [[ -f "${tracker}" ]] || { echo -1; return; }
  value="$(tr -d '[:space:]' < "${tracker}")"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Invalid checkpoint tracker: ${tracker}" >&2; exit 1; }
  echo "${value}"
}

prepare_iteration_candidate() {
  local expected_shards candidate_tmp
  require_file "${SOURCE_ITER_DIR}/metadata.json"
  require_file "${SOURCE_ITER_DIR}/.metadata"
  expected_shards="$(find "${SOURCE_ITER_DIR}" -maxdepth 1 -name '__*_0.distcp' -type f | wc -l)"
  [[ "${expected_shards}" == "8" ]] || {
    echo "Expected 8 distributed checkpoint shards in ${SOURCE_ITER_DIR}, found ${expected_shards}" >&2
    exit 1
  }

  if [[ -e "${CANDIDATE_LOAD_ROOT}" ]]; then
    [[ "$(tracker_iteration "${CANDIDATE_LOAD_ROOT}")" == "${SOURCE_ITERATION}" ]] || {
      echo "Existing candidate does not point to iteration ${SOURCE_ITERATION}: ${CANDIDATE_LOAD_ROOT}" >&2
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
  echo "Prepared read-only iteration candidate with hard links: ${CANDIDATE_LOAD_ROOT}"
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
  (( ${#library_dirs[@]} > 0 )) || { echo "No CUDA or pip NVIDIA library directories found" >&2; exit 1; }
  joined="$(IFS=:; echo "${library_dirs[*]}")"
  export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

if [[ "${DRY_RUN}" == "0" ]]; then
  require_file "${ACTIVATE_SCRIPT}"
  require_file "${TRAIN_DATA}"
  require_file "${VALID_DATA}"
  prepare_iteration_candidate

  for output in "${SAVE_DIR}" "${RUN_DIR}" "${TENSORBOARD_DIR}" "${LOG_PATH}"; do
    [[ ! -e "${output}" ]] || {
      echo "Refusing to overwrite existing experiment-B output: ${output}" >&2
      exit 1
    }
  done

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
if [[ "${DRY_RUN}" == "1" ]]; then
  base_args+=(--dry-run)
fi

cmd=(env
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "NPROC_PER_NODE=${NPROC_PER_NODE}"
  "TP=${TP}" "PP=${PP}"
  "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
  "TRAIN_DATA=${TRAIN_DATA}"
  "VALID_DATA=${VALID_DATA}"
  "LOAD_CHECKPOINT=${CANDIDATE_LOAD_ROOT}"
  "SAVE_DIR=${SAVE_DIR}"
  "TRAIN_ITERS=${TRAIN_ITERS}"
  "LR_WARMUP_ITERS=${LR_WARMUP_ITERS}"
  "SAVE_INTERVAL=${SAVE_INTERVAL}"
  "EVAL_INTERVAL=${EVAL_INTERVAL}"
  "EVAL_ITERS=${EVAL_ITERS}"
  "LOG_INTERVAL=${LOG_INTERVAL}"
  "MASTER_PORT=${MASTER_PORT}"
  "FINETUNE=1"
  "LOAD_OPTIM=0"
  "LOAD_RNG=0"
  "${REPO_ROOT}/scripts/train_phase1_qwen0p5b.sh"
  "${base_args[@]}"
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --lr-warmup-iters "${LR_WARMUP_ITERS}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --lr-decay-iters "${LR_DECAY_ITERS}"
  --seed "${SEED}"
  --dataloader-type cyclic
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
  printf '[dry-run] source_iteration=%s candidate=%q\n' "${SOURCE_ITERATION}" "${CANDIDATE_LOAD_ROOT}"
  printf '[dry-run] '
  printf '%q ' \
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    "NPROC_PER_NODE=${NPROC_PER_NODE}" \
    "TRAIN_ITERS=${TRAIN_ITERS}" \
    "FINETUNE=1" "LOAD_OPTIM=0" "LOAD_RNG=0"
  printf '\n'
  printf '[dry-run] FINETUNE=1 LOAD_OPTIM=0 LOAD_RNG=0 shuffled_by=dataloader-type-cyclic\n'
  "${cmd[@]}"
  exit 0
fi

mkdir -p "${SAVE_DIR}" "${RUN_DIR}" "${TENSORBOARD_DIR}" "$(dirname "${LOG_PATH}")"
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
  echo "dataloader_type=cyclic"
  echo "shuffle=true"
  echo "validation_mode=sampled"
  echo "eval_interval=${EVAL_INTERVAL}"
  echo "eval_iters=${EVAL_ITERS}"
  echo "finetune=1"
  echo "load_optim=0"
  echo "load_rng=0"
  echo "seed=${SEED}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "micro_batch_size=${MICRO_BATCH_SIZE}"
  echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
  echo "tensorboard_dir=${TENSORBOARD_DIR}"
} > "${manifest}"

echo "[$(date -u +%FT%TZ)] starting ${EXPERIMENT_NAME}" | tee -a "${LOG_PATH}"
echo "Manifest: ${manifest}" | tee -a "${LOG_PATH}"
set +e
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
train_status=${PIPESTATUS[0]}
set -e
if (( train_status != 0 )); then
  echo "Experiment B failed with status ${train_status}" | tee -a "${LOG_PATH}" >&2
  exit "${train_status}"
fi

actual_iteration="$(tracker_iteration "${SAVE_DIR}")"
if [[ "${actual_iteration}" != "${TRAIN_ITERS}" ]]; then
  echo "Experiment B ended at iteration ${actual_iteration}, expected ${TRAIN_ITERS}" | tee -a "${LOG_PATH}" >&2
  exit 1
fi
printf 'completed_at=%s\niteration=%s\n' "$(date -u +%FT%TZ)" "${actual_iteration}" > "${RUN_DIR}/TRAINING_COMPLETE"
echo "[$(date -u +%FT%TZ)] completed ${EXPERIMENT_NAME} at iteration ${actual_iteration}" | tee -a "${LOG_PATH}"
