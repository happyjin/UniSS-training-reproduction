#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --resume) RESUME=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen3b_unist198_phase1_full_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

configure_python_nvidia_libraries() {
  local library_dirs=() cuda_root directory site_packages joined
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

if [[ "${RESUME}" == "1" ]]; then
  LOAD_CHECKPOINT="${SAVE_DIR}"
  FINETUNE_FLAG=0
  LOAD_OPTIM_FLAG=1
  LOAD_RNG_FLAG=1
else
  FINETUNE_FLAG=1
  LOAD_OPTIM_FLAG=0
  LOAD_RNG_FLAG=0
fi

base_args=()
if [[ "${DRY_RUN}" == "1" ]]; then
  base_args+=(--dry-run)
fi

cmd=(env
  "USER_ROOT=${USER_ROOT}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "NPROC_PER_NODE=${NPROC_PER_NODE}"
  "TP=${TP}" "PP=${PP}"
  "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
  "TRAIN_DATA=${TRAIN_DATA}"
  "VALID_DATA=${VALID_DATA}"
  "LOAD_CHECKPOINT=${LOAD_CHECKPOINT}"
  "SAVE_DIR=${SAVE_DIR}"
  "TRAIN_ITERS=${TRAIN_ITERS}"
  "LR_WARMUP_ITERS=${LR_WARMUP_ITERS}"
  "SAVE_INTERVAL=${SAVE_INTERVAL}"
  "EVAL_INTERVAL=${EVAL_INTERVAL}"
  "EVAL_ITERS=${EVAL_ITERS}"
  "LOG_INTERVAL=${LOG_INTERVAL}"
  "MASTER_PORT=${MASTER_PORT}"
  "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
  "TENSORBOARD_LOG_INTERVAL=${TENSORBOARD_LOG_INTERVAL}"
  "TENSORBOARD_MEMORY_INTERVAL=${TENSORBOARD_MEMORY_INTERVAL}"
  "FINETUNE=${FINETUNE_FLAG}"
  "LOAD_OPTIM=${LOAD_OPTIM_FLAG}"
  "LOAD_RNG=${LOAD_RNG_FLAG}"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  "${REPO_ROOT}/scripts/train_phase1_qwen3b.sh"
  "${base_args[@]}"
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --save-retain-interval "${SAVE_RETAIN_INTERVAL}"
  --attention-backend fused
  --rerun-mode disabled
  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl native)

if [[ -n "${LR_DECAY_ITERS:-}" ]]; then
  cmd+=(--lr-decay-iters "${LR_DECAY_ITERS}")
fi
if [[ -n "${SEED:-}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ -n "${DATALOADER_TYPE:-}" ]]; then
  cmd+=(--dataloader-type "${DATALOADER_TYPE}")
fi

if [[ "${TP}" != "1" ]]; then
  cmd+=(--sequence-parallel)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

require_file "${ACTIVATE_SCRIPT}"
require_file "${TRAIN_DATA}"
require_file "${VALID_DATA}"
require_file "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt"

if [[ "${RESUME}" == "1" ]]; then
  latest="$(tr -d '[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
  [[ "${latest}" =~ ^[0-9]+$ && "${latest}" -gt 0 ]] || {
    echo "Cannot resume: ${SAVE_DIR}/latest_checkpointed_iteration.txt is '${latest}'" >&2
    exit 1
  }
  echo "Resuming from iteration ${latest} in ${SAVE_DIR}"
else
  if [[ -e "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "SAVE_DIR already has a checkpoint. Use --resume instead of overwriting: ${SAVE_DIR}" >&2
    exit 1
  fi
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

mkdir -p "${SAVE_DIR}" "${RUN_DIR}" "${TENSORBOARD_DIR}" "$(dirname "${LOG_PATH}")"
{
  echo "experiment=${EXPERIMENT_NAME}"
  echo "created_at=$(date -u +%FT%TZ)"
  echo "resume=${RESUME}"
  echo "train_iters=${TRAIN_ITERS}"
  echo "load_checkpoint=${LOAD_CHECKPOINT}"
  echo "save_dir=${SAVE_DIR}"
  echo "tensorboard_dir=${TENSORBOARD_DIR}"
  echo "tp=${TP}"
  echo "save_retain_interval=${SAVE_RETAIN_INTERVAL}"
  echo "backup_every=${BACKUP_EVERY}"
  echo "local_keep=${LOCAL_KEEP}"
  echo "s3_checkpoint_prefix=${S3_CHECKPOINT_PREFIX}"
  echo "lr=${LR}"
  echo "min_lr=${MIN_LR}"
  echo "lr_decay_style=${LR_DECAY_STYLE}"
  echo "lr_decay_iters=${LR_DECAY_ITERS:-}"
  echo "seed=${SEED:-}"
  echo "dataloader_type=${DATALOADER_TYPE:-}"
} | tee -a "${RUN_DIR}/manifest.txt"

echo "[$(date -u +%FT%TZ)] starting 3B Phase1 full training resume=${RESUME}" | tee -a "${LOG_PATH}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
echo "[$(date -u +%FT%TZ)] training process exited" | tee -a "${LOG_PATH}"
