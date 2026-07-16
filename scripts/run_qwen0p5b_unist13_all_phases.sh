#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
ENV_ROOT="${ENV_ROOT:-${USER_ROOT}/conda_envs/uniss-train}"

export PATH="${ENV_ROOT}/bin:${PATH}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/cache/pip}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TP="${TP:-1}"
export PP="${PP:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${REPO_ROOT}/logs" "${REPO_ROOT}/runs"

PHASE1_TRAIN="${PHASE1_TRAIN:-${REPO_ROOT}/data/megatron/phase1_unist13/packed_train.jsonl}"
PHASE1_VALID="${PHASE1_VALID:-${REPO_ROOT}/data/megatron/validation_unist_dev/phase1_valid_packed.jsonl}"
PHASE2_TRAIN="${PHASE2_TRAIN:-${REPO_ROOT}/data/megatron/phase2_unist13_mix/packed_train.jsonl}"
PHASE2_VALID="${PHASE2_VALID:-${REPO_ROOT}/data/megatron/validation_unist_dev/phase2_valid_packed.jsonl}"
PHASE3_TRAIN="${PHASE3_TRAIN:-${PHASE2_TRAIN}}"
PHASE3_VALID="${PHASE3_VALID:-${PHASE2_VALID}}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-${REPO_ROOT}/checkpoints/qwen2_0p5b_uniss_vocab}"
PHASE1_SAVE="${PHASE1_SAVE:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase1_unist13_full}"
PHASE2_SAVE="${PHASE2_SAVE:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase2_unist13_full}"
PHASE3_SAVE="${PHASE3_SAVE:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase3_unist13_full}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_checkpoint() {
  local path="$1"
  if [[ ! -f "${path}/latest_checkpointed_iteration.txt" ]]; then
    echo "Missing Megatron checkpoint tracker: ${path}/latest_checkpointed_iteration.txt" >&2
    exit 1
  fi
}

if [[ "${DRY_RUN}" == "0" ]]; then
  require_file "${PHASE1_TRAIN}"
  require_file "${PHASE1_VALID}"
  require_file "${PHASE2_TRAIN}"
  require_file "${PHASE2_VALID}"
  require_file "${PHASE3_TRAIN}"
  require_file "${PHASE3_VALID}"
  require_checkpoint "${BASE_CHECKPOINT}"
fi

run_phase() {
  local phase="$1"
  local script="$2"
  local train_data="$3"
  local valid_data="$4"
  local load_checkpoint="$5"
  local save_dir="$6"
  local log_path="$7"
  shift 7

  local cmd=(
    env
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "NPROC_PER_NODE=${NPROC_PER_NODE}"
    "TP=${TP}"
    "PP=${PP}"
    "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
    "TRAIN_DATA=${train_data}"
    "VALID_DATA=${valid_data}"
    "LOAD_CHECKPOINT=${load_checkpoint}"
    "SAVE_DIR=${save_dir}"
    "$@"
    "${script}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[%s] ' "${phase}"
    printf '%q ' "${cmd[@]}"
    printf '> %q 2>&1\n' "${log_path}"
    return 0
  fi

  echo "[$(date -u +%FT%TZ)] starting ${phase}" | tee -a "${log_path}"
  "${cmd[@]}" >> "${log_path}" 2>&1
  echo "[$(date -u +%FT%TZ)] finished ${phase}" | tee -a "${log_path}"
}

run_phase "phase1" "${REPO_ROOT}/scripts/train_phase1_qwen0p5b.sh" \
  "${PHASE1_TRAIN}" "${PHASE1_VALID}" "${BASE_CHECKPOINT}" "${PHASE1_SAVE}" \
  "${REPO_ROOT}/logs/uniss_qwen0p5b_phase1_unist13_full.log" \
  "TRAIN_ITERS=${PHASE1_TRAIN_ITERS:-1269}" \
  "LR_WARMUP_ITERS=${PHASE1_LR_WARMUP_ITERS:-423}" \
  "SAVE_INTERVAL=${SAVE_INTERVAL:-100}" \
  "EVAL_INTERVAL=${EVAL_INTERVAL:-100}" \
  "EVAL_ITERS=${EVAL_ITERS:-10}" \
  "LOG_INTERVAL=${LOG_INTERVAL:-10}" \
  "MASTER_PORT=${PHASE1_MASTER_PORT:-29511}"

run_phase "phase2" "${REPO_ROOT}/scripts/train_phase2_qwen0p5b.sh" \
  "${PHASE2_TRAIN}" "${PHASE2_VALID}" "${PHASE1_SAVE}" "${PHASE2_SAVE}" \
  "${REPO_ROOT}/logs/uniss_qwen0p5b_phase2_unist13_full.log" \
  "TRAIN_ITERS=${PHASE2_TRAIN_ITERS:-1045}" \
  "LR_WARMUP_ITERS=${PHASE2_LR_WARMUP_ITERS:-53}" \
  "SAVE_INTERVAL=${SAVE_INTERVAL:-100}" \
  "EVAL_INTERVAL=${EVAL_INTERVAL:-100}" \
  "EVAL_ITERS=${EVAL_ITERS:-10}" \
  "LOG_INTERVAL=${LOG_INTERVAL:-10}" \
  "MASTER_PORT=${PHASE2_MASTER_PORT:-29512}"

run_phase "phase3" "${REPO_ROOT}/scripts/train_phase3_qwen0p5b.sh" \
  "${PHASE3_TRAIN}" "${PHASE3_VALID}" "${PHASE2_SAVE}" "${PHASE3_SAVE}" \
  "${REPO_ROOT}/logs/uniss_qwen0p5b_phase3_unist13_full.log" \
  "TRAIN_ITERS=${PHASE3_TRAIN_ITERS:-4341}" \
  "LR_WARMUP_ITERS=${PHASE3_LR_WARMUP_ITERS:-0}" \
  "SAVE_INTERVAL=${SAVE_INTERVAL:-100}" \
  "EVAL_INTERVAL=${EVAL_INTERVAL:-100}" \
  "EVAL_ITERS=${EVAL_ITERS:-10}" \
  "LOG_INTERVAL=${LOG_INTERVAL:-10}" \
  "MASTER_PORT=${PHASE3_MASTER_PORT:-29513}"
