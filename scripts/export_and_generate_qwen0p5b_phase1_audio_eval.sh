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

export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/cache/pip}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"

PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase1_unist13_full}"
HF_REFERENCE="${HF_REFERENCE:-${REPO_ROOT}/checkpoints/qwen2_0p5b_uniss_vocab_hf}"
HF_OUTPUT="${HF_OUTPUT:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase1_unist13_full_hf}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_outputs/qwen0p5b_phase1_unist13_validation_audio}"

SPLIT="${SPLIT:-dev}"
LIMIT_RECORDS="${LIMIT_RECORDS:-3}"
MODES="${MODES:-tts}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1500}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
DTYPE="${DTYPE:-bfloat16}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EVAL_CUDA_VISIBLE_DEVICES}}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${REPO_ROOT}/logs" "${REPO_ROOT}/eval_outputs"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "${DRY_RUN}" == "0" ]]; then
  require_file "${PHASE1_CHECKPOINT}/latest_checkpointed_iteration.txt"
  require_dir "${HF_REFERENCE}"
  require_dir "${SPEECH_TOKENIZER}"
fi

run_cmd "${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
  --hf-model "${HF_REFERENCE}" \
  --megatron-path "${PHASE1_CHECKPOINT}" \
  --hf-output "${HF_OUTPUT}" \
  --no-progress

run_cmd env \
  HF_CHECKPOINT="${HF_OUTPUT}" \
  SPEECH_TOKENIZER="${SPEECH_TOKENIZER}" \
  SPLIT="${SPLIT}" \
  STEP_NAME="qwen0p5b_phase1_unist13_validation_audio" \
  LIMIT_RECORDS="${LIMIT_RECORDS}" \
  MODES="${MODES}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_P="${TOP_P}" \
  REPETITION_PENALTY="${REPETITION_PENALTY}" \
  DTYPE="${DTYPE}" \
  SAVE_SOURCE_AUDIO=1 \
  "${REPO_ROOT}/scripts/generate_unist_audio_eval.sh"
