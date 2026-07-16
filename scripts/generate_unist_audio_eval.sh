#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
ENV_ROOT="${ENV_ROOT:-${USER_ROOT}/conda_envs/uniss-train}"

export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/cache/pip}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${REPO_ROOT}/eval_outputs"

HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT to a Hugging Face checkpoint directory.}"
SPLIT="${SPLIT:-dev}"
STEP_NAME="${STEP_NAME:-manual}"
LIMIT_RECORDS="${LIMIT_RECORDS:-8}"
MODES="${MODES:-quality performance}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1500}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
DTYPE="${DTYPE:-bfloat16}"

case "${SPLIT}" in
  dev)
    INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/data/raw/UniST/dev-00000.parquet}"
    ;;
  test)
    INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/data/raw/UniST/test-00000.parquet}"
    ;;
  *)
    INPUT_PATH="${INPUT_PATH:-${SPLIT}}"
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_outputs/${STEP_NAME}_${SPLIT}}"

CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
"${ENV_ROOT}/bin/python" "${REPO_ROOT}/training/generate_unist_eval_audio.py" \
  --input "${INPUT_PATH}" \
  --model "${HF_CHECKPOINT}" \
  --speech-tokenizer "${SPEECH_TOKENIZER}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode ${MODES} \
  --limit-records "${LIMIT_RECORDS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --repetition-penalty "${REPETITION_PENALTY}" \
  --dtype "${DTYPE}" \
  --device cuda:0 \
  --local-files-only \
  --save-reference-audio \
  --overwrite
