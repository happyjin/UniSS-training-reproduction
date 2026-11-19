#!/usr/bin/env bash
set -euo pipefail

# Public reproduction recipe for the batch-4.6 training semantics. The caller
# supplies checkpoint and data roots; no internal paths are embedded here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the warm-start checkpoint directory}"
: "${WDS_MANIFEST_DIR:?Set WDS_MANIFEST_DIR to the committed JSONL manifest directory}"
export MODEL_PATH WDS_MANIFEST_DIR

export MODEL_TYPE="${MODEL_TYPE:-qwen2_5_3b_omnispeak_a}"
export BRIDGE_ALIGNMENT_DATA="${BRIDGE_ALIGNMENT_DATA:-${WDS_MANIFEST_DIR}/manifest-*.jsonl}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${REPO_ROOT}/.cache/modelscope_reply_add_s2s_wds}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MODELSCOPE_CACHE}/datasets}"

export TALKER_METHOD="reply_add"
export TALKER_TRAINING_SCOPE="whole"
export UNFREEZE_TALKER_LAYERS="0"
export UNFREEZE_THINKER="1"
export QWEN_OMNISPEAK_USE_ICL="false"
export BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT="true"
export WDS_AUDIO_MODE="required"
export WDS_AUDIO_STRICT="true"

export CODEC_EOS_LOSS_WEIGHT="10"
export CODEC_HISTORY_CORRUPTION_PROB="0.05"
export RESIDUAL_LOSS="true"
export RESIDUAL_LOSS_WEIGHT="0.3"

export NUM_TRAIN_EPOCHS="5"
export LEARNING_RATE="1e-5"
export WARMUP_RATIO="0.03"
export MAX_LENGTH="4096"
export PER_DEVICE_TRAIN_BATCH_SIZE="64"
export PER_DEVICE_EVAL_BATCH_SIZE="64"
export GRADIENT_ACCUMULATION_STEPS="1"
export SPLIT_DATASET_RATIO="0.01"
export GROUP_BY_LENGTH="false"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
export DATALOADER_PERSISTENT_WORKERS="true"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-16}"
export GRADIENT_CHECKPOINTING="true"
export DEEPSPEED="zero2"

export EVAL_STRATEGY="steps"
export EVAL_STEPS="1000"
export SAVE_STEPS="1000"
export SAVE_TOTAL_LIMIT="3"
export LOGGING_STEPS="10"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29646}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_s2s_translation}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-wds-eos-history}"
export RUN_NAME="${RUN_NAME:-reply_add-thinker-talker-wds-eos10-hist005-bs64-s2s}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/reply_add-wds-eos10-hist005-bs64-s2s}"

# The new data/loss distribution needs fresh optimizer and scheduler state.
unset RESUME_FROM

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" "${REPO_ROOT}/test_script/batch4_6/preflight.py"
if [[ "${PREFLIGHT_ONLY:-false}" == "true" ]]; then
    echo "[wds-eos-history] PREFLIGHT_ONLY=true; training launch skipped."
    exit 0
fi

exec bash "${SCRIPT_DIR}/train_speech_to_speech_translation_upstream_wds.sh" "$@"
