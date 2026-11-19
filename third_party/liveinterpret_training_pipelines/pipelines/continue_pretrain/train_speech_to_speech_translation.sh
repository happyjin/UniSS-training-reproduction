#!/usr/bin/env bash
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# Continue-pretrain task: speech-to-speech translation.
# Trains thinker + bridge + talker end-to-end (Code2Wav stays frozen).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"

# ---------------------------------------------------------------------------
# 任务身份 + 任务级超参（pipelines 自治）
# ---------------------------------------------------------------------------

# ----- 任务身份 -----
: "${MODEL_TYPE:=qwen3_omnispeak_d}"
: "${MODEL_PATH:=${SOURCE_ROOT}/models/qwen3_omnispeak_d}"

# ----- 任务数据 -----
LOCAL_DATA="${REPO_ROOT}/datasets/jsonl/speech_to_speech_translation.jsonl"
if [[ -z "${CONTINUE_PRETRAIN_DATA:-}" ]]; then
    if [[ -s "${SOURCE_ROOT}/datasets/jsonl/speech_to_speech_translation.jsonl" ]]; then
        export CONTINUE_PRETRAIN_DATA="${SOURCE_ROOT}/datasets/jsonl/speech_to_speech_translation.jsonl"
    elif [[ -s "${LOCAL_DATA}" ]]; then
        export CONTINUE_PRETRAIN_DATA="${LOCAL_DATA}"
    fi
fi

# ----- 任务输出 -----
# OUTPUT_ROOT 由 common_args 统一提供（${SOURCE_ROOT}/output/qwen_omnispeak_output），无需在此设置
: "${WANDB_PROJECT:=qwen_omnispeak_continue_pretrain}"
: "${RUN_NAME:=${MODEL_TYPE}-continue_pretrain-speech_to_speech_translation-$(date +%Y%m%d_%H%M%S)}"

# ----- 任务超参 -----
: "${NUM_TRAIN_EPOCHS:=1}"
: "${LEARNING_RATE:=1e-5}"
: "${WARMUP_RATIO:=0.03}"
: "${MAX_LENGTH:=4096}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=16}"
: "${SAVE_STEPS:=500}"
: "${EVAL_STEPS:=500}"
: "${DATALOADER_NUM_WORKERS:=1}"
: "${DATALOADER_PERSISTENT_WORKERS:=false}"
: "${DATALOADER_PREFETCH_FACTOR:=}"
: "${SPLIT_DATASET_RATIO:=0.01}"

export MODEL_PATH MODEL_TYPE OUTPUT_ROOT WANDB_PROJECT RUN_NAME \
       NUM_TRAIN_EPOCHS LEARNING_RATE WARMUP_RATIO MAX_LENGTH \
       PER_DEVICE_TRAIN_BATCH_SIZE PER_DEVICE_EVAL_BATCH_SIZE \
       GRADIENT_ACCUMULATION_STEPS SAVE_STEPS EVAL_STEPS \
       DATALOADER_NUM_WORKERS DATALOADER_PERSISTENT_WORKERS DATALOADER_PREFETCH_FACTOR \
       SPLIT_DATASET_RATIO

# ---------------------------------------------------------------------------
# 本地 dev conda 兜底；anyscale 上 AUTO_ACTIVATE_CONDA=false 跳过整段。
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-base}"
AUTO_ACTIVATE_CONDA="${AUTO_ACTIVATE_CONDA:-true}"
if [[ "${AUTO_ACTIVATE_CONDA}" == "true" ]]; then
    CONDA_SH="${CONDA_SH:-}"
    if [[ -z "${CONDA_SH}" ]]; then
        for candidate in \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/miniconda/etc/profile.d/conda.sh" \
            "/root/miniconda3/etc/profile.d/conda.sh" \
            "/opt/miniconda3/etc/profile.d/conda.sh"; do
            if [[ -f "${candidate}" ]]; then
                CONDA_SH="${candidate}"
                break
            fi
        done
    fi
    if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
        source "${CONDA_SH}"
        conda activate "${CONDA_ENV}"
    else
        echo "[WARN] conda.sh not found; continuing with current shell environment." >&2
    fi
fi

# ---------------------------------------------------------------------------
# 上游 anyscale env 注入的 KEY
# ---------------------------------------------------------------------------
if [[ -d "${SOURCE_ROOT}/models/Qwen3-TTS-Tokenizer-12Hz" ]]; then
    export QWEN3_TTS_TOKENIZER="${QWEN3_TTS_TOKENIZER:-${SOURCE_ROOT}/models/Qwen3-TTS-Tokenizer-12Hz}"
else
    export QWEN3_TTS_TOKENIZER="${QWEN3_TTS_TOKENIZER:-Qwen3/Qwen3-TTS-Tokenizer-12Hz}"
fi
export DEEPSPEED="${DEEPSPEED:-zero2}"
export TUNER_TYPE="${TUNER_TYPE:-full}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

source "${REPO_ROOT}/pipelines/continue_pretrain/common_args.sh"
run_continue_pretrain \
    continue_pretrain_speech_to_speech_translation \
    speech_to_speech_translation \
    "$@"
