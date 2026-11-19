#!/usr/bin/env bash
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# Variant 3: `megatron sft` (TP/EP/DP + ZeRO-1) for Stage-3 SFT
# simultaneous (read-write) policy learning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"

: "${MODEL_TYPE:=qwen3_omnispeak_d}"
: "${MODEL_PATH:=${SOURCE_ROOT}/models/qwen3_omnispeak_d}"

# ----- 任务数据 -----
LOCAL_DATA="${REPO_ROOT}/datasets/jsonl/simultaneous_policy_learning.jsonl"
if [[ -z "${SIMULTANEOUS_POLICY_DATA:-}" ]]; then
    if [[ -s "${SOURCE_ROOT}/datasets/jsonl/simultaneous_policy_learning.jsonl" ]]; then
        export SIMULTANEOUS_POLICY_DATA="${SOURCE_ROOT}/datasets/jsonl/simultaneous_policy_learning.jsonl"
    elif [[ -s "${LOCAL_DATA}" ]]; then
        export SIMULTANEOUS_POLICY_DATA="${LOCAL_DATA}"
    fi
fi

# OUTPUT_ROOT 由 common_args 统一提供（${SOURCE_ROOT}/output/qwen_omnispeak_output），无需在此设置
: "${WANDB_PROJECT:=qwen_omnispeak_simultaneous_policy_megatron}"
: "${RUN_NAME:=${MODEL_TYPE}-simultaneous_policy_learning-megatron-$(date +%Y%m%d_%H%M%S)}"

: "${TUNER_TYPE:=full}"
: "${TP:=2}"
: "${PP:=1}"
: "${CP:=1}"
: "${EP:=1}"
: "${SEQUENCE_PARALLEL:=false}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${LR:=1e-5}"
: "${MAX_LENGTH:=4096}"
: "${SAVE_STEPS:=500}"
: "${EVAL_STEPS:=500}"
: "${GLOBAL_BATCH_SIZE:=16}"
: "${SIMULTANEOUS_POLICY_STREAMING:=true}"
: "${SIMULTANEOUS_POLICY_EXTRA_LAG_MS:=0}"

export MODEL_PATH MODEL_TYPE OUTPUT_ROOT WANDB_PROJECT RUN_NAME TUNER_TYPE \
       TP PP CP EP SEQUENCE_PARALLEL NUM_TRAIN_EPOCHS LR MAX_LENGTH \
       SAVE_STEPS EVAL_STEPS GLOBAL_BATCH_SIZE SIMULTANEOUS_POLICY_STREAMING \
       SIMULTANEOUS_POLICY_EXTRA_LAG_MS

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

if [[ -z "${QWEN3_TTS_TOKENIZER:-}" ]]; then
    if [[ -n "${TOKENIZER_DIR:-}" ]]; then
        export QWEN3_TTS_TOKENIZER="${TOKENIZER_DIR}"
    elif [[ -d "${SOURCE_ROOT}/models/Qwen3-TTS-Tokenizer-12Hz" ]]; then
        export QWEN3_TTS_TOKENIZER="${SOURCE_ROOT}/models/Qwen3-TTS-Tokenizer-12Hz"
    else
        export QWEN3_TTS_TOKENIZER="Qwen3/Qwen3-TTS-Tokenizer-12Hz"
    fi
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

source "${REPO_ROOT}/pipelines/sft/common_args_megatron.sh"
run_simultaneous_policy_megatron simultaneous_policy_learning simultaneous_policy_learning "$@"
