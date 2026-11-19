#!/usr/bin/env bash
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# Stage-3 SFT joint task: joint-stage1.
# Set USE_RAY=true for multi-node Ray; the same entrypoint also supports local
# non-Ray runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"

: "${MODEL_TYPE:=qwen3_omnispeak_d}"
: "${MODEL_PATH:=${SOURCE_ROOT}/models/qwen3_omnispeak_d}"

# ----- 任务数据 -----
LOCAL_DATA="${REPO_ROOT}/datasets/jsonl/sft_joint_stage1.jsonl"
if [[ -z "${JOINT_STAGE1_DATA:-}" && -z "${SFT_JOINT_STAGE1_DATA:-}" ]]; then
    if [[ -s "${SOURCE_ROOT}/datasets/jsonl/sft_joint_stage1.jsonl" ]]; then
        export JOINT_STAGE1_DATA="${SOURCE_ROOT}/datasets/jsonl/sft_joint_stage1.jsonl"
    elif [[ -s "${LOCAL_DATA}" ]]; then
        export JOINT_STAGE1_DATA="${LOCAL_DATA}"
    fi
elif [[ -z "${JOINT_STAGE1_DATA:-}" && -n "${SFT_JOINT_STAGE1_DATA:-}" ]]; then
    export JOINT_STAGE1_DATA="${SFT_JOINT_STAGE1_DATA}"
fi
export SFT_JOINT_STAGE1_DATA="${SFT_JOINT_STAGE1_DATA:-${JOINT_STAGE1_DATA:-}}"

# OUTPUT_ROOT 由 common_args 统一提供（${SOURCE_ROOT}/output/qwen_omnispeak_output），无需在此设置
: "${WANDB_PROJECT:=qwen_omnispeak_sft_joint_stage1}"
: "${RUN_NAME:=${MODEL_TYPE}-joint-stage1-$(date +%Y%m%d_%H%M%S)}"
: "${RAY_EXP_NAME:=sft_joint_stage1}"

: "${TUNER_TYPE:=full}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${LEARNING_RATE:=1e-5}"
: "${MAX_LENGTH:=4096}"
: "${SAVE_STEPS:=500}"
: "${EVAL_STEPS:=500}"
: "${GRADIENT_ACCUMULATION_STEPS:=16}"
: "${JOINT_STAGE1_STREAMING:=true}"
: "${JOINT_STAGE1_EXTRA_LAG_MS:=0}"

export MODEL_PATH MODEL_TYPE OUTPUT_ROOT WANDB_PROJECT RUN_NAME RAY_EXP_NAME \
       TUNER_TYPE NUM_TRAIN_EPOCHS LEARNING_RATE MAX_LENGTH SAVE_STEPS \
       EVAL_STEPS GRADIENT_ACCUMULATION_STEPS JOINT_STAGE1_DATA \
       SFT_JOINT_STAGE1_DATA JOINT_STAGE1_STREAMING JOINT_STAGE1_EXTRA_LAG_MS

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
export DEEPSPEED="${DEEPSPEED:-zero2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

source "${REPO_ROOT}/pipelines/sft/common_args.sh"
run_joint_stage1 sft_joint_stage1 joint-stage1 "$@"
