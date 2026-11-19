#!/usr/bin/env bash
# Variant 3: megatron sft (TP/PP/CP/EP + ZeRO-1) -- large-scale speech-to-text translation.
# Text-supervised (thinker text CE): fully covered by the Megatron path.
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
LOCAL_DATA="${REPO_ROOT}/datasets/jsonl/speech_to_text_translation.jsonl"
if [[ -z "${CONTINUE_PRETRAIN_DATA:-}" ]]; then
    if [[ -s "${SOURCE_ROOT}/datasets/jsonl/speech_to_text_translation.jsonl" ]]; then
        export CONTINUE_PRETRAIN_DATA="${SOURCE_ROOT}/datasets/jsonl/speech_to_text_translation.jsonl"
    elif [[ -s "${LOCAL_DATA}" ]]; then
        export CONTINUE_PRETRAIN_DATA="${LOCAL_DATA}"
    fi
fi

# ----- 任务输出 -----
# OUTPUT_ROOT 由 common_args 统一提供（${SOURCE_ROOT}/output/qwen_omnispeak_output），无需在此设置
: "${WANDB_PROJECT:=qwen_omnispeak_continue_pretrain_megatron}"
: "${RUN_NAME:=${MODEL_TYPE}-continue_pretrain-speech_to_text_translation-megatron-$(date +%Y%m%d_%H%M%S)}"

# ----- 任务超参 -----
export TUNER_TYPE="${TUNER_TYPE:-full}"
export TP="${TP:-2}"
export PP="${PP:-1}"
export CP="${CP:-1}"
export EP="${EP:-1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export LR="${LR:-1e-5}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

source "${REPO_ROOT}/pipelines/continue_pretrain/common_args_megatron.sh"
run_continue_pretrain_megatron continue_pretrain_speech_to_text_translation speech_to_text_translation "$@"
