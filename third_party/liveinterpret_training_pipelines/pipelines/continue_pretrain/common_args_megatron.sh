#!/usr/bin/env bash
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# Variant 3: `megatron sft` for qwen_omnispeak continue-pretrain.
# Parallel strategy: TP / PP / CP / EP + DistributedOptimizer (ZeRO-1).
#
# Each task script sources this file and calls
#   ``run_continue_pretrain_megatron <dataset_name> <task_name>``.
#
# ===========================================================================
# Megatron talker codec loss for qwen_omnispeak (variant 3)
# ---------------------------------------------------------------------------
# The non-invasive mcore-bridge adaptation maps the *thinker* transformer to a
# Megatron GPTModel (TP/EP sharded) and loads audio_tower / visual / bridge /
# TALKER as replicated submodules. mcore_talker_loss.py captures the thinker
# hidden states and computes the talker codec loss (main + residual CE), the
# SAME objective as the DeepSpeed path (verified by test_mcore_talker_loss.py).
#
#   * Text-supervised tasks (speech_to_text_translation, multi_speaker
#     segmentation): pure thinker text CE -- full TP/PP/CP/EP + SP supported.
#   * s2s / joint tasks: thinker text CE + talker codec loss. The codec loss
#     requires PP=1, CP=1, sequence_parallel=false, no packing/padding-free
#     (the thinker still uses full TP/EP/DP). The s2s/joint task scripts set
#     SEQUENCE_PARALLEL=false accordingly. Disable the codec loss with
#     QWEN_OMNISPEAK_MCORE_CODEC_LOSS=false (thinker text training only).
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Layout: <repo>/liveinterpret_training_pipelines/pipelines/continue_pretrain/
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"

# ---------------------------------------------------------------------------
# Data / tokenizer / cache paths (CONTINUE_PRETRAIN_DATA set by task script)
# ---------------------------------------------------------------------------
export CONTINUE_PRETRAIN_DATA="${CONTINUE_PRETRAIN_DATA:-${REPO_ROOT}/pipelines/continue_pretrain/data_format.jsonl}"
export QWEN3_TTS_TOKENIZER="${QWEN3_TTS_TOKENIZER:-Qwen3/Qwen3-TTS-Tokenizer-12Hz}"
export CONTINUE_PRETRAIN_CODEC_CACHE="${CONTINUE_PRETRAIN_CODEC_CACHE:-${SOURCE_ROOT}/.cache/continue_pretrain_codec}"
export CONTINUE_PRETRAIN_APPEND_TARGET_TEXT="${CONTINUE_PRETRAIN_APPEND_TARGET_TEXT:-true}"
export CONTINUE_PRETRAIN_STRICT="${CONTINUE_PRETRAIN_STRICT:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PLUGIN_PATH="${PLUGIN_PATH:-${REPO_ROOT}/custom/plugin.py}"
MODEL_PATH="${MODEL_PATH:-${SOURCE_ROOT}/models/qwen3_omnispeak_d}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_omnispeak_d}"
TEMPLATE="${TEMPLATE:-qwen_omnispeak_continue_pretrain}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_output}"

# ---------------------------------------------------------------------------
# Model parallelism (TP / PP / CP / EP). DP is derived from world_size.
# ---------------------------------------------------------------------------
TP="${TP:-2}"
PP="${PP:-1}"
CP="${CP:-1}"
EP="${EP:-1}"
ETP="${ETP:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"

# Talker codec loss controls (forwarded to the trainer patch). Auto-inactive
# for text-only rows; active for s2s rows that carry codec supervision.
export QWEN_OMNISPEAK_MCORE_CODEC_LOSS="${QWEN_OMNISPEAK_MCORE_CODEC_LOSS:-true}"
if [[ -n "${MCORE_HIDDEN_LAYER:-}" ]]; then
    export QWEN_OMNISPEAK_MCORE_HIDDEN_LAYER="${MCORE_HIDDEN_LAYER}"
fi
if [[ -n "${RESIDUAL_LOSS:-}" ]]; then
    export QWEN_OMNISPEAK_RESIDUAL_LOSS="${RESIDUAL_LOSS}"
fi
if [[ -n "${RESIDUAL_LOSS_WEIGHT:-}" ]]; then
    export QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT="${RESIDUAL_LOSS_WEIGHT}"
fi

# ---------------------------------------------------------------------------
# Tuner / freeze configuration
# TUNER_TYPE: ``full`` (default) or ``lora``.
# ---------------------------------------------------------------------------
TUNER_TYPE="${TUNER_TYPE:-full}"
FREEZE_LLM="${FREEZE_LLM:-false}"
FREEZE_VIT="${FREEZE_VIT:-true}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-false}"

LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# ---------------------------------------------------------------------------
# Trainer hyper-params
# ---------------------------------------------------------------------------
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.03}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.01}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
PACKING="${PACKING:-false}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_continue_pretrain_megatron}"

# ---------------------------------------------------------------------------
# Distributed launcher env (read by megatron / torchrun)
# ---------------------------------------------------------------------------
_visible_gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        python - <<'PY'
import os
visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
devices = [item.strip() for item in visible.split(',') if item.strip() and item.strip() != '-1']
print(len(devices))
PY
    else
        python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
    fi
}

if [[ -z "${NPROC_PER_NODE:-}" && "${USE_RAY:-false}" != "true" ]]; then
    GPU_COUNT="$(_visible_gpu_count)"
    if [[ "${GPU_COUNT}" -gt 1 ]]; then
        export NPROC_PER_NODE="${GPU_COUNT}"
    fi
fi
# CRITICAL: only export torchrun-launch envs when NOT using Ray.
if [[ "${USE_RAY:-false}" != "true" && "${USE_RAY:-false}" != "1" ]]; then
    export NNODES="${NNODES:-1}"
    export NODE_RANK="${NODE_RANK:-0}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
fi

_continue_pretrain_megatron_tuner_args() {
    local args=()
    if [[ "${TUNER_TYPE}" == "lora" ]]; then
        args+=(--tuner_type lora)
        args+=(--target_modules "${LORA_TARGET_MODULES}")
        args+=(--lora_rank "${LORA_RANK}")
        args+=(--lora_alpha "${LORA_ALPHA}")
    else
        args+=(--tuner_type full)
    fi
    args+=(--freeze_llm "${FREEZE_LLM}")
    args+=(--freeze_vit "${FREEZE_VIT}")
    args+=(--freeze_aligner "${FREEZE_ALIGNER}")
    printf '%s\n' "${args[@]}"
}

run_continue_pretrain_megatron() {
    local dataset_name="$1"
    local task_name="$2"
    shift 2
    local output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/${task_name}_${TUNER_TYPE}}"

    mkdir -p "${output_dir}" "${CONTINUE_PRETRAIN_CODEC_CACHE}"

    local tuner_args=()
    while IFS= read -r line; do
        tuner_args+=("${line}")
    done < <(_continue_pretrain_megatron_tuner_args)

    local moe_args=()
    if [[ "${EP}" -gt 1 ]]; then
        moe_args+=(--expert_model_parallel_size "${EP}")
        moe_args+=(--expert_tensor_parallel_size "${ETP}")
        moe_args+=(--moe_grouped_gemm true)
        moe_args+=(--moe_permute_fusion true)
        moe_args+=(--moe_shared_expert_overlap true)
        moe_args+=(--moe_aux_loss_coeff 1e-3)
    fi

    megatron sft \
        --external_plugins "${PLUGIN_PATH}" \
        --model "${MODEL_PATH}" \
        --model_type "${MODEL_TYPE}" \
        --template "${TEMPLATE}" \
        --trust_remote_code true \
        --dataset "${dataset_name}" \
        --load_from_cache_file false \
        --split_dataset_ratio "${SPLIT_DATASET_RATIO}" \
        "${tuner_args[@]}" \
        --tensor_model_parallel_size "${TP}" \
        --pipeline_model_parallel_size "${PP}" \
        --context_parallel_size "${CP}" \
        --sequence_parallel "${SEQUENCE_PARALLEL}" \
        "${moe_args[@]}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --global_batch_size "${GLOBAL_BATCH_SIZE}" \
        --recompute_granularity "${RECOMPUTE_GRANULARITY}" \
        --recompute_method "${RECOMPUTE_METHOD}" \
        --recompute_num_layers "${RECOMPUTE_NUM_LAYERS}" \
        --finetune true \
        --cross_entropy_loss_fusion true \
        --lr "${LR}" \
        --min_lr "${MIN_LR}" \
        --lr_warmup_fraction "${LR_WARMUP_FRACTION}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --max_length "${MAX_LENGTH}" \
        --packing "${PACKING}" \
        --torch_dtype "${TORCH_DTYPE}" \
        --attention_backend "${ATTENTION_BACKEND}" \
        --save_safetensors true \
        --no_save_optim true \
        --no_save_rng true \
        --eval_steps "${EVAL_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --logging_steps "${LOGGING_STEPS}" \
        --report_to "${REPORT_TO}" \
        --run_name "${RUN_NAME}" \
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
        --dataset_num_proc "${DATASET_NUM_PROC}" \
        --output_dir "${output_dir}" \
        "$@"
}
