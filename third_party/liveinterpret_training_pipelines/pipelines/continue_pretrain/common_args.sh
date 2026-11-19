#!/usr/bin/env bash
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# Common arguments shared by every continue-pretrain training pipeline for
# qwen_omnispeak. Each task script sources this file and then calls
# ``run_continue_pretrain <dataset_name> <task_name>``.
#
# Supports:
#   * full-parameter and LoRA training (`TUNER_TYPE=full|lora`)
#   * distributed training via the standard ms-swift / torchrun env vars
#     (NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
#   * deepspeed zero2/zero3 via DEEPSPEED env var
#   * freezing the (optional) Code2Wav vocoder by regex (FREEZE_PARAMETERS)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Layout: <repo>/liveinterpret_training_pipelines/pipelines/continue_pretrain/
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"

# ---------------------------------------------------------------------------
# Data / tokenizer / cache paths
# ---------------------------------------------------------------------------
export CONTINUE_PRETRAIN_DATA="${CONTINUE_PRETRAIN_DATA:-${REPO_ROOT}/pipelines/continue_pretrain/data_format.jsonl}"
export QWEN3_TTS_TOKENIZER="${QWEN3_TTS_TOKENIZER:-Qwen3/Qwen3-TTS-Tokenizer-12Hz}"
export CONTINUE_PRETRAIN_CODEC_CACHE="${CONTINUE_PRETRAIN_CODEC_CACHE:-${SOURCE_ROOT}/.cache/continue_pretrain_codec}"
export CONTINUE_PRETRAIN_APPEND_TARGET_TEXT="${CONTINUE_PRETRAIN_APPEND_TARGET_TEXT:-true}"
export CONTINUE_PRETRAIN_STRICT="${CONTINUE_PRETRAIN_STRICT:-false}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${SOURCE_ROOT}/.cache/hf_modules_qwen_omnispeak_continue_pretrain}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Plugin / model
# ---------------------------------------------------------------------------
PLUGIN_PATH="${PLUGIN_PATH:-${REPO_ROOT}/custom/plugin.py}"
# 注：MODEL_PATH / MODEL_TYPE / OUTPUT_ROOT 已下放到各 train_*.sh，本文件仅在
# 调用方未提前设置时给出兜底值，便于单独 source common_args.sh 时不缺变量。
: "${MODEL_PATH:=${SOURCE_ROOT}/models/qwen3_omnispeak_d}"
: "${MODEL_TYPE:=qwen3_omnispeak_d}"
TEMPLATE="${TEMPLATE:-qwen_omnispeak_continue_pretrain}"

# ---------------------------------------------------------------------------
# Tuner / freeze configuration
# ---------------------------------------------------------------------------
# TUNER_TYPE: ``full`` (default) or ``lora``.
TUNER_TYPE="${TUNER_TYPE:-full}"

# Code2Wav freeze regex. The user requirement is: the only parameters
# guaranteed to be frozen are the talker's Code2Wav (vocoder). The current
# qwen_omnispeak package does not include Code2Wav, so this regex is a safe
# best-effort match. Override with FREEZE_PARAMETERS to match your own naming.
FREEZE_PARAMETERS="${FREEZE_PARAMETERS:-^(model\\.talker\\.code2wav|model\\.code2wav|model\\.talker\\.codec_decoder|model\\.codec_decoder|model\\.talker\\.vocoder|model\\.vocoder)(\\.|$)}"

# Optional comma-separated extra trainable parameters.
TRAINABLE_PARAMETERS="${TRAINABLE_PARAMETERS:-}"

# LoRA-specific arguments (only used when TUNER_TYPE=lora).
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
# When TUNER_TYPE=lora, ms-swift's multimodal target_regex respects these
# scope flags. By default we let LoRA touch the language model and aligner
# (thinker text + bridge), and allow generator (talker) too for s2s rows.
FREEZE_LLM="${FREEZE_LLM:-false}"
FREEZE_VIT="${FREEZE_VIT:-true}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-false}"

# ---------------------------------------------------------------------------
# Trainer hyper-params
# ---------------------------------------------------------------------------
DEEPSPEED="${DEEPSPEED:-zero2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
# OUTPUT_ROOT: 所有训练任务统一默认输出根，便于 anyscale 按统一路径归档 S3。
: "${OUTPUT_ROOT:=${SOURCE_ROOT}/output/qwen_omnispeak_output}"
export OUTPUT_ROOT

REPORT_TO="${REPORT_TO:-wandb}"
: "${WANDB_PROJECT:=qwen_omnispeak_continue_pretrain}"
: "${WANDB_DIR:=${OUTPUT_ROOT}/wandb}"
: "${WANDB_WATCH:=false}"
: "${WANDB_LOG_MODEL:=false}"
export WANDB_PROJECT WANDB_DIR WANDB_WATCH WANDB_LOG_MODEL

# Attention impl probe with cascade fallback FA3 → FA2 → sdpa.
if [[ "${ATTN_IMPL}" == "flash_attention_3" || "${ATTN_IMPL}" == "flash_attn_3" ]]; then
    if ! python - <<'PY' >/dev/null 2>&1
import importlib.util as u
import sys
sys.exit(0 if (u.find_spec('flash_attn_3') and u.find_spec('flash_attn_interface')) else 1)
PY
    then
        echo "[WARN] flash_attn_3 / flash_attn_interface not importable; trying flash_attention_2." >&2
        ATTN_IMPL="flash_attention_2"
    fi
fi
if [[ "${ATTN_IMPL}" == "flash_attention_2" || "${ATTN_IMPL}" == "flash_attn" ]]; then
    if ! python - <<'PY' >/dev/null 2>&1
import importlib.util as u
import sys
sys.exit(0 if u.find_spec('flash_attn') else 1)
PY
    then
        echo "[WARN] flash_attn (FA2) is unavailable; falling back to sdpa." >&2
        ATTN_IMPL="sdpa"
    fi
fi

# 任务级超参的兜底默认值。任务脚本都会先 export 这些变量，再 source
# common_args.sh；这里保留一套兜底，仅当某场景单独 source 此文件、且任务脚本
# 未注入相关默认值时使用。任务级真正的默认值见 pipelines/<dir>/<task>.sh。
: "${NUM_TRAIN_EPOCHS:=1}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=16}"
: "${LEARNING_RATE:=1e-5}"
: "${WARMUP_RATIO:=0.03}"
: "${MAX_LENGTH:=4096}"
: "${SAVE_STEPS:=500}"
: "${EVAL_STEPS:=500}"
: "${LOGGING_STEPS:=10}"
: "${SAVE_TOTAL_LIMIT:=3}"
: "${DATASET_NUM_PROC:=1}"
: "${DATALOADER_NUM_WORKERS:=1}"
: "${DATALOADER_PERSISTENT_WORKERS:=false}"
: "${DATALOADER_PREFETCH_FACTOR:=}"
: "${SPLIT_DATASET_RATIO:=0.01}"
: "${GRADIENT_CHECKPOINTING:=true}"

# ---------------------------------------------------------------------------
# Distributed launcher env (read by ms-swift / torchrun)
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

if [[ -z "${NPROC_PER_NODE:-}" && -n "${DEEPSPEED}" && "${DEEPSPEED}" != "none" && "${USE_RAY:-false}" != "true" ]]; then
    GPU_COUNT="$(_visible_gpu_count)"
    if [[ "${GPU_COUNT}" -gt 1 ]]; then
        export NPROC_PER_NODE="${GPU_COUNT}"
    fi
fi
# CRITICAL: only export torchrun-launch envs (NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT)
# when we are NOT using Ray. With USE_RAY=true, these env vars must remain unset, because
# swift/cli/main.py:use_torchrun() returns True the moment NNODES (or NPROC_PER_NODE) is
# set, which makes `swift sft` re-spawn itself under `python -m torch.distributed.run`.
# That torchrun wrapper boots a LocalElasticAgent which in turn writes
# TORCHELASTIC_USE_AGENT_STORE=True (and friends LOCAL_WORLD_SIZE=1, GROUP_RANK=0,
# TORCHELASTIC_RUN_ID, ...) into os.environ. swift/ray_utils/base.py:_create_workers
# then does `env_vars = os.environ.copy()` and ships those torchelastic markers via
# RuntimeEnv to all SwiftSft ray actors. Each actor then calls
# torch.distributed.init_process_group → torch/distributed/rendezvous.py:_create_c10d_store,
# which sees TORCHELASTIC_USE_AGENT_STORE=True and forces is_master=False on EVERY rank,
# including rank 0. All ranks become TCPStore CLIENTS connecting to a master that
# nobody is hosting → stack stuck forever.
if [[ "${USE_RAY:-false}" != "true" && "${USE_RAY:-false}" != "1" ]]; then
    export NNODES="${NNODES:-1}"
    export NODE_RANK="${NODE_RANK:-0}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
fi

# ---------------------------------------------------------------------------
# Ray (optional). Enable by setting USE_RAY=true. ms-swift's `swift sft`
# natively supports Ray via `--use_ray / --ray_exp_name / --device_groups`.
# RAY_DEVICE_GROUPS must be a JSON string when set, e.g.
#   '{"train":{"device":"GPU","workers":4,"ranks":[0,1,2,3]}}'
# Refer to https://swift.readthedocs.io/zh-cn/latest/Instruction/Ray.html
# ---------------------------------------------------------------------------
USE_RAY="${USE_RAY:-false}"
RAY_EXP_NAME="${RAY_EXP_NAME:-}"
RAY_DEVICE_GROUPS="${RAY_DEVICE_GROUPS:-}"

# ---------------------------------------------------------------------------
# Helper: build extra tuner CLI args based on TUNER_TYPE.
# ---------------------------------------------------------------------------
_continue_pretrain_tuner_args() {
    local args=()
    if [[ "${TUNER_TYPE}" == "lora" ]]; then
        args+=(--tuner_type lora)
        args+=(--target_modules "${LORA_TARGET_MODULES}")
        args+=(--lora_rank "${LORA_RANK}")
        args+=(--lora_alpha "${LORA_ALPHA}")
        args+=(--lora_dropout "${LORA_DROPOUT}")
        args+=(--freeze_llm "${FREEZE_LLM}")
        args+=(--freeze_vit "${FREEZE_VIT}")
        args+=(--freeze_aligner "${FREEZE_ALIGNER}")
        # Always freeze the optional Code2Wav module on top of LoRA's frozen base.
        args+=(--freeze_parameters "${FREEZE_PARAMETERS}")
    else
        args+=(--tuner_type full)
        # Full-parameter training: only Code2Wav (if present) is frozen.
        args+=(--freeze_parameters "${FREEZE_PARAMETERS}")
    fi
    if [[ -n "${TRAINABLE_PARAMETERS}" ]]; then
        # ms-swift accepts --trainable_parameters as space-separated list; we
        # also accept a comma-separated env value for convenience.
        local p
        IFS=',' read -ra _PARTS <<< "${TRAINABLE_PARAMETERS}"
        for p in "${_PARTS[@]}"; do
            [[ -z "${p}" ]] && continue
            args+=(--trainable_parameters "${p}")
        done
    fi
    printf '%s\n' "${args[@]}"
}

run_continue_pretrain() {
    local dataset_name="$1"
    local task_name="$2"
    shift 2
    local output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/${task_name}_${TUNER_TYPE}}"

    mkdir -p "${output_dir}" "${CONTINUE_PRETRAIN_CODEC_CACHE}"

    local tuner_args=()
    while IFS= read -r line; do
        tuner_args+=("${line}")
    done < <(_continue_pretrain_tuner_args)

    local ray_args=()
    if [[ "${USE_RAY}" == "true" || "${USE_RAY}" == "1" ]]; then
        ray_args+=(--use_ray true)
        if [[ -n "${RAY_EXP_NAME}" ]]; then
            ray_args+=(--ray_exp_name "${RAY_EXP_NAME}")
        fi
        if [[ -n "${RAY_DEVICE_GROUPS}" ]]; then
            ray_args+=(--device_groups "${RAY_DEVICE_GROUPS}")
        fi
    fi

    # Build dataloader args. persistent_workers / prefetch_factor only valid
    # when num_workers > 0.
    local dataloader_args=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS}")
    if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
        dataloader_args+=(--dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}")
        if [[ -n "${DATALOADER_PREFETCH_FACTOR}" ]]; then
            dataloader_args+=(--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}")
        fi
    fi

    swift sft \
        --external_plugins "${PLUGIN_PATH}" \
        --model "${MODEL_PATH}" \
        --model_type "${MODEL_TYPE}" \
        --template "${TEMPLATE}" \
        --trust_remote_code true \
        --dataset "${dataset_name}" \
        --load_from_cache_file false \
        --remove_unused_columns false \
        "${tuner_args[@]}" \
        "${ray_args[@]}" \
        --torch_dtype "${TORCH_DTYPE}" \
        --attn_impl "${ATTN_IMPL}" \
        --deepspeed "${DEEPSPEED}" \
        --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --learning_rate "${LEARNING_RATE}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --max_length "${MAX_LENGTH}" \
        --split_dataset_ratio "${SPLIT_DATASET_RATIO}" \
        --eval_steps "${EVAL_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --logging_steps "${LOGGING_STEPS}" \
        --report_to "${REPORT_TO}" \
        --run_name "${RUN_NAME}" \
        "${dataloader_args[@]}" \
        --dataset_num_proc "${DATASET_NUM_PROC}" \
        --output_dir "${output_dir}" \
        "$@"
}
