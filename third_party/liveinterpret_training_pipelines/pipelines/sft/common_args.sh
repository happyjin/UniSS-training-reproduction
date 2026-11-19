#!/usr/bin/env bash
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# Common arguments shared by the Stage-3 SFT *streaming speech-to-speech
# translation* pipelines for qwen_omnispeak (DeepSpeed / `swift sft`).
#
# Stage-3 sibling tasks share this file; a task script sources it and then calls one
# of the helpers:
#   * ``run_simultaneous_policy <dataset_name> <task_name>``  — simultaneous
#     (read-write) policy learning.
#   * ``run_normal_translation <dataset_name> <task_name>``   — normal_translation.
#   * ``run_bilingual_translation <dataset_name> <task_name>`` — bilingual_translation.
#   * ``run_multi_speaker_translation <dataset_name> <task_name>`` — multi_speaker_translation.
#   * ``run_joint_stage1 <dataset_name> <task_name>`` — joint normal / bilingual / multi-speaker translation.
#
# These tasks are mechanically identical: codec-only supervision; a real-timeline
# target-codec stream with silence-filled listen gaps; a monotonic
# ``codec_to_sem_visible`` read-write attention schedule derived from each
# sample's chunk-level ``alignment`` timestamps, supervised by the existing codec
# CE loss (no new loss / head is added). They also share the same timeline sample
# validation rules: source/target chunks must be strictly non-overlapping, and
# the target timeline duration must satisfy the common tail bound. They differ
# only in the env-var prefix (``SIMULTANEOUS_POLICY_*`` /
# ``NORMAL_TRANSLATION_*`` / ``BILINGUAL_TRANSLATION_*`` /
# ``MULTI_SPEAKER_TRANSLATION_*``) plus task-specific TEMPLATE / OUTPUT_ROOT /
# cache defaults, which each ``run_*`` helper sets for itself so the tasks never
# cross-talk.
#
# Supports:
#   * full-parameter and LoRA training (`TUNER_TYPE=full|lora`)
#   * distributed training via the standard ms-swift / torchrun env vars
#     (NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
#   * deepspeed zero2/zero3 via DEEPSPEED env var
#   * freezing the (optional) Code2Wav vocoder by regex (FREEZE_PARAMETERS)
#   * *_STREAMING / *_EXTRA_LAG_MS switches: toggle the streaming mask (ablation
#     -> offline) and a wait-k style global lag offset.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Layout: <repo>/liveinterpret_training_pipelines/pipelines/sft/
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"

# OUTPUT_ROOT: 所有训练任务统一默认输出根，便于 anyscale 按统一路径归档 S3。
# 各 run_* helper 调用时传入的 "${OUTPUT_ROOT:-...}" 任务级兜底仅在此处未设时才生效。
: "${OUTPUT_ROOT:=${SOURCE_ROOT}/output/qwen_omnispeak_output}"
export OUTPUT_ROOT

# ---------------------------------------------------------------------------
# Tokenizer (shared by both tasks). Prefer the local NVMe/repo copy when it
# exists; otherwise fall back to HuggingFace Hub for local dev without synced
# models.
# ---------------------------------------------------------------------------
if [[ -z "${QWEN3_TTS_TOKENIZER:-}" ]]; then
    if [[ -n "${TOKENIZER_DIR:-}" ]]; then
        export QWEN3_TTS_TOKENIZER="${TOKENIZER_DIR}"
    elif [[ -d "${SOURCE_ROOT}/models/Qwen3-TTS-Tokenizer-12Hz" ]]; then
        export QWEN3_TTS_TOKENIZER="${SOURCE_ROOT}/models/Qwen3-TTS-Tokenizer-12Hz"
    else
        export QWEN3_TTS_TOKENIZER="Qwen3/Qwen3-TTS-Tokenizer-12Hz"
    fi
fi

# ---------------------------------------------------------------------------
# Plugin / model
# ---------------------------------------------------------------------------
PLUGIN_PATH="${PLUGIN_PATH:-${REPO_ROOT}/custom/plugin.py}"
MODEL_PATH="${MODEL_PATH:-${SOURCE_ROOT}/models/qwen3_omnispeak_d}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_omnispeak_d}"

# ---------------------------------------------------------------------------
# Tuner / freeze configuration
# ---------------------------------------------------------------------------
# TUNER_TYPE: ``full`` (default) or ``lora``.
TUNER_TYPE="${TUNER_TYPE:-full}"

# Code2Wav freeze regex. As with bridge/continue_pretrain, the only parameters
# guaranteed to be frozen are the talker's Code2Wav (vocoder); thinker + bridge
# + talker backbone stay trainable. Override FREEZE_PARAMETERS to match your
# own naming.
FREEZE_PARAMETERS="${FREEZE_PARAMETERS:-^(model\\.talker\\.code2wav|model\\.code2wav|model\\.talker\\.codec_decoder|model\\.codec_decoder|model\\.talker\\.vocoder|model\\.vocoder)(\\.|$)}"

# Optional comma-separated extra trainable parameters.
TRAINABLE_PARAMETERS="${TRAINABLE_PARAMETERS:-}"

# LoRA-specific arguments (only used when TUNER_TYPE=lora).
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
FREEZE_LLM="${FREEZE_LLM:-false}"
FREEZE_VIT="${FREEZE_VIT:-true}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-false}"

# ---------------------------------------------------------------------------
# Trainer hyper-params
# ---------------------------------------------------------------------------
DEEPSPEED="${DEEPSPEED:-zero2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"

REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_WATCH="${WANDB_WATCH:-false}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"

# Attention impl probe with cascade fallback FA3 → FA2 → sdpa.
# `find_spec` is used so the probe does NOT trigger CUDA init at script-time.
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

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.01}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

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

if [[ "${USE_RAY:-false}" != "true" && "${USE_RAY:-false}" != "1" ]]; then
    if [[ -z "${NPROC_PER_NODE:-}" && -n "${DEEPSPEED}" && "${DEEPSPEED}" != "none" ]]; then
        GPU_COUNT="$(_visible_gpu_count)"
        if [[ "${GPU_COUNT}" -gt 1 ]]; then
            export NPROC_PER_NODE="${GPU_COUNT}"
        fi
    fi
    export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
    export NNODES="${NNODES:-1}"
    export NODE_RANK="${NODE_RANK:-0}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
fi

# ---------------------------------------------------------------------------
# Ray (optional). Enable by setting USE_RAY=true. ms-swift's `swift sft`
# natively supports Ray via `--use_ray / --ray_exp_name / --device_groups`.
# RAY_DEVICE_GROUPS must be a JSON string when set, e.g.
#   '{"nproc_per_node":8,"default":{"device":"GPU","ranks":[0,1,2,3,4,5,6,7],"workers":["default"]}}'
# On Anyscale, dispatch_train.py derives it from the compute config.
# ---------------------------------------------------------------------------
USE_RAY="${USE_RAY:-false}"
RAY_EXP_NAME="${RAY_EXP_NAME:-}"
RAY_DEVICE_GROUPS="${RAY_DEVICE_GROUPS:-}"

# ---------------------------------------------------------------------------
# Helper: build extra tuner CLI args based on TUNER_TYPE (shared by both tasks).
# ---------------------------------------------------------------------------
_common_tuner_args() {
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
        args+=(--freeze_parameters "${FREEZE_PARAMETERS}")
    else
        args+=(--tuner_type full)
        args+=(--freeze_parameters "${FREEZE_PARAMETERS}")
    fi
    if [[ -n "${TRAINABLE_PARAMETERS}" ]]; then
        local p
        IFS=',' read -ra _PARTS <<< "${TRAINABLE_PARAMETERS}"
        for p in "${_PARTS[@]}"; do
            [[ -z "${p}" ]] && continue
            args+=(--trainable_parameters "${p}")
        done
    fi
    printf '%s\n' "${args[@]}"
}

# ---------------------------------------------------------------------------
# Shared `swift sft` invocation. Task-specific TEMPLATE / OUTPUT_ROOT / codec
# cache / streaming info are passed in by the per-task wrappers below.
# Args: <dataset_name> <task_name> <template> <output_root> <codec_cache>
#       <info_label> <streaming> <extra_lag_ms> [extra swift args...]
# ---------------------------------------------------------------------------
_run_swift_sft() {
    local dataset_name="$1"
    local task_name="$2"
    local template="$3"
    local output_root="$4"
    local codec_cache="$5"
    local info_label="$6"
    local streaming="$7"
    local extra_lag="$8"
    shift 8
    local output_dir="${OUTPUT_DIR:-${output_root}/${task_name}_${TUNER_TYPE}}"

    export WANDB_DIR="${WANDB_DIR:-${output_root}/wandb}"

    mkdir -p "${output_dir}" "${codec_cache}"

    local tuner_args=()
    while IFS= read -r line; do
        tuner_args+=("${line}")
    done < <(_common_tuner_args)

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

    local dataloader_args=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS}")
    if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
        dataloader_args+=(--dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}")
        if [[ -n "${DATALOADER_PREFETCH_FACTOR}" ]]; then
            dataloader_args+=(--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}")
        fi
    fi

    echo "[INFO] ${info_label}: streaming=${streaming} extra_lag_ms=${extra_lag}" >&2

    swift sft \
        --external_plugins "${PLUGIN_PATH}" \
        --model "${MODEL_PATH}" \
        --model_type "${MODEL_TYPE}" \
        --template "${template}" \
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

# ---------------------------------------------------------------------------
# Task: simultaneous (read-write) policy learning.
# ---------------------------------------------------------------------------
run_simultaneous_policy() {
    local dataset_name="$1"
    local task_name="$2"
    shift 2

    # Data / tokenizer / cache paths.
    export SIMULTANEOUS_POLICY_DATA="${SIMULTANEOUS_POLICY_DATA:-${REPO_ROOT}/pipelines/sft/data_format.jsonl}"
    export SIMULTANEOUS_POLICY_CODEC_CACHE="${SIMULTANEOUS_POLICY_CODEC_CACHE:-${SOURCE_ROOT}/.cache/simultaneous_policy_codec}"
    # The codec tokenizer resolves its on-disk cache via these env vars; reuse the
    # continue-pretrain key so both pipelines can share encoded codes.
    export CONTINUE_PRETRAIN_CODEC_CACHE="${CONTINUE_PRETRAIN_CODEC_CACHE:-${SIMULTANEOUS_POLICY_CODEC_CACHE}}"
    # This Stage-3 task is codec-only (no target-text prediction), so there is no
    # assistant text turn / thinker text CE — hence no *_APPEND_TARGET_TEXT.
    export SIMULTANEOUS_POLICY_STRICT="${SIMULTANEOUS_POLICY_STRICT:-false}"

    # Streaming read-write schedule switches. SIMULTANEOUS_POLICY_STREAMING=false
    # reverts to offline full-context codec attention (ablation baseline);
    # EXTRA_LAG_MS adds a global wait-k style lag (0 = pure data-driven policy).
    export SIMULTANEOUS_POLICY_STREAMING="${SIMULTANEOUS_POLICY_STREAMING:-true}"
    export SIMULTANEOUS_POLICY_EXTRA_LAG_MS="${SIMULTANEOUS_POLICY_EXTRA_LAG_MS:-0}"
    export SIMULTANEOUS_POLICY_SOURCE_DURATION_FROM_FILE="${SIMULTANEOUS_POLICY_SOURCE_DURATION_FROM_FILE:-true}"

    # trust_remote_code loads the modeling .py from the *export* dir; if you
    # edited custom/qwen_omnispeak/modeling_qwen_omnispeak.py make sure the
    # export copies were synced and this cache cleared (README §"Editing the model").
    export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${SOURCE_ROOT}/.cache/hf_modules_qwen_omnispeak_simul}"

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_simultaneous_policy}"

    _run_swift_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_simultaneous_policy}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_simultaneous_policy}" \
        "${SIMULTANEOUS_POLICY_CODEC_CACHE}" \
        "simultaneous policy" \
        "${SIMULTANEOUS_POLICY_STREAMING}" \
        "${SIMULTANEOUS_POLICY_EXTRA_LAG_MS}" \
        "$@"
}

# ---------------------------------------------------------------------------
# Task: normal_translation.
#
# Mechanically identical to simultaneous_policy_learning; it shares the same
# timeline sample validation rules and uses the isolated ``NORMAL_TRANSLATION_*``
# env prefix so this pipeline never cross-talks with the simultaneous-policy one.
# ---------------------------------------------------------------------------
run_normal_translation() {
    local dataset_name="$1"
    local task_name="$2"
    shift 2

    # Data / tokenizer / cache paths.
    export NORMAL_TRANSLATION_DATA="${NORMAL_TRANSLATION_DATA:-${REPO_ROOT}/pipelines/sft/data_format.jsonl}"
    export NORMAL_TRANSLATION_CODEC_CACHE="${NORMAL_TRANSLATION_CODEC_CACHE:-${SOURCE_ROOT}/.cache/normal_translation_codec}"
    # The codec tokenizer resolves its on-disk cache via these env vars; reuse the
    # continue-pretrain key so all pipelines can share encoded codes.
    export CONTINUE_PRETRAIN_CODEC_CACHE="${CONTINUE_PRETRAIN_CODEC_CACHE:-${NORMAL_TRANSLATION_CODEC_CACHE}}"
    # This Stage-3 task is codec-only (no target-text prediction), so there is no
    # assistant text turn / thinker text CE.
    export NORMAL_TRANSLATION_STRICT="${NORMAL_TRANSLATION_STRICT:-false}"

    # Streaming read-write schedule switches. NORMAL_TRANSLATION_STREAMING=false
    # reverts to offline full-context codec attention (ablation baseline);
    # EXTRA_LAG_MS adds a global wait-k style lag (0 = pure data-driven policy).
    export NORMAL_TRANSLATION_STREAMING="${NORMAL_TRANSLATION_STREAMING:-true}"
    export NORMAL_TRANSLATION_EXTRA_LAG_MS="${NORMAL_TRANSLATION_EXTRA_LAG_MS:-0}"
    export NORMAL_TRANSLATION_SOURCE_DURATION_FROM_FILE="${NORMAL_TRANSLATION_SOURCE_DURATION_FROM_FILE:-true}"

    # trust_remote_code loads the modeling .py from the *export* dir; if you
    # edited custom/qwen_omnispeak/modeling_qwen_omnispeak.py make sure the
    # export copies were synced and this cache cleared (README §"Editing the model").
    export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${SOURCE_ROOT}/.cache/hf_modules_qwen_omnispeak_normal_translation}"

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_normal_translation}"

    _run_swift_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_normal_translation}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_normal_translation}" \
        "${NORMAL_TRANSLATION_CODEC_CACHE}" \
        "normal translation" \
        "${NORMAL_TRANSLATION_STREAMING}" \
        "${NORMAL_TRANSLATION_EXTRA_LAG_MS}" \
        "$@"
}

# ---------------------------------------------------------------------------
# Task: bilingual_translation.
#
# Mechanically identical to normal_translation; it shares the same timeline
# sample validation rules and uses the isolated ``BILINGUAL_TRANSLATION_*`` env
# prefix so this pipeline never cross-talks with the simultaneous-policy or
# normal_translation ones.
# ---------------------------------------------------------------------------
run_bilingual_translation() {
    local dataset_name="$1"
    local task_name="$2"
    shift 2

    # Data / tokenizer / cache paths.
    export BILINGUAL_TRANSLATION_DATA="${BILINGUAL_TRANSLATION_DATA:-${REPO_ROOT}/pipelines/sft/data_format.jsonl}"
    export BILINGUAL_TRANSLATION_CODEC_CACHE="${BILINGUAL_TRANSLATION_CODEC_CACHE:-${SOURCE_ROOT}/.cache/bilingual_translation_codec}"
    # The codec tokenizer resolves its on-disk cache via these env vars; reuse the
    # continue-pretrain key so all pipelines can share encoded codes.
    export CONTINUE_PRETRAIN_CODEC_CACHE="${CONTINUE_PRETRAIN_CODEC_CACHE:-${BILINGUAL_TRANSLATION_CODEC_CACHE}}"
    # This Stage-3 task is codec-only (no target-text prediction), so there is no
    # assistant text turn / thinker text CE.
    export BILINGUAL_TRANSLATION_STRICT="${BILINGUAL_TRANSLATION_STRICT:-false}"

    # Streaming read-write schedule switches. BILINGUAL_TRANSLATION_STREAMING=false
    # reverts to offline full-context codec attention (ablation baseline);
    # EXTRA_LAG_MS adds a global wait-k style lag (0 = pure data-driven policy).
    export BILINGUAL_TRANSLATION_STREAMING="${BILINGUAL_TRANSLATION_STREAMING:-true}"
    export BILINGUAL_TRANSLATION_EXTRA_LAG_MS="${BILINGUAL_TRANSLATION_EXTRA_LAG_MS:-0}"
    export BILINGUAL_TRANSLATION_SOURCE_DURATION_FROM_FILE="${BILINGUAL_TRANSLATION_SOURCE_DURATION_FROM_FILE:-true}"

    # trust_remote_code loads the modeling .py from the *export* dir; if you
    # edited custom/qwen_omnispeak/modeling_qwen_omnispeak.py make sure the
    # export copies were synced and this cache cleared (README §"Editing the model").
    export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${SOURCE_ROOT}/.cache/hf_modules_qwen_omnispeak_bilingual_translation}"

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_bilingual_translation}"

    _run_swift_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_bilingual_translation}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_bilingual_translation}" \
        "${BILINGUAL_TRANSLATION_CODEC_CACHE}" \
        "bilingual translation" \
        "${BILINGUAL_TRANSLATION_STREAMING}" \
        "${BILINGUAL_TRANSLATION_EXTRA_LAG_MS}" \
        "$@"
}

# ---------------------------------------------------------------------------
# Task: multi_speaker_translation.
#
# Mechanically identical to normal_translation; it shares the same timeline
# sample validation rules and uses the isolated ``MULTI_SPEAKER_TRANSLATION_*``
# env prefix so this pipeline never cross-talks with the other Stage-3 pipelines.
# ---------------------------------------------------------------------------
run_multi_speaker_translation() {
    local dataset_name="$1"
    local task_name="$2"
    shift 2

    # Data / tokenizer / cache paths.
    export MULTI_SPEAKER_TRANSLATION_DATA="${MULTI_SPEAKER_TRANSLATION_DATA:-${REPO_ROOT}/pipelines/sft/data_format.jsonl}"
    export MULTI_SPEAKER_TRANSLATION_CODEC_CACHE="${MULTI_SPEAKER_TRANSLATION_CODEC_CACHE:-${SOURCE_ROOT}/.cache/multi_speaker_translation_codec}"
    # The codec tokenizer resolves its on-disk cache via these env vars; reuse the
    # continue-pretrain key so all pipelines can share encoded codes.
    export CONTINUE_PRETRAIN_CODEC_CACHE="${CONTINUE_PRETRAIN_CODEC_CACHE:-${MULTI_SPEAKER_TRANSLATION_CODEC_CACHE}}"
    # This Stage-3 task is codec-only (no target-text prediction), so there is no
    # assistant text turn / thinker text CE.
    export MULTI_SPEAKER_TRANSLATION_STRICT="${MULTI_SPEAKER_TRANSLATION_STRICT:-false}"

    # Streaming read-write schedule switches. MULTI_SPEAKER_TRANSLATION_STREAMING=false
    # reverts to offline full-context codec attention (ablation baseline);
    # EXTRA_LAG_MS adds a global wait-k style lag (0 = pure data-driven policy).
    export MULTI_SPEAKER_TRANSLATION_STREAMING="${MULTI_SPEAKER_TRANSLATION_STREAMING:-true}"
    export MULTI_SPEAKER_TRANSLATION_EXTRA_LAG_MS="${MULTI_SPEAKER_TRANSLATION_EXTRA_LAG_MS:-0}"
    export MULTI_SPEAKER_TRANSLATION_SOURCE_DURATION_FROM_FILE="${MULTI_SPEAKER_TRANSLATION_SOURCE_DURATION_FROM_FILE:-true}"

    # trust_remote_code loads the modeling .py from the *export* dir; if you
    # edited custom/qwen_omnispeak/modeling_qwen_omnispeak.py make sure the
    # export copies were synced and this cache cleared (README §"Editing the model").
    export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${SOURCE_ROOT}/.cache/hf_modules_qwen_omnispeak_multi_speaker_translation}"

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_multi_speaker_translation}"

    _run_swift_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_multi_speaker_translation}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_multi_speaker_translation}" \
        "${MULTI_SPEAKER_TRANSLATION_CODEC_CACHE}" \
        "multi speaker translation" \
        "${MULTI_SPEAKER_TRANSLATION_STREAMING}" \
        "${MULTI_SPEAKER_TRANSLATION_EXTRA_LAG_MS}" \
        "$@"
}

# ---------------------------------------------------------------------------
# Task: joint-stage1 (joint normal / bilingual / multi-speaker translation).
# ---------------------------------------------------------------------------
run_joint_stage1() {
    local dataset_name="$1"
    local task_name="$2"
    shift 2

    # Data / tokenizer / cache paths.
    export JOINT_STAGE1_DATA="${JOINT_STAGE1_DATA:-${SFT_JOINT_STAGE1_DATA:-${REPO_ROOT}/pipelines/sft/data_format.jsonl}}"
    export SFT_JOINT_STAGE1_DATA="${SFT_JOINT_STAGE1_DATA:-${JOINT_STAGE1_DATA}}"
    export SFT_JOINT_STAGE1_CODEC_CACHE="${SFT_JOINT_STAGE1_CODEC_CACHE:-${SOURCE_ROOT}/.cache/sft_joint_stage1_codec}"
    # The codec tokenizer resolves its on-disk cache via these env vars; reuse the
    # continue-pretrain key so all Stage-3 SFT pipelines can share encoded codes.
    export CONTINUE_PRETRAIN_CODEC_CACHE="${CONTINUE_PRETRAIN_CODEC_CACHE:-${SFT_JOINT_STAGE1_CODEC_CACHE}}"

    export JOINT_STAGE1_STRICT="${JOINT_STAGE1_STRICT:-false}"
    export JOINT_STAGE1_STREAMING="${JOINT_STAGE1_STREAMING:-true}"
    export JOINT_STAGE1_EXTRA_LAG_MS="${JOINT_STAGE1_EXTRA_LAG_MS:-0}"
    export JOINT_STAGE1_SOURCE_DURATION_FROM_FILE="${JOINT_STAGE1_SOURCE_DURATION_FROM_FILE:-true}"

    # trust_remote_code loads the modeling .py from the *export* dir; if you
    # edited custom/qwen_omnispeak/modeling_qwen_omnispeak.py make sure the
    # export copies were synced and this cache cleared (README §"Editing the model").
    export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${SOURCE_ROOT}/.cache/hf_modules_qwen_omnispeak_joint_stage1}"

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_sft_joint_stage1}"

    _run_swift_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_joint_stage1}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_sft_joint_stage1}" \
        "${SFT_JOINT_STAGE1_CODEC_CACHE}" \
        "joint-stage1" \
        "${JOINT_STAGE1_STREAMING}" \
        "${JOINT_STAGE1_EXTRA_LAG_MS}" \
        "$@"
}
