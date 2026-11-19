#!/usr/bin/env bash
# Copyright (c) ModelScope Contributors. All rights reserved.
#
# Variant 3: `megatron sft` (TP / EP + DistributedOptimizer / ZeRO-1) shared by
# the Stage-3 SFT *streaming speech-to-speech translation* pipelines for
# qwen_omnispeak.
#
# Stage-3 sibling tasks share this file; a task script sources it and then calls one
# of the helpers:
#   * ``run_simultaneous_policy_megatron <dataset_name> <task_name>`` —
#     simultaneous (read-write) policy learning.
#   * ``run_normal_translation_megatron <dataset_name> <task_name>``   —
#     normal_translation.
#   * ``run_bilingual_translation_megatron <dataset_name> <task_name>`` —
#     bilingual_translation.
#   * ``run_multi_speaker_translation_megatron <dataset_name> <task_name>`` —
#     multi_speaker_translation.
#   * ``run_joint_stage1_megatron <dataset_name> <task_name>`` — joint normal /
#     bilingual / multi-speaker translation.
#
# ===========================================================================
# How the read-write policy is trained under Megatron (ZERO task-specific code)
# ---------------------------------------------------------------------------
# The non-invasive mcore-bridge adaptation maps the *thinker* transformer to a
# Megatron GPTModel (TP/EP sharded) and loads audio_tower / visual / bridge /
# TALKER as replicated submodules. ``mcore_talker_loss.py`` captures the thinker
# hidden states and computes the talker codec loss (main + residual CE) -- the
# SAME objective as the DeepSpeed path (verified by test_mcore_talker_loss.py).
#
# That codec-loss path is **task-agnostic**: it pops ``codec_to_sem_visible``
# from the batch and forwards it into ``build_block_causal_mask``, and (when the
# batch carries no ``has_codec_supervision``) enters ``bridge_mode`` to zero the
# meaningless thinker text loss. Both task templates produce a codec batch +
# schedule with the EXACT SAME structure, so NO Megatron-side changes are
# required:
#   * WRITE (when to speak) is supervised by the codec CE over the real-timeline
#     codec stream (silence frames + speech frames);
#   * READ (how much source is visible) is enforced by the streaming attention
#     mask, per target codec frame.
#
# Constraints (same as continue-pretrain s2s): the codec / streaming path
# requires PP=1, CP=1, sequence_parallel=false, no packing/padding-free. The
# thinker still uses full TP/EP/DP/ZeRO-1. Disable the codec loss entirely with
# QWEN_OMNISPEAK_MCORE_CODEC_LOSS=false (thinker text training only), and disable
# the streaming mask with *_STREAMING=false (offline ablation).
#
# The Stage-3 streaming tasks are mechanically identical and share the same
# timeline sample validation rules: source/target chunks must be strictly
# non-overlapping, and the target timeline duration must satisfy the common tail
# bound. They differ only in the env-var prefix (``SIMULTANEOUS_POLICY_*`` /
# ``NORMAL_TRANSLATION_*`` / ``BILINGUAL_TRANSLATION_*`` /
# ``MULTI_SPEAKER_TRANSLATION_*``) plus task-specific TEMPLATE / OUTPUT_ROOT /
# cache defaults, which each ``run_*_megatron`` helper sets for itself so the
# tasks never cross-talk.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Layout: <repo>/liveinterpret_training_pipelines/pipelines/sft/
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"

# OUTPUT_ROOT: 所有训练任务统一默认输出根，便于 anyscale 按统一路径归档 S3。
# 各 run_*_megatron helper 调用时传入的 "${OUTPUT_ROOT:-...}" 任务级兜底仅在此处未设时才生效。
: "${OUTPUT_ROOT:=${SOURCE_ROOT}/output/qwen_omnispeak_output}"
export OUTPUT_ROOT

# ---------------------------------------------------------------------------
# Tokenizer / CUDA alloc (shared by both tasks). Prefer the local NVMe/repo copy
# when it exists; otherwise fall back to HuggingFace Hub for local dev without
# synced models.
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Plugin / model. Anyscale injects PLUGIN_PATH / MODEL_PATH / MODEL_TYPE via
# env.paths.sh; local runs fall back to the repository layout.
# ---------------------------------------------------------------------------
PLUGIN_PATH="${PLUGIN_PATH:-${REPO_ROOT}/custom/plugin.py}"
: "${MODEL_PATH:=${SOURCE_ROOT}/models/qwen3_omnispeak_d}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_omnispeak_d}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# ---------------------------------------------------------------------------
# Model parallelism (TP / PP / CP / EP). DP is derived from world_size.
# NOTE: the talker codec + streaming path requires PP=1, CP=1, SP=false.
# ---------------------------------------------------------------------------
TP="${TP:-2}"
PP="${PP:-1}"
CP="${CP:-1}"
EP="${EP:-1}"
ETP="${ETP:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-false}"

# Talker codec loss controls (forwarded to the trainer patch). Auto-inactive for
# text-only rows; active for rows that carry codec supervision (target_audios).
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

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
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

# ---------------------------------------------------------------------------
# Helper: build extra tuner CLI args based on TUNER_TYPE (shared by both tasks).
# ---------------------------------------------------------------------------
_common_megatron_tuner_args() {
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

# ---------------------------------------------------------------------------
# Shared `megatron sft` invocation. Task-specific TEMPLATE / OUTPUT_ROOT / codec
# cache / streaming info are passed in by the per-task wrappers below.
# Args: <dataset_name> <task_name> <template> <output_root> <codec_cache>
#       <info_label> <streaming> <extra_lag_ms> [extra megatron args...]
# ---------------------------------------------------------------------------
_run_megatron_sft() {
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
    done < <(_common_megatron_tuner_args)

    local moe_args=()
    if [[ "${EP}" -gt 1 ]]; then
        moe_args+=(--expert_model_parallel_size "${EP}")
        moe_args+=(--expert_tensor_parallel_size "${ETP}")
        moe_args+=(--moe_grouped_gemm true)
        moe_args+=(--moe_permute_fusion true)
        moe_args+=(--moe_shared_expert_overlap true)
        moe_args+=(--moe_aux_loss_coeff 1e-3)
    fi

    echo "[INFO] ${info_label} (megatron): streaming=${streaming} extra_lag_ms=${extra_lag} codec_loss=${QWEN_OMNISPEAK_MCORE_CODEC_LOSS} TP=${TP} PP=${PP} CP=${CP} EP=${EP} SP=${SEQUENCE_PARALLEL}" >&2

    megatron sft \
        --external_plugins "${PLUGIN_PATH}" \
        --model "${MODEL_PATH}" \
        --model_type "${MODEL_TYPE}" \
        --template "${template}" \
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

# ---------------------------------------------------------------------------
# Task: simultaneous (read-write) policy learning.
# ---------------------------------------------------------------------------
run_simultaneous_policy_megatron() {
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

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_simultaneous_policy_megatron}"

    _run_megatron_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_simultaneous_policy}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_simultaneous_policy_megatron}" \
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
run_normal_translation_megatron() {
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

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_normal_translation_megatron}"

    _run_megatron_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_normal_translation}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_normal_translation_megatron}" \
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
run_bilingual_translation_megatron() {
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

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_bilingual_translation_megatron}"

    _run_megatron_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_bilingual_translation}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_bilingual_translation_megatron}" \
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
run_multi_speaker_translation_megatron() {
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

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_multi_speaker_translation_megatron}"

    _run_megatron_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_multi_speaker_translation}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_multi_speaker_translation_megatron}" \
        "${MULTI_SPEAKER_TRANSLATION_CODEC_CACHE}" \
        "multi speaker translation" \
        "${MULTI_SPEAKER_TRANSLATION_STREAMING}" \
        "${MULTI_SPEAKER_TRANSLATION_EXTRA_LAG_MS}" \
        "$@"
}

# ---------------------------------------------------------------------------
# Task: joint-stage1 (joint normal / bilingual / multi-speaker translation).
# ---------------------------------------------------------------------------
run_joint_stage1_megatron() {
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

    export WANDB_PROJECT="${WANDB_PROJECT:-qwen_omnispeak_sft_joint_stage1_megatron}"

    _run_megatron_sft \
        "${dataset_name}" \
        "${task_name}" \
        "${TEMPLATE:-qwen_omnispeak_joint_stage1}" \
        "${OUTPUT_ROOT:-${SOURCE_ROOT}/output/qwen_omnispeak_sft_joint_stage1_megatron}" \
        "${SFT_JOINT_STAGE1_CODEC_CACHE}" \
        "joint-stage1" \
        "${JOINT_STAGE1_STREAMING}" \
        "${JOINT_STAGE1_EXTRA_LAG_MS}" \
        "$@"
}
