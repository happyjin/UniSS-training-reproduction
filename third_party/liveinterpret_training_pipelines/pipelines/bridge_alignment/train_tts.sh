#!/usr/bin/env bash
# =============================================================================
# train_tts.sh — 统一的 TTS bridge-alignment 训练入口
#
# 通过 TRAIN_BACKEND 选择后端（deepspeed / megatron）。TTS 任务固定 bridge 模式
# （sem_block）。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ---------------------------------------------------------------------------
# 后端
# ---------------------------------------------------------------------------
: "${TRAIN_BACKEND:=deepspeed}"

# ---------------------------------------------------------------------------
# 任务身份
# ---------------------------------------------------------------------------
: "${MODEL_TYPE:=qwen3_omnispeak_d}"
: "${MODEL_PATH:=${NVME_MOUNT:-/}/origin_models/custom_models/qwen3_omnispeak_d}"

# ---------------------------------------------------------------------------
# 训练目标：bridge（经典 bridge alignment）
# ---------------------------------------------------------------------------
export QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT="${QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT:-false}"
: "${TALKER_METHOD:=sem_block}"
_target_tag="bridge"
_method_tag="${TALKER_METHOD}"

# ---------------------------------------------------------------------------
# 任务数据（WDS）
# ---------------------------------------------------------------------------
: "${WDS_AUDIO_MODE:=required}"
: "${WDS_AUDIO_STRICT:=true}"
: "${BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT:=true}"
: "${QWEN_OMNISPEAK_USE_ICL:=false}"

if [[ -n "${BRIDGE_ALIGNMENT_DATA:-}" && -d "${BRIDGE_ALIGNMENT_DATA}" ]]; then
    BRIDGE_ALIGNMENT_DATA="${BRIDGE_ALIGNMENT_DATA%/}/*.jsonl"
fi

export WDS_AUDIO_MODE WDS_AUDIO_STRICT
export BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT QWEN_OMNISPEAK_USE_ICL
export BRIDGE_ALIGNMENT_DATA

# per-method cache
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${NVME_MOUNT:-/}/.cache/datasets_${_method_tag}_tts}"

echo "[tts] BACKEND=${TRAIN_BACKEND} TARGET=${_target_tag} METHOD=${_method_tag}"
echo "[tts] DATA=${BRIDGE_ALIGNMENT_DATA:-<unset>}"
echo "[tts] WDS audio at /curated_audio (symlink -> /nvme/data)"
echo "[tts] NORMALIZE_PROMPT=${BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT} USE_ICL=${QWEN_OMNISPEAK_USE_ICL}"
echo "[tts] UPSTREAM_TALKER_INPUT=${QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT:-false}"

# ---------------------------------------------------------------------------
# 可训参数（deepspeed 用 --trainable_parameters；megatron 用 --freeze_*）
# ---------------------------------------------------------------------------
if [[ "${TRAIN_BACKEND}" == "deepspeed" ]]; then
    _trainable_params="model.bridge"
    : "${TALKER_TRAINING_SCOPE:=layers}"
    : "${UNFREEZE_TALKER_LAYERS:=0}"
    case "${UNFREEZE_TALKER_LAYERS}" in
        0)  _trainable_regex="" ;;
        2)  _trainable_regex='model\.talker\.model\.layers\.[0-1]\.' ;;
        8)  _trainable_regex='model\.talker\.model\.layers\.[0-7]\.' ;;
        28) _trainable_regex='model\.talker\.model\.layers\.\d+\.' ;;
        *) echo "[ERROR] UNFREEZE_TALKER_LAYERS must be 0/2/8/28" >&2; exit 1 ;;
    esac
    : "${UNFREEZE_THINKER:=0}"
    : "${UNFREEZE_MTP:=0}"
    : "${UNFREEZE_CODEC_HEAD:=0}"
    case "${TALKER_TRAINING_SCOPE}" in
        whole)
            _trainable_params="${_trainable_params} model.talker"
            _trainable_regex=""
            ;;
        layers)
            [[ "${UNFREEZE_MTP}" == "1" || "${UNFREEZE_MTP}" == "true" ]] && _trainable_params="${_trainable_params} model.talker.code_predictor"
            [[ "${UNFREEZE_CODEC_HEAD}" == "1" || "${UNFREEZE_CODEC_HEAD}" == "true" ]] && _trainable_params="${_trainable_params} model.talker.codec_head"
            ;;
        *) echo "[ERROR] TALKER_TRAINING_SCOPE must be 'layers' or 'whole'" >&2; exit 1 ;;
    esac
    [[ "${UNFREEZE_THINKER}" == "1" || "${UNFREEZE_THINKER}" == "true" ]] && _trainable_params="${_trainable_params} model.thinker"
    export _trainable_params _trainable_regex
fi

# ---------------------------------------------------------------------------
# 任务输出
# ---------------------------------------------------------------------------
: "${MODEL_SIZE_TAG:=3b}"
if [[ "${TRAIN_BACKEND}" == "deepspeed" ]]; then
    _unfreeze_tag="${TALKER_TRAINING_SCOPE:-layers}"
    if [[ "${_unfreeze_tag}" == "layers" ]]; then
        case "${UNFREEZE_TALKER_LAYERS:-0}" in
            0)  _unfreeze_tag="adapter" ;;
            28) _unfreeze_tag="full" ;;
            *)  _unfreeze_tag="u${UNFREEZE_TALKER_LAYERS}" ;;
        esac
    fi
else
    _unfreeze_tag="${FREEZE_ALIGNER:-false}"
    [[ "${_unfreeze_tag}" == "false" ]] && _unfreeze_tag="bridge"
fi

_ngpu="${NPROC_PER_NODE:-1}"
: "${RUN_NAME:=${_method_tag}-${_target_tag}-${_unfreeze_tag}-${MODEL_SIZE_TAG}-tts-$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME
: "${OUTPUT_DIR:=${OUTPUT_ROOT:-${NVME_MOUNT:-/}/output}/${_method_tag}-${_target_tag}-${_unfreeze_tag}-${MODEL_SIZE_TAG}-tts}"

# ---------------------------------------------------------------------------
# 任务超参覆盖（仅覆盖与 common 默认不同的项）
# ---------------------------------------------------------------------------
: "${NUM_TRAIN_EPOCHS:=1}"
: "${EVAL_STEPS:=200}"
: "${RESIDUAL_LOSS_WEIGHT:=0.3}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=16}"
: "${DATALOADER_NUM_WORKERS:=0}"

# ---------------------------------------------------------------------------
# source 统一的 common_args.sh
# ---------------------------------------------------------------------------
source "${REPO_ROOT}/pipelines/bridge_alignment/common_args.sh"

# ===========================================================================
# [DIAG] 工件路径证据采集
# ===========================================================================
echo "==================== ARTIFACT PATH DIAGNOSTICS ===================="
echo "[DIAG] MODEL_PATH            = ${MODEL_PATH}"
echo "[DIAG] QWEN3_TTS_TOKENIZER   = ${QWEN3_TTS_TOKENIZER}"
echo "[DIAG] BRIDGE_ALIGNMENT_DATA = ${BRIDGE_ALIGNMENT_DATA}"
echo "[DIAG] NVME_MOUNT            = ${NVME_MOUNT:-/}"
echo "[DIAG] /curated_audio        = $(ls -ld /curated_audio 2>/dev/null || echo '<missing>')"
echo "[DIAG] MODEL_PATH is dir?    = $([[ -d "${MODEL_PATH}" ]] && echo YES || echo NO)"
if ls "${MODEL_PATH}"/*.safetensors >/dev/null 2>&1; then
  echo "[DIAG] model safetensors:"
  ls -l "${MODEL_PATH}"/*.safetensors | sed 's/^/    /'
else
  echo "[DIAG] <no *.safetensors found under MODEL_PATH>"
fi
echo "=================================================================="

echo "============================================================"
echo "[tts] BACKEND=${TRAIN_BACKEND} TARGET=${_target_tag}"
echo "[tts] MODEL=${MODEL_TYPE} @ ${MODEL_PATH}"
if [[ "${TRAIN_BACKEND}" == "deepspeed" ]]; then
    echo "[tts] DEEPSPEED=${DEEPSPEED}  ATTN_IMPL=${ATTN_IMPL}"
    echo "[tts] trainable_params=${_trainable_params}  regex=${_trainable_regex:-<none>}"
    echo "[tts] LR=${LEARNING_RATE}  BS=${PER_DEVICE_TRAIN_BATCH_SIZE}x${_ngpu}x${GRADIENT_ACCUMULATION_STEPS}"
else
    echo "[tts] SCENE=${SCENE}  TP=${TP} PP=${PP} CP=${CP}"
    echo "[tts] ATTENTION_BACKEND=${ATTENTION_BACKEND}"
    echo "[tts] LR=${LR}  MBS=${MICRO_BATCH_SIZE}  GBS=${GLOBAL_BATCH_SIZE}"
fi
echo "[tts] WANDB project=${WANDB_PROJECT}  RUN_NAME=${RUN_NAME}"
echo "[tts] OUTPUT=${OUTPUT_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 调用统一分发函数
# ---------------------------------------------------------------------------
run_train bridge_alignment_tts tts "$@"
