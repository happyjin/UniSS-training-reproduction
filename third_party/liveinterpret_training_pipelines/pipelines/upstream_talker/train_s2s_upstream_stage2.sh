#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Stage-2 upstream talker S2S training (reply-add / full-add).
#
# 相对上一阶段 (train_s2s_upstream.sh) 的区别：
#   * 从上一阶段训练得到的 checkpoint 继续训练（PREV_CHECKPOINT / MODEL_PATH）。
#   * 新增“解冻 thinker”开关 (UNFREEZE_THINKER)，把 ``model.thinker`` 加入可训练参数。
#   * 新增“thinker 侧文本 CE loss”开关 (THINKER_CE_LOSS / THINKER_LOSS_WEIGHT)，
#     通过环境变量 QWEN_OMNISPEAK_TEXT_LOSS / QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT 生效。
#   * 输出目录按批次 (BATCH_TAG) 建子文件夹，实验管理更规范。
#
# 说明：无论上一阶段来自哪个 branch（full-add / reply-add），只要把上一阶段的
#       checkpoint 目录传给 PREV_CHECKPOINT 即可（swift full-tune 保存的是完整
#       模型，checkpoint 目录本身就是一个可加载的 model 目录）。bridge-concat
#       分支请用 pipelines/bridge_alignment 下对应脚本 + 本文件同款 thinker 开关。
#
# Usage:
#   # full-add, 上阶段全解冻 talker，本阶段再解冻 thinker + 加 thinker CE：
#   BATCH_TAG=batch5.1 \
#   TALKER_METHOD=full_add UNFREEZE_TALKER_LAYERS=28 \
#   UNFREEZE_MTP=1 UNFREEZE_CODEC_HEAD=1 \
#   UNFREEZE_THINKER=1 THINKER_CE_LOSS=1 THINKER_LOSS_WEIGHT=1.0 \
#   PREV_CHECKPOINT=/abs/path/to/prev/vN-YYYYMMDD/checkpoint-XXXX \
#   bash train_s2s_upstream_stage2.sh
#
#   # reply-add, 解冻 thinker 但关掉 thinker CE（仅靠 codec loss 回传 thinker）：
#   TALKER_METHOD=reply_add UNFREEZE_THINKER=1 THINKER_CE_LOSS=0 \
#   PREV_CHECKPOINT=/abs/path/to/prev/checkpoint-XXXX \
#   bash train_s2s_upstream_stage2.sh
#
# Env knobs (在 train_s2s_upstream.sh 基础上新增/变更的部分):
#   PREV_CHECKPOINT         上一阶段 checkpoint 目录（等价于设置 MODEL_PATH）
#   UNFREEZE_THINKER        0 | 1   是否解冻 thinker (default: 1)
#   THINKER_CE_LOSS         0 | 1   是否启用 thinker 文本 CE loss (default: 1)
#   THINKER_LOSS_WEIGHT     thinker CE loss 权重 (default: 1.0)
#   BATCH_TAG               批次号，输出/日志都按此建子目录 (default: misc)
#
# 其余 env（TALKER_METHOD / UNFREEZE_TALKER_LAYERS / UNFREEZE_MTP /
# UNFREEZE_CODEC_HEAD / LEARNING_RATE / batch sizes / RESIDUAL_LOSS[_WEIGHT] …）
# 与 train_s2s_upstream.sh 完全一致，继承 common_args.sh 默认值。
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"
: "${WORKSPACE_ROOT:=$(cd "${REPO_ROOT}/../.." && pwd)}"
: "${DATA_ROOT:=${WORKSPACE_ROOT}/data}"
: "${S2S_DATA_VERSION:=bidirectional_v2_prompt_fixed}"
: "${MODELSCOPE_CACHE:=${WORKSPACE_ROOT}/.cache/modelscope_${S2S_DATA_VERSION}_stage2}"
: "${HF_DATASETS_CACHE:=${MODELSCOPE_CACHE}/datasets}"
export MODELSCOPE_CACHE HF_DATASETS_CACHE

# ----- Talker input method -----
: "${TALKER_METHOD:=full_add}"
case "${TALKER_METHOD}" in
    reply_add)
        export QWEN_OMNISPEAK_FULL_ADD_LAYOUT=false
        _method_tag="reply_add"
        ;;
    full_add)
        export QWEN_OMNISPEAK_FULL_ADD_LAYOUT=true
        _method_tag="full_add"
        ;;
    *)
        echo "[ERROR] TALKER_METHOD must be 'reply_add' or 'full_add', got: ${TALKER_METHOD}" >&2
        exit 1
        ;;
esac

# ----- Unfreeze talker layers -----
: "${UNFREEZE_TALKER_LAYERS:=28}"
case "${UNFREEZE_TALKER_LAYERS}" in
    0)  _talker_regex="" ;;
    2)  _talker_regex='model\.talker\.model\.layers\.[0-1]\.' ;;
    8)  _talker_regex='model\.talker\.model\.layers\.[0-7]\.' ;;
    28) _talker_regex='model\.talker\.model\.layers\.\d+\.' ;;
    *)
        echo "[ERROR] UNFREEZE_TALKER_LAYERS must be 0/2/8/28, got: ${UNFREEZE_TALKER_LAYERS}" >&2
        exit 1
        ;;
esac

# ----- Model: variant A (28-layer talker, use_upstream_talker_input=True) -----
: "${MODEL_TYPE:=qwen2_5_3b_omnispeak_a}"
: "${MODEL_PATH:=${SOURCE_ROOT}/models/qwen2_5_3b_omnispeak_a}"

# ----- 上一阶段 checkpoint：作为本阶段初始化权重（新优化器/新调度）-----
# PREV_CHECKPOINT 优先级高于 MODEL_PATH。若想继续同一次 run 的优化器状态/epoch，
# 请改用 RESUME_FROM（--resume_from_checkpoint）。
: "${PREV_CHECKPOINT:=}"
if [[ -n "${PREV_CHECKPOINT}" ]]; then
    if [[ ! -d "${PREV_CHECKPOINT}" ]]; then
        echo "[ERROR] PREV_CHECKPOINT 不是有效目录: ${PREV_CHECKPOINT}" >&2
        exit 1
    fi
    MODEL_PATH="${PREV_CHECKPOINT}"
fi
if [[ "${MODEL_PATH}" == "${SOURCE_ROOT}/models/qwen2_5_3b_omnispeak_a" ]]; then
    echo "[WARN] MODEL_PATH 仍指向 base 模型；stage-2 通常应设置 PREV_CHECKPOINT" >&2
    echo "[WARN]   指向上一阶段训练得到的 checkpoint 目录。" >&2
fi

# ----- Thinker 解冻 + thinker 文本 CE loss（本阶段新增的两个开关）-----
: "${UNFREEZE_THINKER:=1}"
: "${THINKER_CE_LOSS:=1}"
: "${THINKER_LOSS_WEIGHT:=1.0}"
# thinker CE loss 通过 modeling 里的 _text_loss_settings 读取的 env 生效。
if [[ "${THINKER_CE_LOSS}" == "1" || "${THINKER_CE_LOSS}" == "true" ]]; then
    export QWEN_OMNISPEAK_TEXT_LOSS=true
    export QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT="${THINKER_LOSS_WEIGHT}"
else
    export QWEN_OMNISPEAK_TEXT_LOSS=false
fi

# ----- Speaker prompt: S2S preserves SOURCE speaker voice -----
: "${QWEN_OMNISPEAK_USE_SPEAKER_PROMPT:=true}"
export QWEN_OMNISPEAK_USE_SPEAKER_PROMPT
if [[ -d "${SOURCE_ROOT}/models/Qwen3-TTS-12Hz-1.7B-Base" ]]; then
    export QWEN3_TTS_SPEAKER_ENCODER="${QWEN3_TTS_SPEAKER_ENCODER:-${SOURCE_ROOT}/models/Qwen3-TTS-12Hz-1.7B-Base}"
elif [[ -d "${SOURCE_ROOT}/project/Qwen3-TTS-12Hz-1.7B-Base" ]]; then
    export QWEN3_TTS_SPEAKER_ENCODER="${QWEN3_TTS_SPEAKER_ENCODER:-${SOURCE_ROOT}/project/Qwen3-TTS-12Hz-1.7B-Base}"
fi

# ----- Data -----
if [[ -z "${BRIDGE_ALIGNMENT_DATA:-}" ]]; then
    if compgen -G "${DATA_ROOT}/common_voice_s2s_${S2S_DATA_VERSION}/converted/*.parquet" >/dev/null; then
        export BRIDGE_ALIGNMENT_DATA="${DATA_ROOT}/common_voice_s2s_${S2S_DATA_VERSION}/converted/*.parquet"
    elif [[ -s "${SOURCE_ROOT}/datasets/jsonl_zeyang/speech_to_speech_translation_v2.tgt30s.local.dedup.s2sprompt.jsonl" ]]; then
        export BRIDGE_ALIGNMENT_DATA="${SOURCE_ROOT}/datasets/jsonl_zeyang/speech_to_speech_translation_v2.tgt30s.local.dedup.s2sprompt.jsonl"
    else
        export BRIDGE_ALIGNMENT_DATA="${SOURCE_ROOT}/datasets/jsonl_zeyang/speech_to_speech_translation_v2.tgt30s.local.dedup.jsonl"
    fi
fi

# ----- Naming tags (标准见 docs/EXPERIMENTS.md §0.1 实验命名规范) -----
: "${MODEL_SIZE_TAG:=3b}"
case "${UNFREEZE_TALKER_LAYERS}" in
    0)  _unfreeze_tag="adapter" ;;
    28) _unfreeze_tag="full" ;;
    *)  _unfreeze_tag="u${UNFREEZE_TALKER_LAYERS}" ;;
esac
# thinker 解冻后在 unfreeze 字段里标注（下划线在字段内、不破坏 dash 分隔）。
if [[ "${UNFREEZE_THINKER}" == "1" || "${UNFREEZE_THINKER}" == "true" ]]; then
    _unfreeze_tag="thk_${_unfreeze_tag}"
fi

# ----- wandb: project + group -----
: "${WANDB_PROJECT:=qwen_omnispeak_bridge_alignment}"
: "${WANDB_RUN_GROUP:=}"
if [[ -n "${WANDB_RUN_GROUP}" ]]; then
    export WANDB_RUN_GROUP
fi

# ----- Hyperparams -----
: "${NUM_TRAIN_EPOCHS:=5}"
: "${LEARNING_RATE:=1e-5}"
: "${WARMUP_RATIO:=0.03}"
: "${MAX_LENGTH:=4096}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=16}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=16}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${SAVE_STEPS:=500}"
: "${EVAL_STEPS:=100}"
: "${DATALOADER_NUM_WORKERS:=0}"
: "${DATALOADER_PERSISTENT_WORKERS:=false}"
: "${DATALOADER_PREFETCH_FACTOR:=}"
: "${SPLIT_DATASET_RATIO:=0.01}"
: "${LOAD_FROM_CACHE_FILE:=true}"
: "${MAX_CODEC_FRAMES:=384}"
export MAX_CODEC_FRAMES
: "${RESIDUAL_LOSS:=true}"
: "${RESIDUAL_LOSS_WEIGHT:=0.3}"
: "${CODEC_EOS_LOSS_WEIGHT:=1.0}"
: "${CODEC_HISTORY_CORRUPTION_PROB:=0.0}"

export MODEL_PATH MODEL_TYPE WANDB_PROJECT RUN_NAME \
       NUM_TRAIN_EPOCHS LEARNING_RATE WARMUP_RATIO MAX_LENGTH \
       PER_DEVICE_TRAIN_BATCH_SIZE PER_DEVICE_EVAL_BATCH_SIZE \
       GRADIENT_ACCUMULATION_STEPS SAVE_STEPS EVAL_STEPS \
       DATALOADER_NUM_WORKERS DATALOADER_PERSISTENT_WORKERS DATALOADER_PREFETCH_FACTOR \
       SPLIT_DATASET_RATIO LOAD_FROM_CACHE_FILE RESIDUAL_LOSS RESIDUAL_LOSS_WEIGHT \
       CODEC_EOS_LOSS_WEIGHT CODEC_HISTORY_CORRUPTION_PROB

# ---------------------------------------------------------------------------
# Conda / tokenizer / deepspeed (same as bridge_alignment scripts)
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-base}"
AUTO_ACTIVATE_CONDA="${AUTO_ACTIVATE_CONDA:-true}"
if [[ "${AUTO_ACTIVATE_CONDA}" == "true" ]]; then
    CONDA_SH="${CONDA_SH:-}"
    if [[ -z "${CONDA_SH}" ]]; then
        for candidate in \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/miniconda/etc/profile.d/conda.sh" \
            "/opt/dlami/nvme/zeyangsong/miniconda3/etc/profile.d/conda.sh" \
            "/root/miniconda3/etc/profile.d/conda.sh" \
            "/opt/miniconda3/etc/profile.d/conda.sh"; do
            if [[ -f "${candidate}" ]]; then
                CONDA_SH="${candidate}"
                break
            fi
        done
    fi
    if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
        set +u
        source "${CONDA_SH}"
        conda activate "${CONDA_ENV}"
        set -u
    else
        echo "[WARN] conda.sh not found; continuing with current shell environment." >&2
    fi
fi

if [[ -d "${SOURCE_ROOT}/models/Qwen3-TTS-Tokenizer-12Hz" ]]; then
    export QWEN3_TTS_TOKENIZER="${QWEN3_TTS_TOKENIZER:-${SOURCE_ROOT}/models/Qwen3-TTS-Tokenizer-12Hz}"
else
    export QWEN3_TTS_TOKENIZER="${QWEN3_TTS_TOKENIZER:-Qwen3/Qwen3-TTS-Tokenizer-12Hz}"
fi
export DEEPSPEED="${DEEPSPEED:-zero2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# common_args.sh handles: env defaults / GPU count / attn probe / deepspeed guard / wandb
source "${REPO_ROOT}/pipelines/bridge_alignment/common_args.sh"

# ---------------------------------------------------------------------------
# Build trainable parameters
#   Always: model.thinker_adapter (+ talker layers by UNFREEZE_TALKER_LAYERS regex)
#   UNFREEZE_MTP=1         -> + model.talker.code_predictor
#   UNFREEZE_CODEC_HEAD=1  -> + model.talker.codec_head
#   UNFREEZE_THINKER=1     -> + model.thinker  (整个 thinker 参与训练；此时 modeling
#                            里 run_thinker 自动走 grad 分支，thinker 文本 CE 的
#                            requires_grad=True，配合 QWEN_OMNISPEAK_TEXT_LOSS 生效)
# ---------------------------------------------------------------------------
_trainable_params="model.thinker_adapter"
: "${UNFREEZE_MTP:=0}"
if [[ "${UNFREEZE_MTP}" == "1" || "${UNFREEZE_MTP}" == "true" ]]; then
    _trainable_params="${_trainable_params} model.talker.code_predictor"
fi
: "${UNFREEZE_CODEC_HEAD:=0}"
if [[ "${UNFREEZE_CODEC_HEAD}" == "1" || "${UNFREEZE_CODEC_HEAD}" == "true" ]]; then
    _trainable_params="${_trainable_params} model.talker.codec_head"
fi
if [[ "${UNFREEZE_THINKER}" == "1" || "${UNFREEZE_THINKER}" == "true" ]]; then
    _trainable_params="${_trainable_params} model.thinker"
fi
_trainable_regex="${_talker_regex}"

_trainable_args=(--trainable_parameters ${_trainable_params})
if [[ -n "${_trainable_regex}" ]]; then
    _trainable_args+=(--trainable_parameters_regex "${_trainable_regex}")
fi

# ---------------------------------------------------------------------------
# Resume from checkpoint (继续同一次 run 的优化器/epoch；与 PREV_CHECKPOINT 二选一)
# ---------------------------------------------------------------------------
_resume_args=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    _resume_args+=(--resume_from_checkpoint "${RESUME_FROM}")
    echo "[upstream-stg2] RESUME_FROM=${RESUME_FROM}"
fi

# ---------------------------------------------------------------------------
# Standardized run name + output dir  (标准见 docs/EXPERIMENTS.md §0.1)
#   实验名: {method}-{unfreeze}-{model}-lr{lr}-bs{perdev}x{ngpu}-s2s-stg2[-{date}]
#   目录名: <BATCH_TAG>/upstream_<实验名去掉日期后缀>   (按批次建子目录)
# ---------------------------------------------------------------------------
: "${BATCH_TAG:=misc}"
_ngpu="${NPROC_PER_NODE:-1}"
_exp_tag="${_method_tag}-${_unfreeze_tag}-${MODEL_SIZE_TAG}-lr${LEARNING_RATE}-bs${PER_DEVICE_TRAIN_BATCH_SIZE}x${_ngpu}-s2s-stg2"
: "${RUN_NAME:=${_exp_tag}-$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/${BATCH_TAG}/upstream_${_exp_tag}}"
mkdir -p "${OUTPUT_DIR}" "${BRIDGE_ALIGNMENT_CODEC_CACHE}"

# ---------------------------------------------------------------------------
# wandb 额外 config
# ---------------------------------------------------------------------------
_effective_bs=$(( PER_DEVICE_TRAIN_BATCH_SIZE * _ngpu * GRADIENT_ACCUMULATION_STEPS ))
export WANDB_EXTRA_CONFIG="{\"stage\": 2, \"batch_tag\": \"${BATCH_TAG}\", \"talker_method\": \"${_method_tag}\", \"full_add_layout\": ${QWEN_OMNISPEAK_FULL_ADD_LAYOUT}, \"unfreeze_talker_layers\": ${UNFREEZE_TALKER_LAYERS}, \"unfreeze_mtp\": ${UNFREEZE_MTP}, \"unfreeze_codec_head\": ${UNFREEZE_CODEC_HEAD}, \"unfreeze_thinker\": ${UNFREEZE_THINKER}, \"thinker_ce_loss\": ${THINKER_CE_LOSS}, \"thinker_loss_weight\": ${THINKER_LOSS_WEIGHT}, \"residual_loss_weight\": ${RESIDUAL_LOSS_WEIGHT}, \"codec_eos_loss_weight\": ${CODEC_EOS_LOSS_WEIGHT}, \"codec_history_corruption_prob\": ${CODEC_HISTORY_CORRUPTION_PROB}, \"model_size\": \"${MODEL_SIZE_TAG}\", \"num_gpus\": ${_ngpu}, \"effective_batch_size\": ${_effective_bs}, \"prev_checkpoint\": \"${PREV_CHECKPOINT:-}\"}"

# ---------------------------------------------------------------------------
# Dataloader args
# ---------------------------------------------------------------------------
_dataloader_args=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS}")
if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
    _dataloader_args+=(--dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}")
    [[ -n "${DATALOADER_PREFETCH_FACTOR}" ]] && _dataloader_args+=(--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}")
fi

# ---------------------------------------------------------------------------
# Ray (optional)
# ---------------------------------------------------------------------------
_ray_args=()
if [[ "${USE_RAY:-false}" == "true" || "${USE_RAY:-false}" == "1" ]]; then
    _ray_args+=(--use_ray true)
    [[ -n "${RAY_EXP_NAME:-}" ]] && _ray_args+=(--ray_exp_name "${RAY_EXP_NAME}")
    [[ -n "${RAY_DEVICE_GROUPS:-}" ]] && _ray_args+=(--device_groups "${RAY_DEVICE_GROUPS}")
fi

echo "============================================================"
echo "[upstream-stg2] BATCH_TAG=${BATCH_TAG}"
echo "[upstream-stg2] METHOD=${_method_tag}  FULL_ADD_LAYOUT=${QWEN_OMNISPEAK_FULL_ADD_LAYOUT}"
echo "[upstream-stg2] UNFREEZE_TALKER_LAYERS=${UNFREEZE_TALKER_LAYERS}"
echo "[upstream-stg2] UNFREEZE_MTP=${UNFREEZE_MTP}  UNFREEZE_CODEC_HEAD=${UNFREEZE_CODEC_HEAD}"
echo "[upstream-stg2] UNFREEZE_THINKER=${UNFREEZE_THINKER}"
echo "[upstream-stg2] THINKER_CE_LOSS=${THINKER_CE_LOSS} (QWEN_OMNISPEAK_TEXT_LOSS=${QWEN_OMNISPEAK_TEXT_LOSS}) weight=${QWEN_OMNISPEAK_TEXT_LOSS_WEIGHT:-<n/a>}"
echo "[upstream-stg2] RESIDUAL_LOSS=${RESIDUAL_LOSS} weight=${RESIDUAL_LOSS_WEIGHT}"
echo "[upstream-stg2] CODEC_EOS_WEIGHT=${CODEC_EOS_LOSS_WEIGHT}  HISTORY_CORRUPTION=${CODEC_HISTORY_CORRUPTION_PROB}"
echo "[upstream-stg2] trainable_params=${_trainable_params}  regex=${_trainable_regex:-<none>}"
echo "[upstream-stg2] MODEL=${MODEL_PATH}"
echo "[upstream-stg2] PREV_CHECKPOINT=${PREV_CHECKPOINT:-<none>}"
echo "[upstream-stg2] DATA=${BRIDGE_ALIGNMENT_DATA}"
echo "[upstream-stg2] MODELSCOPE_CACHE=${MODELSCOPE_CACHE}"
echo "[upstream-stg2] LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE}"
echo "[upstream-stg2] WDS mode=${WDS_AUDIO_MODE} root=${WDS_AUDIO_ROOT:-<unset>} strict=${WDS_AUDIO_STRICT}"
echo "[upstream-stg2] SPEAKER=${QWEN_OMNISPEAK_USE_SPEAKER_PROMPT}"
echo "[upstream-stg2] LR=${LEARNING_RATE}  BS=${PER_DEVICE_TRAIN_BATCH_SIZE}x${_ngpu}(x ga${GRADIENT_ACCUMULATION_STEPS})=eff${_effective_bs}  EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[upstream-stg2] WANDB project=${WANDB_PROJECT}  group=${WANDB_RUN_GROUP:-<none>}"
echo "[upstream-stg2] RUN_NAME=${RUN_NAME}"
echo "[upstream-stg2] OUTPUT=${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Unified training log (规范见 docs/EXPERIMENTS.md §2.2)
#   logs/<BATCH_TAG>/<RUN_NAME>.log
# ---------------------------------------------------------------------------
_LOG_DIR="${REPO_ROOT}/logs/${BATCH_TAG}"
mkdir -p "${_LOG_DIR}"
_LOG_FILE="${_LOG_DIR}/${RUN_NAME}.log"
echo "[upstream-stg2] LOG=${_LOG_FILE}"
echo "============================================================"

swift sft \
    --external_plugins "${PLUGIN_PATH}" \
    --model "${MODEL_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --template "${TEMPLATE}" \
    --dataset bridge_alignment_speech_to_speech_translation \
    --load_from_cache_file "${LOAD_FROM_CACHE_FILE}" \
    --remove_unused_columns false \
    "${_ray_args[@]}" \
    --tuner_type full \
    --freeze_parameters_ratio 1.0 \
    "${_trainable_args[@]}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --attn_impl "${ATTN_IMPL}" \
    --deepspeed "${DEEPSPEED}" \
    --gradient_checkpointing true \
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
    "${_dataloader_args[@]}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --output_dir "${OUTPUT_DIR}" \
    "${_resume_args[@]}" \
    "$@" 2>&1 | tee -a "${_LOG_FILE}"
