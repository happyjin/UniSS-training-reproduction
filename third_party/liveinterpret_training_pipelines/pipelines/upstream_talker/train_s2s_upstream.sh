#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Unified upstream talker S2S training (reply-add / full-add).
#
# Usage:
#   # reply-add, unfreeze 2 talker layers:
#   TALKER_METHOD=reply_add  UNFREEZE_TALKER_LAYERS=2  bash train_s2s_upstream.sh
#
#   # full-add, unfreeze 8 talker layers:
#   TALKER_METHOD=full_add   UNFREEZE_TALKER_LAYERS=8  bash train_s2s_upstream.sh
#
#   # full-add, adapter only (no talker layers):
#   TALKER_METHOD=full_add   UNFREEZE_TALKER_LAYERS=0  bash train_s2s_upstream.sh
#
# Env knobs:
#   TALKER_METHOD           reply_add | full_add  (default: full_add)
#   UNFREEZE_TALKER_LAYERS  0 | 2 | 8 | 28       (default: 0 = adapter only)
#   TALKER_TRAINING_SCOPE   layers | whole        (whole includes MTP/codec head)
#   UNFREEZE_THINKER        0 | 1                  (default: 0)
#   BRIDGE_ALIGNMENT_DATA   path to the training JSONL/Parquet
#   UPSTREAM_DATASET_NAME   registered ms-swift dataset name
#   UPSTREAM_TASK_TAG       run/output naming suffix (default: s2s)
#   UPSTREAM_DEFAULT_DATA   task wrapper's default dataset path, if any
#   MODEL_PATH / MODEL_TYPE override the model (default: qwen2_5_3b_omnispeak_a)
#   CODEC_EOS_LOSS_WEIGHT / CODEC_HISTORY_CORRUPTION_PROB (default: 1.0 / 0.0)
#   RESUME_FROM             path to a checkpoint dir to resume from
#
# All other env vars (LEARNING_RATE, NUM_TRAIN_EPOCHS, batch sizes, etc.)
# inherit from common_args.sh defaults and can be overridden.
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"
: "${WORKSPACE_ROOT:=$(cd "${REPO_ROOT}/../.." && pwd)}"
: "${DATA_ROOT:=${WORKSPACE_ROOT}/data}"
: "${S2S_DATA_VERSION:=bidirectional_v2_prompt_fixed}"
: "${UPSTREAM_DATASET_NAME:=bridge_alignment_speech_to_speech_translation}"
: "${UPSTREAM_TASK_TAG:=s2s}"
: "${UPSTREAM_DEFAULT_DATA:=}"
: "${UPSTREAM_DEFAULT_SPEAKER_PROMPT:=true}"
: "${TALKER_TRAINING_SCOPE:=layers}"
: "${UNFREEZE_THINKER:=0}"

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

# ----- Per-method/version datasets cache (avoid stale map-cache reuse) -----
# ms-swift local-path loading uses MODELSCOPE_CACHE/datasets explicitly.
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${WORKSPACE_ROOT}/.cache/modelscope_${_method_tag}_${UPSTREAM_TASK_TAG}_${S2S_DATA_VERSION}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MODELSCOPE_CACHE}/datasets}"

# ----- Unfreeze talker layers -----
: "${UNFREEZE_TALKER_LAYERS:=0}"
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

# ----- Speaker prompt: S2S preserves SOURCE speaker voice -----
: "${QWEN_OMNISPEAK_USE_SPEAKER_PROMPT:=${UPSTREAM_DEFAULT_SPEAKER_PROMPT}}"
export QWEN_OMNISPEAK_USE_SPEAKER_PROMPT
if [[ -d "${SOURCE_ROOT}/models/Qwen3-TTS-12Hz-1.7B-Base" ]]; then
    export QWEN3_TTS_SPEAKER_ENCODER="${QWEN3_TTS_SPEAKER_ENCODER:-${SOURCE_ROOT}/models/Qwen3-TTS-12Hz-1.7B-Base}"
elif [[ -d "${SOURCE_ROOT}/project/Qwen3-TTS-12Hz-1.7B-Base" ]]; then
    export QWEN3_TTS_SPEAKER_ENCODER="${QWEN3_TTS_SPEAKER_ENCODER:-${SOURCE_ROOT}/project/Qwen3-TTS-12Hz-1.7B-Base}"
fi

# ----- Data -----
if [[ -z "${BRIDGE_ALIGNMENT_DATA:-}" ]]; then
    if [[ -n "${UPSTREAM_DEFAULT_DATA}" ]]; then
        if [[ ! -s "${UPSTREAM_DEFAULT_DATA}" ]]; then
            echo "[ERROR] UPSTREAM_DEFAULT_DATA does not exist or is empty: ${UPSTREAM_DEFAULT_DATA}" >&2
            exit 1
        fi
        export BRIDGE_ALIGNMENT_DATA="${UPSTREAM_DEFAULT_DATA}"
    elif compgen -G "${DATA_ROOT}/common_voice_s2s_${S2S_DATA_VERSION}/converted/*.parquet" >/dev/null; then
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

# ----- wandb: project + group -----
# WANDB_RUN_GROUP 用于把同一 ablation 主题的多个 run 聚到一个 group，
# 例如 talker 输入方式对比：WANDB_RUN_GROUP="talker input ablation"。
: "${WANDB_PROJECT:=qwen_omnispeak_t2s_translation}"
: "${WANDB_RUN_GROUP:=}"
if [[ -n "${WANDB_RUN_GROUP}" ]]; then
    export WANDB_RUN_GROUP
fi

# ----- Hyperparams -----
: "${NUM_TRAIN_EPOCHS:=5}"
: "${LEARNING_RATE:=1e-4}"
: "${WARMUP_RATIO:=0.03}"
: "${MAX_LENGTH:=4096}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=16}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=16}"
# 默认无梯度累积（有效 batch = per_device × 卡数）。talker input ablation 用 ga=1。
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${SAVE_STEPS:=500}"
: "${EVAL_STEPS:=100}"
: "${DATALOADER_NUM_WORKERS:=0}"
: "${DATALOADER_PERSISTENT_WORKERS:=false}"
: "${DATALOADER_PREFETCH_FACTOR:=}"
: "${SPLIT_DATASET_RATIO:=0.01}"
: "${MAX_CODEC_FRAMES:=384}"
export MAX_CODEC_FRAMES
: "${RESIDUAL_LOSS:=true}"
: "${RESIDUAL_LOSS_WEIGHT:=0.3}"
: "${CODEC_EOS_LOSS_WEIGHT:=1.0}"
: "${CODEC_HISTORY_CORRUPTION_PROB:=0.0}"
# group_by_length: 按序列长度分桶采样，减少 batch 内 padding 浪费与 DDP 负载不均。
# 需要 train_dataloader_shuffle=True（swift 默认），分桶内仍带随机因子。
# 开启后 swift 对 multimodal 会关闭 lazy_tokenize，全量 encode 并写入 lengths 列
# （由 load_from_cache_file 缓存），再由 LengthGroupedSampler 分桶。
: "${GROUP_BY_LENGTH:=false}"

export MODEL_PATH MODEL_TYPE WANDB_PROJECT RUN_NAME \
       NUM_TRAIN_EPOCHS LEARNING_RATE WARMUP_RATIO MAX_LENGTH \
       PER_DEVICE_TRAIN_BATCH_SIZE PER_DEVICE_EVAL_BATCH_SIZE \
       GRADIENT_ACCUMULATION_STEPS SAVE_STEPS EVAL_STEPS \
       DATALOADER_NUM_WORKERS DATALOADER_PERSISTENT_WORKERS DATALOADER_PREFETCH_FACTOR \
       SPLIT_DATASET_RATIO RESIDUAL_LOSS RESIDUAL_LOSS_WEIGHT \
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
            "/opt/dlami/nvme/zeyangsong/miniconda/etc/profile.d/conda.sh" \
            "/root/miniconda3/etc/profile.d/conda.sh" \
            "/opt/miniconda3/etc/profile.d/conda.sh"; do
            if [[ -f "${candidate}" ]]; then
                CONDA_SH="${candidate}"
                break
            fi
        done
    fi
    if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
        # conda activate.d scripts may reference unbound vars (e.g.
        # NVCC_PREPEND_FLAGS); temporarily relax nounset to avoid abort.
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
# Do not export an empty key: that masks credentials stored by `wandb login`.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
else
    unset WANDB_API_KEY
fi

# common_args.sh handles: env defaults / GPU count / attn probe / deepspeed guard / wandb
source "${REPO_ROOT}/pipelines/bridge_alignment/common_args.sh"

# ---------------------------------------------------------------------------
# Build trainable parameters
#   Always: model.thinker_adapter (+ talker layers by UNFREEZE_TALKER_LAYERS regex)
#   UNFREEZE_MTP=1  additionally unfreezes the residual code predictor (MTP),
#                   i.e. ``model.talker.code_predictor`` (small_to_mtp_projection
#                   + its transformer + grouped lm_heads). Default 0 = MTP frozen
#                   (uses pretrained Qwen3-TTS residual predictor as-is).
#   UNFREEZE_CODEC_HEAD=1  additionally unfreezes the main-code head
#                   ``model.talker.codec_head`` (main codec logits projection).
#                   Default 0 = codec_head frozen (uses pretrained weights).
# ---------------------------------------------------------------------------
_trainable_params="model.thinker_adapter"
: "${UNFREEZE_MTP:=0}"
: "${UNFREEZE_CODEC_HEAD:=0}"
case "${TALKER_TRAINING_SCOPE}" in
    whole)
        # Prefix selection covers all talker parameters, including transformer
        # layers, text_projection, embeddings, MTP/code_predictor and codec_head.
        _trainable_params="${_trainable_params} model.talker"
        _trainable_regex=""
        ;;
    layers)
        if [[ "${UNFREEZE_MTP}" == "1" || "${UNFREEZE_MTP}" == "true" ]]; then
            _trainable_params="${_trainable_params} model.talker.code_predictor"
        fi
        if [[ "${UNFREEZE_CODEC_HEAD}" == "1" || "${UNFREEZE_CODEC_HEAD}" == "true" ]]; then
            _trainable_params="${_trainable_params} model.talker.codec_head"
        fi
        _trainable_regex="${_talker_regex}"
        ;;
    *)
        echo "[ERROR] TALKER_TRAINING_SCOPE must be 'layers' or 'whole', got: ${TALKER_TRAINING_SCOPE}" >&2
        exit 1
        ;;
esac
if [[ "${UNFREEZE_THINKER}" == "1" || "${UNFREEZE_THINKER}" == "true" ]]; then
    _trainable_params="${_trainable_params} model.thinker"
fi

_trainable_args=(--trainable_parameters ${_trainable_params})
if [[ -n "${_trainable_regex}" ]]; then
    _trainable_args+=(--trainable_parameters_regex "${_trainable_regex}")
fi

# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------
_resume_args=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    _resume_args+=(--resume_from_checkpoint "${RESUME_FROM}")
    echo "[upstream] RESUME_FROM=${RESUME_FROM}"
fi

# ---------------------------------------------------------------------------
# Standardized run name + output dir  (标准见 docs/EXPERIMENTS.md §0.1)
#   实验名: {method}-{unfreeze}-{model}-lr{lr}-bs{perdev}x{ngpu}-s2s[-{date}]
#   目录名: upstream_<实验名去掉日期后缀>
# 说明: bs 标签只记 per_device × 卡数；有效 batch 还要再乘
#       GRADIENT_ACCUMULATION_STEPS（当前 =${GRADIENT_ACCUMULATION_STEPS}）。
# ---------------------------------------------------------------------------
_ngpu="${NPROC_PER_NODE:-1}"
_exp_tag="${_method_tag}-${_unfreeze_tag}-${MODEL_SIZE_TAG}-lr${LEARNING_RATE}-bs${PER_DEVICE_TRAIN_BATCH_SIZE}x${_ngpu}-${UPSTREAM_TASK_TAG}"
: "${RUN_NAME:=${_exp_tag}-$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME
: "${OUTPUT_DIR:=${OUTPUT_ROOT}/upstream_${_exp_tag}}"
mkdir -p "${OUTPUT_DIR}" "${BRIDGE_ALIGNMENT_CODEC_CACHE}"

# ---------------------------------------------------------------------------
# wandb 额外 config：把环境变量驱动的 ablation 变量写进 wandb.config
# （标准 TrainingArguments/lr/bs/ga 由 HF WandbCallback 自动记录，这里补齐
#  swift CLI 之外、由 env 驱动的 talker 输入方式等字段）。
# 由 custom/wandb_experiment_config.py 在 wandb.init 后合并进 run.config。
# ---------------------------------------------------------------------------
_effective_bs=$(( PER_DEVICE_TRAIN_BATCH_SIZE * _ngpu * GRADIENT_ACCUMULATION_STEPS ))
export WANDB_EXTRA_CONFIG="{\"task\": \"${UPSTREAM_TASK_TAG}\", \"dataset_name\": \"${UPSTREAM_DATASET_NAME}\", \"talker_method\": \"${_method_tag}\", \"full_add_layout\": ${QWEN_OMNISPEAK_FULL_ADD_LAYOUT}, \"talker_training_scope\": \"${TALKER_TRAINING_SCOPE}\", \"unfreeze_talker_layers\": ${UNFREEZE_TALKER_LAYERS}, \"unfreeze_mtp\": ${UNFREEZE_MTP}, \"unfreeze_codec_head\": ${UNFREEZE_CODEC_HEAD}, \"unfreeze_thinker\": ${UNFREEZE_THINKER}, \"speaker_prompt\": ${QWEN_OMNISPEAK_USE_SPEAKER_PROMPT}, \"residual_loss_weight\": ${RESIDUAL_LOSS_WEIGHT}, \"codec_eos_loss_weight\": ${CODEC_EOS_LOSS_WEIGHT}, \"codec_history_corruption_prob\": ${CODEC_HISTORY_CORRUPTION_PROB}, \"model_size\": \"${MODEL_SIZE_TAG}\", \"num_gpus\": ${_ngpu}, \"effective_batch_size\": ${_effective_bs}}"

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
echo "[upstream] TASK=${UPSTREAM_TASK_TAG}  DATASET=${UPSTREAM_DATASET_NAME}"
echo "[upstream s2s] METHOD=${_method_tag}  FULL_ADD_LAYOUT=${QWEN_OMNISPEAK_FULL_ADD_LAYOUT}"
echo "[upstream s2s] UNFREEZE_TALKER_LAYERS=${UNFREEZE_TALKER_LAYERS}"
echo "[upstream s2s] UNFREEZE_MTP=${UNFREEZE_MTP} (code_predictor trainable=$([[ "${UNFREEZE_MTP}" == "1" || "${UNFREEZE_MTP}" == "true" ]] && echo yes || echo no))"
echo "[upstream s2s] UNFREEZE_CODEC_HEAD=${UNFREEZE_CODEC_HEAD} (codec_head trainable=$([[ "${UNFREEZE_CODEC_HEAD}" == "1" || "${UNFREEZE_CODEC_HEAD}" == "true" ]] && echo yes || echo no))"
echo "[upstream s2s] TALKER_TRAINING_SCOPE=${TALKER_TRAINING_SCOPE}  UNFREEZE_THINKER=${UNFREEZE_THINKER}"
echo "[upstream s2s] RESIDUAL_LOSS=${RESIDUAL_LOSS} weight=${RESIDUAL_LOSS_WEIGHT}"
echo "[upstream s2s] CODEC_EOS_WEIGHT=${CODEC_EOS_LOSS_WEIGHT}  HISTORY_CORRUPTION=${CODEC_HISTORY_CORRUPTION_PROB}"
echo "[upstream s2s] trainable_params=${_trainable_params}  regex=${_trainable_regex:-<none>}"
echo "[upstream s2s] MODEL=${MODEL_PATH}"
echo "[upstream s2s] DATA=${BRIDGE_ALIGNMENT_DATA}"
echo "[upstream s2s] WDS mode=${WDS_AUDIO_MODE} root=${WDS_AUDIO_ROOT:-<unset>} strict=${WDS_AUDIO_STRICT}"
echo "[upstream s2s] SPEAKER=${QWEN_OMNISPEAK_USE_SPEAKER_PROMPT}"
echo "[upstream s2s] LR=${LEARNING_RATE}  BS=${PER_DEVICE_TRAIN_BATCH_SIZE}x${_ngpu}(x ga${GRADIENT_ACCUMULATION_STEPS})=eff${_effective_bs}  EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[upstream s2s] GROUP_BY_LENGTH=${GROUP_BY_LENGTH}"
echo "[upstream s2s] WANDB project=${WANDB_PROJECT}  group=${WANDB_RUN_GROUP:-<none>}"
echo "[upstream s2s] WANDB_EXTRA_CONFIG=${WANDB_EXTRA_CONFIG}"
echo "[upstream s2s] RUN_NAME=${RUN_NAME}"
echo "[upstream s2s] OUTPUT=${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Unified training log (规范见 docs/EXPERIMENTS.md §2.2 日志与 checkpoint 管理)
#   logs/<BATCH_TAG>/<RUN_NAME>.log  —— 每条 run 一个 stdout 日志，按批次归档。
#   BATCH_TAG 用批次号（如 batch4.2），未指定则 misc。用 tee 同时落盘 + 打印，
#   这样无论是否用 nohup 外部重定向，日志都会规范落到 logs/ 下。
# ---------------------------------------------------------------------------
: "${BATCH_TAG:=misc}"
: "${DISABLE_INNER_TEE:=false}"
_LOG_DIR="${REPO_ROOT}/logs/${BATCH_TAG}"
_LOG_FILE="${_LOG_DIR}/${RUN_NAME}.log"
if [[ "${DISABLE_INNER_TEE}" != "true" ]]; then
    mkdir -p "${_LOG_DIR}"
    echo "[upstream s2s] LOG=${_LOG_FILE}"
else
    echo "[upstream s2s] LOG=managed by outer resource monitor"
fi
echo "============================================================"

_swift_command=(
    swift sft
    --external_plugins "${PLUGIN_PATH}"
    --model "${MODEL_PATH}"
    --model_type "${MODEL_TYPE}"
    --template "${TEMPLATE}"
    --dataset "${UPSTREAM_DATASET_NAME}"
    --load_from_cache_file true
    --group_by_length "${GROUP_BY_LENGTH}"
    --remove_unused_columns false
    "${_ray_args[@]}"
    --tuner_type full
    --freeze_parameters_ratio 1.0
    "${_trainable_args[@]}"
    --torch_dtype "${TORCH_DTYPE}"
    --attn_impl "${ATTN_IMPL}"
    --deepspeed "${DEEPSPEED}"
    --gradient_checkpointing true
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_ratio "${WARMUP_RATIO}"
    --max_length "${MAX_LENGTH}"
    --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
    --eval_steps "${EVAL_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --logging_steps "${LOGGING_STEPS}"
    --report_to "${REPORT_TO:-wandb}"
    --run_name "${RUN_NAME}"
    "${_dataloader_args[@]}"
    --dataset_num_proc "${DATASET_NUM_PROC}"
    --output_dir "${OUTPUT_DIR}"
    "${_resume_args[@]}"
    "$@"
)
if [[ "${DISABLE_INNER_TEE}" == "true" ]]; then
    "${_swift_command[@]}"
else
    "${_swift_command[@]}" 2>&1 | tee -a "${_LOG_FILE}"
fi
