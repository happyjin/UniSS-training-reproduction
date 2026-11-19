#!/usr/bin/env bash
# =============================================================================
# train_speech_to_speech_translation.sh — 统一的 S2S bridge-alignment 训练入口
#
# 通过 TRAIN_BACKEND 选择后端（deepspeed / megatron），TRAIN_TARGET 选择训练目标
# （adapter / bridge）。两个变量都由 yaml env 注入，默认 deepspeed + adapter。
#
# 本脚本默认使用 indexed WebDataset (WDS) 格式数据。数据路径由 launch_train.sh
# 从 S3 同步到节点 NVMe，WDS 音频通过 /curated_audio 符号链接命中 /nvme/data/audio。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ---------------------------------------------------------------------------
# 后端 + 训练目标
# ---------------------------------------------------------------------------
: "${TRAIN_BACKEND:=deepspeed}"
: "${TRAIN_TARGET:=adapter}"

# ---------------------------------------------------------------------------
# 场景预设（原 zeyang_dev bridge_alignment/run_batch4_*.sh 的场景矩阵内化，
# 全部属于 adapter 家族）。一个任务的多套参数配置用 SCENE + case 区分；
# 预设只设默认值，env 可逐项覆盖；SCENE=custom（默认）不做任何预设。
# ---------------------------------------------------------------------------
: "${SCENE:=custom}"
# eval/save 节奏默认值按 SCENE 区分（2026-08-27）：四方对照场景 batch4_8_adapter_cmp
# 为 1000（对齐对照基准任务 A 的 eval_steps/save_steps）；其余场景 500（2026-08-26 起，
# 原 1000）：全量数据（~18.99M 条、2 epochs、GBS 1024 → ~37k steps）下 1000 步粒度过粗。
_SCENE_EVAL_SAVE_DEFAULT=500
[[ "${SCENE}" == "batch4_8_adapter_cmp" ]] && _SCENE_EVAL_SAVE_DEFAULT=1000
# PER_DEVICE_TRAIN_BATCH_SIZE 默认值按 SCENE 区分（2026-08-28）：四方对照场景
# batch4_8_adapter_cmp 的有效 batch 对齐 A=512（数学结果可比的前提），per_device 按
# DP 卡数派生 = 512 / DP：单机 8 卡 → 64（B）；ray 双节点 16 卡 → 32（D），ga=1。
# DP 卡数：ray 模式取 ray_entrypoint 注入的 RAY_WORLD_SIZE，否则取 launch_train.sh
# 导出的 NPROC_PER_NODE。注意 megatron 后端不读该变量（走 GBS/MBS，不受影响）。
_SCENE_BS_DEFAULT=64
if [[ "${SCENE}" == "batch4_8_adapter_cmp" ]]; then
    _cmp_ngpu="${RAY_WORLD_SIZE:-${NPROC_PER_NODE:-8}}"
    _SCENE_BS_DEFAULT=$(( 512 / _cmp_ngpu ))
fi
case "${SCENE}" in
    batch4_8_adapter|batch4_8_thinker|batch4_8_thinker_cosine|stage2_5_unist|batch4_8_adapter_cmp)
        # 五场景共享的 adapter + whole talker 基座（逐项对齐原 run 脚本）
        : "${TALKER_METHOD:=reply_add}"
        : "${TALKER_TRAINING_SCOPE:=whole}"
        : "${UNFREEZE_TALKER_LAYERS:=0}"
        : "${UNFREEZE_MTP:=0}"
        : "${UNFREEZE_CODEC_HEAD:=0}"
        : "${LEARNING_RATE:=1e-5}"
        : "${MAX_LENGTH:=576}"
        : "${WARMUP_RATIO:=0.03}"
        : "${PER_DEVICE_TRAIN_BATCH_SIZE:=${_SCENE_BS_DEFAULT}}"
        : "${PER_DEVICE_EVAL_BATCH_SIZE:=64}"
        : "${GRADIENT_ACCUMULATION_STEPS:=1}"
        : "${SPLIT_DATASET_RATIO:=0.0005}"
        : "${DEEPSPEED:=zero0}"
        : "${EVAL_STEPS:=${_SCENE_EVAL_SAVE_DEFAULT}}"
        : "${SAVE_STEPS:=${_SCENE_EVAL_SAVE_DEFAULT}}"
        : "${QWEN_OMNISPEAK_RESIDUAL_LOSS:=true}"
        : "${QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT:=0.3}"
        : "${QWEN_OMNISPEAK_CODEC_EOS_LOSS_WEIGHT:=1.0}"
        ;;
    custom) ;;
    *)
        echo "[ERROR] SCENE must be one of custom / batch4_8_adapter / batch4_8_thinker / batch4_8_thinker_cosine / stage2_5_unist / batch4_8_adapter_cmp, got: ${SCENE}" >&2
        exit 1
        ;;
esac
case "${SCENE}" in
    batch4_8_adapter)
        # 原 run_batch4_8_adapter_talker_unist.sh
        : "${UNFREEZE_THINKER:=0}"
        : "${LR_SCHEDULER_TYPE:=constant_with_warmup}"
        : "${NUM_TRAIN_EPOCHS:=2}"
        ;;
    batch4_8_thinker)
        # 原 run_batch4_8_thinker_unist.sh（+thinker，grouped_lr 低 LR 保护）
        : "${UNFREEZE_THINKER:=1}"
        : "${THINKER_LR:=2e-6}"
        : "${OPTIMIZER:=grouped_lr}"
        : "${LR_SCHEDULER_TYPE:=constant_with_warmup}"
        : "${NUM_TRAIN_EPOCHS:=1}"
        ;;
    batch4_8_thinker_cosine)
        # 原 run_batch4_8_thinker_unist_cosine.sh（调度器改 cosine）
        : "${UNFREEZE_THINKER:=1}"
        : "${THINKER_LR:=2e-6}"
        : "${OPTIMIZER:=grouped_lr}"
        : "${LR_SCHEDULER_TYPE:=cosine}"
        : "${NUM_TRAIN_EPOCHS:=1}"
        ;;
    stage2_5_unist)
        # 原 run_batch4_7_stage2_5_unist.sh（thinker 冻结，单 epoch）
        : "${UNFREEZE_THINKER:=0}"
        : "${LR_SCHEDULER_TYPE:=constant_with_warmup}"
        : "${NUM_TRAIN_EPOCHS:=1}"
        ;;
    batch4_8_adapter_cmp)
        # 四方对照预设（2026-08-27：B 单机 deepspeed / C 单机 megatron /
        # D 双节点 ray+deepspeed / E 双节点 megatron，各跑 200 step 后人工停止）。
        # 与 batch4_8_adapter 同基座（adapter + whole talker、thinker 冻结、LR 1e-5、
        # constant_with_warmup、2 epochs、MAX_LENGTH 576），并对齐基准任务 A 的
        # batch 与 eval/save 节奏：deepspeed 侧有效 batch 一律 512（per_device 按
        # DP 卡数派生 = 512/DP：B 单机 64×8×ga1、D 双节点 32×16×ga1，见上方
        # _SCENE_BS_DEFAULT；2026-08-28 修订——原 64×ga1×DP 规则下 D=1024，
        # warmup 日程 2× 快进，数学结果不可与 A/B/C/E 直接比对）、
        # EVAL_STEPS/SAVE_STEPS=1000（上方 _SCENE_EVAL_SAVE_DEFAULT）；
        # megatron 侧 GBS=512/MBS=16（common_args.sh 同名 SCENE 分支）。
        : "${UNFREEZE_THINKER:=0}"
        : "${LR_SCHEDULER_TYPE:=constant_with_warmup}"
        : "${NUM_TRAIN_EPOCHS:=2}"
        ;;
esac
if [[ "${SCENE}" != "custom" ]]; then
    if [[ "${TRAIN_TARGET}" != "adapter" ]]; then
        echo "[ERROR] SCENE=${SCENE} 属于 adapter 家族，与 TRAIN_TARGET=${TRAIN_TARGET} 冲突" >&2
        exit 1
    fi
    echo "[s2s] SCENE=${SCENE} 场景预设生效（各项均可用 env 覆盖）"
fi
# grouped_lr 的 thinker 组学习率由 plugin.py 的 GroupedLrOptimizerCallback 读 env；
# loss 权重由 modeling 读 env（QWEN_OMNISPEAK_ 前缀）
export THINKER_LR OPTIMIZER QWEN_OMNISPEAK_RESIDUAL_LOSS \
       QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT QWEN_OMNISPEAK_CODEC_EOS_LOSS_WEIGHT

# ---------------------------------------------------------------------------
# 任务身份
# ---------------------------------------------------------------------------
: "${MODEL_TYPE:=qwen2_5_3b_omnispeak_a}"
: "${MODEL_PATH:=${NVME_MOUNT:-/}/origin_models/custom_models/${MODEL_TYPE}}"

# ---------------------------------------------------------------------------
# 训练目标：adapter（thinker_adapter）或 bridge（经典 bridge alignment）
# ---------------------------------------------------------------------------
case "${TRAIN_TARGET}" in
    adapter)
        # adapter 模式默认配置：
        #   - TALKER_METHOD=reply_add
        #   - QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT=true
        #   - QWEN_OMNISPEAK_FULL_ADD_LAYOUT=false（reply_add 推导）
        #   - QWEN_OMNISPEAK_USE_SPEAKER_PROMPT=true（adapter 模式默认开启）
        export QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT="${QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT:-true}"
        : "${TALKER_METHOD:=reply_add}"
        case "${TALKER_METHOD}" in
            reply_add) export QWEN_OMNISPEAK_FULL_ADD_LAYOUT=false ;;
            full_add)  export QWEN_OMNISPEAK_FULL_ADD_LAYOUT=true ;;
            *) echo "[ERROR] TALKER_METHOD must be 'reply_add' or 'full_add'" >&2; exit 1 ;;
        esac
        : "${QWEN_OMNISPEAK_USE_SPEAKER_PROMPT:=true}"
        export QWEN_OMNISPEAK_USE_SPEAKER_PROMPT
        export QWEN3_TTS_SPEAKER_ENCODER="${QWEN3_TTS_SPEAKER_ENCODER:-${NVME_MOUNT:-/}/origin_models/Qwen3-TTS-12Hz-1.7B-Base}"
        _target_tag="adapter"
        ;;
    bridge)
        export QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT="${QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT:-false}"
        : "${TALKER_METHOD:=sem_block}"
        _target_tag="bridge"
        ;;
    *)
        echo "[ERROR] TRAIN_TARGET must be 'adapter' or 'bridge', got: ${TRAIN_TARGET}" >&2
        exit 1
        ;;
esac

_method_tag="${TALKER_METHOD}"

# ---------------------------------------------------------------------------
# 训练目标与 forward 结构一致性断言（P1-11）
# TRAIN_TARGET 决定解冻的模块（thinker_adapter / bridge），而
# QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT 决定 codec 前向走 adapter 叠加还是
# bridge 拼接。该 env 可被外部（如任务 yaml）独立覆盖，形成"解冻 A、前向走 B"
# 的冲突组合：被解冻的模块收不到任何梯度，训练静默空转且无任何告警。
# ---------------------------------------------------------------------------
_check_target_forward_consistency() {
    local _up="${QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT:-false}"
    case "${_up,,}" in
        1|true|yes|y|on) _up="true" ;;
        *) _up="false" ;;
    esac
    if [[ "${TRAIN_TARGET}" == "adapter" && "${_up}" != "true" ]]; then
        echo "[ERROR] TRAIN_TARGET=adapter 但 QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT=${QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT:-<unset>}：" \
             "adapter 被解冻而 forward 走 bridge 路径，adapter 收不到梯度。请移除覆盖或对齐 TRAIN_TARGET。" >&2
        exit 1
    fi
    if [[ "${TRAIN_TARGET}" == "bridge" && "${_up}" != "false" ]]; then
        echo "[ERROR] TRAIN_TARGET=bridge 但 QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT=${QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT:-<unset>}：" \
             "bridge 被解冻而 forward 走 adapter 路径，bridge 收不到梯度。请移除覆盖或对齐 TRAIN_TARGET。" >&2
        exit 1
    fi
}
_check_target_forward_consistency

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
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${NVME_MOUNT:-/}/.cache/datasets_${_method_tag}_s2s}"

echo "[s2s] BACKEND=${TRAIN_BACKEND} TARGET=${_target_tag} METHOD=${_method_tag}"
echo "[s2s] DATA=${BRIDGE_ALIGNMENT_DATA:-<unset>}"
echo "[s2s] WDS audio at /curated_audio (symlink -> /nvme/data)"
echo "[s2s] NORMALIZE_PROMPT=${BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT} USE_ICL=${QWEN_OMNISPEAK_USE_ICL}"
echo "[s2s] SPEAKER=${QWEN_OMNISPEAK_USE_SPEAKER_PROMPT:-false}"
echo "[s2s] UPSTREAM_TALKER_INPUT=${QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT:-false}"

# ---------------------------------------------------------------------------
# 可训参数（两后端统一语义：--freeze_parameters_ratio 1.0 全冻结 +
# --trainable_parameters 按名前缀激活，由 TRAIN_TARGET 统一控制 bridge/adapter）。
# 仅参数名前缀不同：deepspeed 在 HF 命名空间 model.*；
# megatron 的 mcore 把 bridge/talker/thinker_adapter 挂在 mm.visual 下 → visual.*。
# ---------------------------------------------------------------------------
if [[ "${TRAIN_BACKEND}" == "deepspeed" ]]; then
    _trainable_prefix="model"
else
    _trainable_prefix="visual"
fi
if [[ "${TRAIN_TARGET}" == "adapter" ]]; then
    _trainable_params="${_trainable_prefix}.thinker_adapter"
else
    _trainable_params="${_trainable_prefix}.bridge"
fi
: "${TALKER_TRAINING_SCOPE:=layers}"
: "${UNFREEZE_TALKER_LAYERS:=0}"
case "${UNFREEZE_TALKER_LAYERS}" in
    0)  _trainable_regex="" ;;
    2)  _trainable_regex="${_trainable_prefix}\.talker\.model\.layers\.[0-1]\." ;;
    8)  _trainable_regex="${_trainable_prefix}\.talker\.model\.layers\.[0-7]\." ;;
    28) _trainable_regex="${_trainable_prefix}\.talker\.model\.layers\.\d+\." ;;
    *) echo "[ERROR] UNFREEZE_TALKER_LAYERS must be 0/2/8/28" >&2; exit 1 ;;
esac
: "${UNFREEZE_THINKER:=0}"
: "${UNFREEZE_MTP:=0}"
: "${UNFREEZE_CODEC_HEAD:=0}"
case "${TALKER_TRAINING_SCOPE}" in
    whole)
        _trainable_params="${_trainable_params} ${_trainable_prefix}.talker"
        _trainable_regex=""
        ;;
    layers)
        [[ "${UNFREEZE_MTP}" == "1" || "${UNFREEZE_MTP}" == "true" ]] && _trainable_params="${_trainable_params} ${_trainable_prefix}.talker.code_predictor"
        [[ "${UNFREEZE_CODEC_HEAD}" == "1" || "${UNFREEZE_CODEC_HEAD}" == "true" ]] && _trainable_params="${_trainable_params} ${_trainable_prefix}.talker.codec_head"
        ;;
    *) echo "[ERROR] TALKER_TRAINING_SCOPE must be 'layers' or 'whole'" >&2; exit 1 ;;
esac
if [[ "${UNFREEZE_THINKER}" == "1" || "${UNFREEZE_THINKER}" == "true" ]]; then
    if [[ "${TRAIN_BACKEND}" == "deepspeed" ]]; then
        _trainable_params="${_trainable_params} model.thinker"
    else
        # megatron 的 thinker 是 mcore GPTModel，命名空间为 language_model（非 visual.*）
        _trainable_params="${_trainable_params} language_model"
    fi
fi
export _trainable_params _trainable_regex

# ---------------------------------------------------------------------------
# 任务输出
# ---------------------------------------------------------------------------
: "${MODEL_SIZE_TAG:=3b}"
_unfreeze_tag="${TALKER_TRAINING_SCOPE:-layers}"
if [[ "${_unfreeze_tag}" == "layers" ]]; then
    case "${UNFREEZE_TALKER_LAYERS:-0}" in
        0)  _unfreeze_tag="adapter" ;;
        28) _unfreeze_tag="full" ;;
        *)  _unfreeze_tag="u${UNFREEZE_TALKER_LAYERS}" ;;
    esac
fi

# ray 模式 NPROC_PER_NODE 保持未设置（use_torchrun 防误判，见 common_args.sh），
# 展示用 GPU 数从 launch_train.sh 派生的 RAY_WORLD_SIZE 取。
_ngpu="${RAY_WORLD_SIZE:-${NPROC_PER_NODE:-1}}"
: "${RUN_NAME:=${_method_tag}-${_target_tag}-${_unfreeze_tag}-${MODEL_SIZE_TAG}-s2s-$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME
: "${OUTPUT_DIR:=${OUTPUT_ROOT:-${NVME_MOUNT:-/}/output}/${_method_tag}-${_target_tag}-${_unfreeze_tag}-${MODEL_SIZE_TAG}-s2s}"

# ---------------------------------------------------------------------------
# 任务超参覆盖（仅覆盖与 common 默认不同的项）
# ---------------------------------------------------------------------------
: "${NUM_TRAIN_EPOCHS:=1}"
# 无前缀默认必须保持 0.3：common_args.sh 会用无前缀变量回写覆盖
# QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT；此处若为 1.0 会把 SCENE 预设的 0.3
# 静默覆盖（2026-08-24 dgi3/23ur 实测因此以 weight=1.0 训练，loss 偏高 ~5.9）。
: "${RESIDUAL_LOSS_WEIGHT:=0.3}"
: "${EVAL_STEPS:=200}"
: "${CODEC_EOS_LOSS_WEIGHT:=1.0}"
: "${CODEC_HISTORY_CORRUPTION_PROB:=0.0}"

# ---------------------------------------------------------------------------
# WANDB_EXTRA_CONFIG（adapter 模式记录完整实验元数据）
# ---------------------------------------------------------------------------
if [[ "${TRAIN_BACKEND}" == "deepspeed" ]]; then
    _effective_bs=$(( ${PER_DEVICE_TRAIN_BATCH_SIZE:-1} * ${_ngpu} * ${GRADIENT_ACCUMULATION_STEPS:-1} ))
    export WANDB_EXTRA_CONFIG="{\"task\": \"s2s\", \"backend\": \"${TRAIN_BACKEND}\", \"target\": \"${_target_tag}\", \"talker_method\": \"${_method_tag}\", \"full_add_layout\": ${QWEN_OMNISPEAK_FULL_ADD_LAYOUT:-false}, \"talker_training_scope\": \"${TALKER_TRAINING_SCOPE:-layers}\", \"unfreeze_talker_layers\": ${UNFREEZE_TALKER_LAYERS:-0}, \"unfreeze_thinker\": ${UNFREEZE_THINKER:-0}, \"speaker_prompt\": ${QWEN_OMNISPEAK_USE_SPEAKER_PROMPT:-false}, \"use_icl\": \"${QWEN_OMNISPEAK_USE_ICL:-false}\", \"residual_loss_weight\": ${RESIDUAL_LOSS_WEIGHT}, \"codec_eos_loss_weight\": ${CODEC_EOS_LOSS_WEIGHT}, \"codec_history_corruption_prob\": ${CODEC_HISTORY_CORRUPTION_PROB}, \"model_size\": \"${MODEL_SIZE_TAG}\", \"num_gpus\": ${_ngpu}, \"effective_batch_size\": ${_effective_bs}}"
fi

# ---------------------------------------------------------------------------
# source 统一的 common_args.sh
# ---------------------------------------------------------------------------
source "${REPO_ROOT}/pipelines/bridge_alignment/common_args.sh"

# common_args 可能派生/覆盖开关，二次断言目标与 forward 结构仍一致（P1-11）。
_check_target_forward_consistency

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
echo "[s2s] BACKEND=${TRAIN_BACKEND} TARGET=${_target_tag}"
echo "[s2s] MODEL=${MODEL_TYPE} @ ${MODEL_PATH}"
if [[ "${TRAIN_BACKEND}" == "deepspeed" ]]; then
    echo "[s2s] DEEPSPEED=${DEEPSPEED}  ATTN_IMPL=${ATTN_IMPL}"
    echo "[s2s] trainable_params=${_trainable_params}  regex=${_trainable_regex:-<none>}"
    echo "[s2s] LR=${LEARNING_RATE}  BS=${PER_DEVICE_TRAIN_BATCH_SIZE}x${_ngpu}x${GRADIENT_ACCUMULATION_STEPS}"
else
    echo "[s2s] SCENE=${SCENE}  TP=${TP} PP=${PP} CP=${CP}"
    echo "[s2s] ATTENTION_BACKEND=${ATTENTION_BACKEND}"
    echo "[s2s] LR=${LR}  MBS=${MICRO_BATCH_SIZE}  GBS=${GLOBAL_BATCH_SIZE}"
fi
echo "[s2s] WANDB project=${WANDB_PROJECT}  RUN_NAME=${RUN_NAME}"
echo "[s2s] OUTPUT=${OUTPUT_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 调用统一分发函数
# ---------------------------------------------------------------------------
run_train bridge_alignment_speech_to_speech_translation speech_to_speech_translation "$@"
