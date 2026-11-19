#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bridge_concat variant for text-to-speech (t2s) translation.
# Uses qwen2_5_3b_omnispeak_a + pre-encoded t2s parquet.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export SOURCE_ROOT

# ---- Model (variant A, same as upstream_talker t2s) ----
: "${MODEL_TYPE:=qwen2_5_3b_omnispeak_a}"
: "${MODEL_PATH:=${SOURCE_ROOT}/models/qwen2_5_3b_omnispeak_a}"

# ---- Data: converted t2s parquet ----
WORKSPACE_ROOT="$(cd "${SOURCE_ROOT}/../.." && pwd)"
T2S_PARQUET="${WORKSPACE_ROOT}/data/sft_t2a_translate_converted/qwen3_tts_codec/text_to_speech_translation.parquet"
if [[ -z "${BRIDGE_ALIGNMENT_DATA:-}" ]]; then
    if [[ -f "${T2S_PARQUET}" ]]; then
        export BRIDGE_ALIGNMENT_DATA="${T2S_PARQUET}"
    else
        echo "[ERROR] t2s parquet not found: ${T2S_PARQUET}" >&2
        exit 1
    fi
fi
echo "[t2s_bridge] BRIDGE_ALIGNMENT_DATA=${BRIDGE_ALIGNMENT_DATA}"

# ---- t2s: no speaker prompt ----
export QWEN_OMNISPEAK_USE_SPEAKER_PROMPT="${QWEN_OMNISPEAK_USE_SPEAKER_PROMPT:-false}"

# ---- Training scope ----
: "${TALKER_TRAINING_SCOPE:=layers}"
: "${UNFREEZE_TALKER_LAYERS:=28}"
: "${BRIDGE_EXTRA_TRAINABLE:=model.talker.code_predictor model.talker.codec_head}"

# ---- 冻结配置（归属本任务脚本：bridge_concat 变体）----
# bridge + 额外可训模块 + talker 层解冻（按 UNFREEZE_TALKER_LAYERS 用 case 区分）；
# TALKER_TRAINING_SCOPE=whole 时改为整 talker 解冻（与 zeyang_dev 语义一致）。
_trainable_params="model.bridge ${BRIDGE_EXTRA_TRAINABLE}"
if [[ "${TALKER_TRAINING_SCOPE}" == "whole" ]]; then
    _trainable_params="${_trainable_params} model.talker"
    _trainable_regex=""
else
    case "${UNFREEZE_TALKER_LAYERS}" in
        0)  _trainable_regex="" ;;
        2)  _trainable_regex='model\.talker\.model\.layers\.[0-1]\.' ;;
        8)  _trainable_regex='model\.talker\.model\.layers\.[0-7]\.' ;;
        28) _trainable_regex='model\.talker\.model\.layers\.[0-9]+\.' ;;
        *)  echo "[ERROR] UNFREEZE_TALKER_LAYERS must be 0/2/8/28, got: ${UNFREEZE_TALKER_LAYERS}" >&2
            exit 1 ;;
    esac
fi
export _trainable_params _trainable_regex

# ---- Run name ----
: "${RUN_NAME:=${MODEL_TYPE}-bridge_concat-t2s_translation-$(date +%Y%m%d_%H%M%S)}"

export MODEL_TYPE MODEL_PATH RUN_NAME \
       TALKER_TRAINING_SCOPE UNFREEZE_TALKER_LAYERS BRIDGE_EXTRA_TRAINABLE

# ---- Ensure conda env binaries are in PATH ----
# conda activate doesn't work reliably inside `env` wrappers with set -e,
# so we directly add the conda env bin to PATH.
_conda_env_bin="/opt/dlami/nvme/zeyangsong/miniconda/envs/${CONDA_ENV:-liveinterpret}/bin"
[[ -d "${_conda_env_bin}" ]] && export PATH="${_conda_env_bin}:${PATH}"
command -v swift >/dev/null 2>&1 || { echo "[ERROR] swift not found in PATH=${PATH}"; exit 1; }

source "${SCRIPT_DIR}/common_args.sh"

run_bridge_alignment \
    bridge_alignment_text_to_speech_translation \
    text_to_speech_translation \
    "$@"
