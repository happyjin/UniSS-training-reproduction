#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"

export UPSTREAM_DATASET_NAME="bridge_alignment_text_to_speech_translation"
export UPSTREAM_TASK_TAG="t2s_translation"
export UPSTREAM_DEFAULT_SPEAKER_PROMPT="false"
export QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT=true
# Text-only input has no reference speaker.  Force this off even if the parent
# shell happens to carry a speaker-prompt setting from an S2S experiment.
export QWEN_OMNISPEAK_USE_SPEAKER_PROMPT=false

if [[ -z "${BRIDGE_ALIGNMENT_DATA:-}" ]]; then
    WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"
    for candidate in \
        "${SOURCE_ROOT}/datasets/parquet/text_to_speech_translation.parquet" \
        "${SOURCE_ROOT}/datasets/jsonl/text_to_speech_translation.jsonl" \
        "${WORKSPACE_ROOT}/data/sft_t2a_translate_converted/qwen3_tts_codec/text_to_speech_translation.parquet" \
        "${WORKSPACE_ROOT}/data/sft_t2a_translate_converted/raw_audio/text_to_speech_translation.jsonl"; do
        if [[ -s "${candidate}" ]]; then
            export UPSTREAM_DEFAULT_DATA="${candidate}"
            break
        fi
    done
fi

if [[ -z "${BRIDGE_ALIGNMENT_DATA:-}" && -z "${UPSTREAM_DEFAULT_DATA:-}" ]]; then
    echo "[ERROR] text-to-speech translation data not found; set BRIDGE_ALIGNMENT_DATA." >&2
    exit 1
fi

exec bash "${SCRIPT_DIR}/train_s2s_upstream.sh" "$@"
