#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${WORKSPACE_ROOT:=$(cd "${REPO_ROOT}/../.." && pwd)}"
: "${DATA_ROOT:=${WORKSPACE_ROOT}/data}"

: "${SOURCE_VERSION:=common_voice_bidirectional_ms_swift_codec_20260727_v1}"
: "${CLEAN_VERSION:=common_voice_bidirectional_ms_swift_codec_20260728_clean_v1}"
: "${SOURCE_MANIFEST_DIR:=${DATA_ROOT}/manifests/${SOURCE_VERSION}}"
: "${CURATED_DIR:=${DATA_ROOT}/common_voice_s2s_bidirectional_v2/curated}"
: "${OUTPUT_DIR:=${DATA_ROOT}/manifests/${CLEAN_VERSION}}"
: "${QUALITY_INDEX:=${DATA_ROOT}/manifests/_quality_indexes/common_voice_s2s_bidirectional_v2.sqlite}"
: "${WDS_AUDIO_ROOT:=${DATA_ROOT}/speech_bidirectional/data}"
: "${CLEANING_CONFIG:=${REPO_ROOT}/dataset_maker/configs/s2s_cleaning_20260728_v1.json}"
: "${PYTHON_BIN:=${WORKSPACE_ROOT}/miniconda/envs/liveinterpret/bin/python}"
: "${WORKERS:=16}"
: "${AUDIO_TAIL_MODE:=all}"

echo "[s2s-clean] SOURCE=${SOURCE_MANIFEST_DIR}"
echo "[s2s-clean] CURATED=${CURATED_DIR}"
echo "[s2s-clean] OUTPUT=${OUTPUT_DIR}"
echo "[s2s-clean] QUALITY_INDEX=${QUALITY_INDEX}"
echo "[s2s-clean] AUDIO_TAIL_MODE=${AUDIO_TAIL_MODE} WORKERS=${WORKERS}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/dataset_maker/clean_s2s_manifest.py" \
  --manifest-dir "${SOURCE_MANIFEST_DIR}" \
  --curated-dir "${CURATED_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --quality-index "${QUALITY_INDEX}" \
  --config "${CLEANING_CONFIG}" \
  --wds-audio-root "${WDS_AUDIO_ROOT}" \
  --workers "${WORKERS}" \
  --audio-tail-mode "${AUDIO_TAIL_MODE}" \
  "$@"
