#!/usr/bin/env bash
set -euo pipefail

RECORDS=1
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --records) RECORDS="$2"; shift 2 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

mkdir -p "${STAGE0_AUDIO_DIR}" "${STAGE0_PREFIX_DIR}" "${STAGE0_PREFIX_TENSORBOARD_DIR}" "${LOG_DIR}"
python -m training.simul_uniss.reconstruct_unist_audio \
  --input "${UNIST_ROOT}/train-00000.parquet" \
  --output-dir "${STAGE0_AUDIO_DIR}" \
  --bicodec-model-dir "${BICODEC_MODEL_DIR}" \
  --device "${STAGE0_DEVICE}" \
  --limit-records "${RECORDS}" \
  --side both 2>&1 | tee -a "${LOG_DIR}/stage0_reconstruct.log"

for ((index=0; index<RECORDS; index++)); do
  python -m training.simul_uniss.prefix_reencode_baseline \
    --manifest "${STAGE0_AUDIO_DIR}/audio_manifest.jsonl" \
    --glm-tokenizer "${GLM_TOKENIZER_DIR}" \
    --output "${STAGE0_PREFIX_DIR}/record_${index}.json" \
    --tensorboard-dir "${STAGE0_PREFIX_TENSORBOARD_DIR}" \
    --device "${STAGE0_DEVICE}" \
    --record-index "${index}" \
    --chunk-ms "${CHUNK_MS}" 2>&1 | tee -a "${LOG_DIR}/stage0_prefix.log"
done
