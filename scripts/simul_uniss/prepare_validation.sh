#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
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

mkdir -p "${VALID_PROCESSED_DIR}" "${VALID_PACKED_DIR}" "${LOG_DIR}"
{
  python -m training.simul_uniss.prepare_data \
    --input "${UNIST_VALID_PARQUET}" \
    --output-dir "${VALID_PROCESSED_DIR}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --chunk-ms "${CHUNK_MS}" \
    --wait-k-chunks "${WAIT_K_CHUNKS}" \
    --max-phrase-tokens "${MAX_PHRASE_TOKENS}"
  python -m training.simul_uniss.mask_action_samples \
    --input "${VALID_SAMPLES_JSONL}" \
    --output "${VALID_ACTION_SAMPLES_JSONL}"
  python -m training.simul_uniss.pack_sequences \
    --input "${VALID_SAMPLES_JSONL}" \
    --output "${VALID_PACKED_INTERLEAVED}" \
    --seq-length "${SEQ_LENGTH}" \
    --drop-overlong
  python -m training.simul_uniss.pack_sequences \
    --input "${VALID_ACTION_SAMPLES_JSONL}" \
    --output "${VALID_PACKED_ACTION}" \
    --seq-length "${SEQ_LENGTH}" \
    --drop-overlong
  python -m training.simul_uniss.latency_metrics \
    --input "${VALID_SCHEDULES_JSONL}" \
    --output "${RUN_DIR}/validation_latency_metrics.json" \
    --tensorboard-dir "${TENSORBOARD_DIR}/validation_latency"
} 2>&1 | tee -a "${LOG_DIR}/prepare_validation.log"
