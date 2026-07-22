#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
LIMIT_RECORDS=""
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --limit-records) LIMIT_RECORDS="$2"; shift 2 ;;
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

mkdir -p "${PROCESSED_DIR}" "${PACKED_DIR}" "${RUN_DIR}" "${TENSORBOARD_DIR}" "${LOG_DIR}"

shards=()
for ((index=SHARD_START; index<SHARD_START+SHARD_COUNT; index++)); do
  printf -v shard '%s/train-%05d.parquet' "${UNIST_ROOT}" "${index}"
  if [[ ! -f "${shard}" ]]; then
    echo "Missing shard: ${shard}" >&2
    exit 1
  fi
  shards+=("${shard}")
done

prepare_cmd=(python -m training.simul_uniss.prepare_data
  --input "${shards[@]}"
  --output-dir "${PROCESSED_DIR}"
  --tokenizer "${TOKENIZER_DIR}"
  --chunk-ms "${CHUNK_MS}"
  --wait-k-chunks "${WAIT_K_CHUNKS}"
  --max-phrase-tokens "${MAX_PHRASE_TOKENS}"
)
if [[ -n "${LIMIT_RECORDS}" ]]; then
  prepare_cmd+=(--limit-records "${LIMIT_RECORDS}" --skip-sha256)
fi

pack_cmd=(python -m training.simul_uniss.pack_sequences
  --input "${SAMPLES_JSONL}"
  --output "${PACKED_TRAIN}"
  --seq-length "${SEQ_LENGTH}"
  --drop-overlong
)

eval_cmd=(python -m training.simul_uniss.stage0_eval
  --input "${SCHEDULES_JSONL}"
  --output "${STAGE0_METRICS}"
  --tensorboard-dir "${TENSORBOARD_DIR}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${prepare_cmd[@]}"; printf '\n'
  printf '%q ' "${pack_cmd[@]}"; printf '\n'
  printf '%q ' "${eval_cmd[@]}"; printf '\n'
  exit 0
fi

{
  echo "[$(date -u +%FT%TZ)] preparing ${SHARD_COUNT} Simul-UniSS shards"
  "${prepare_cmd[@]}"
  "${pack_cmd[@]}"
  "${eval_cmd[@]}"
  echo "[$(date -u +%FT%TZ)] bootstrap preparation complete"
} 2>&1 | tee -a "${LOG_DIR}/prepare_bootstrap_15shard.log"
