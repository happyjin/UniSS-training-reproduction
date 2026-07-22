#!/usr/bin/env bash
set -euo pipefail

DECODER="synthetic"
RECORD_INDEX=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --decoder) DECODER="$2"; shift 2 ;;
    --record-index) RECORD_INDEX="$2"; shift 2 ;;
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

mkdir -p "${STAGE5_OUTPUT_DIR}" "${STAGE5_TENSORBOARD_DIR}" "${LOG_DIR}"
cmd=(python -m training.simul_uniss.replay_streaming
  --input "${SCHEDULES_JSONL}"
  --output-wav "${STAGE5_OUTPUT_DIR}/record_${RECORD_INDEX}_${DECODER}.wav"
  --metrics "${STAGE5_OUTPUT_DIR}/record_${RECORD_INDEX}_${DECODER}.json"
  --tensorboard-dir "${STAGE5_TENSORBOARD_DIR}"
  --record-index "${RECORD_INDEX}"
  --decoder "${DECODER}"
)
if [[ "${DECODER}" == "bicodec" ]]; then
  cmd+=(--bicodec-model-dir "${BICODEC_MODEL_DIR}" --device "${BICODEC_DEVICE}")
fi
"${cmd[@]}" 2>&1 | tee -a "${LOG_DIR}/stage5_streaming_replay.log"
