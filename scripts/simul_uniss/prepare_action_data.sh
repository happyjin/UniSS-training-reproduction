#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
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

mask_cmd=(python -m training.simul_uniss.mask_action_samples
  --input "${SAMPLES_JSONL}"
  --output "${ACTION_SAMPLES_JSONL}"
)
pack_cmd=(python -m training.simul_uniss.pack_sequences
  --input "${ACTION_SAMPLES_JSONL}"
  --output "${ACTION_PACKED_TRAIN}"
  --seq-length "${SEQ_LENGTH}"
  --drop-overlong
)
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${mask_cmd[@]}"; printf '\n'
  printf '%q ' "${pack_cmd[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$(dirname "${ACTION_SAMPLES_JSONL}")" "$(dirname "${ACTION_PACKED_TRAIN}")" "${LOG_DIR}"
{
  "${mask_cmd[@]}"
  "${pack_cmd[@]}"
} 2>&1 | tee -a "${LOG_DIR}/prepare_action_data.log"
