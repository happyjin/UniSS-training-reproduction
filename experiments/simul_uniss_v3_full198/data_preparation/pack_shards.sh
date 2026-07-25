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
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/experiment.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "pack_workers=${PACK_WORKERS} shard_start=${SHARD_START} shard_count=${SHARD_COUNT}"
  "${EXPERIMENT_DIR}/data_preparation/pack_one_shard.sh" --config "${CONFIG_FILE}" --index "${SHARD_START}" --dry-run
  "${EXPERIMENT_DIR}/data_preparation/pack_one_shard.sh" --config "${CONFIG_FILE}" --index "$((SHARD_START + SHARD_COUNT - 1))" --dry-run
  exit 0
fi
seq "${SHARD_START}" "$((SHARD_START + SHARD_COUNT - 1))" | \
  xargs -r -P "${PACK_WORKERS}" -n 1 \
  "${EXPERIMENT_DIR}/data_preparation/pack_one_shard.sh" --config "${CONFIG_FILE}" --index
