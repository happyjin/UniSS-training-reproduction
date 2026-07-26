#!/usr/bin/env bash
set -euo pipefail
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "pack_workers=${PACK_WORKERS} shard_start=${SHARD_START} shard_count=${SHARD_COUNT} seq_length=${SEQ_LENGTH}"
  "${EXPERIMENT_DIR}/data_preparation/pack_one_shard.sh" --index "${SHARD_START}" --dry-run
  "${EXPERIMENT_DIR}/data_preparation/pack_one_shard.sh" --index "$((SHARD_START + SHARD_COUNT - 1))" --dry-run
  exit 0
fi
seq "${SHARD_START}" "$((SHARD_START + SHARD_COUNT - 1))" | \
  xargs -r -P "${PACK_WORKERS}" -n 1 "${EXPERIMENT_DIR}/data_preparation/pack_one_shard.sh" --index
