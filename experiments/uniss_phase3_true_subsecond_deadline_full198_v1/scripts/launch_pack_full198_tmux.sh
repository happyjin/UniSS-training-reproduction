#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env"

SESSION="${PACK_TMUX_SESSION:-uniss_true_subsecond_pack_full198}"
PACK_LOG="${PACK_LOG:-${REPO_ROOT}/logs/uniss_true_subsecond_pack_full198.log}"
PACK_WORKERS="${PACK_WORKERS:-8}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already running: ${SESSION}"
  exit 0
fi
mkdir -p "${PACKED_ROOT}/parts" "$(dirname "${PACK_LOG}")"
command=(
  "${PYTHON}" -m
  experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.pack_completed_shards
  --cache-root "${CACHE_ROOT}"
  --raw-root "${RAW_UNIST_DIR}"
  --parts-root "${PACKED_ROOT}/parts"
  --shard-count 198
  --seq-length "${SEQ_LENGTH}"
  --workers "${PACK_WORKERS}"
  --poll-seconds 30
)
printf -v quoted '%q ' "${command[@]}"
tmux new-session -d -s "${SESSION}" \
  "cd $(printf '%q' "${REPO_ROOT}") && ${quoted} >> $(printf '%q' "${PACK_LOG}") 2>&1"
echo "started ${SESSION}; log=${PACK_LOG}; workers=${PACK_WORKERS}"
