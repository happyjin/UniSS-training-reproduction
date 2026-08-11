#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/config.env"

TARGET="${TARGET_ITERATION:-100}"
TRACKER="${SAVE_DIR}/latest_checkpointed_iteration.txt"
while true; do
  current=-1
  [[ -s "${TRACKER}" ]] && current="$(tr -d '[:space:]' < "${TRACKER}")"
  if (( current == TARGET )); then
    break
  fi
  if (( current > TARGET )); then
    echo "Checkpoint ${current} passed target ${TARGET}" >&2
    exit 1
  fi
  if ! pgrep -f "pretrain_dense_aligned_megatron.py.*${EXPERIMENT_NAME}/overfit1" >/dev/null; then
    echo "Training exited before target checkpoint: current=${current}, target=${TARGET}" >&2
    exit 1
  fi
  printf '%s waiting for overfit checkpoint: %s/%s\n' \
    "$(date -u +%FT%TZ)" "${current}" "${TARGET}"
  sleep 15
done

ITERATION="${TARGET}" CUDA_VISIBLE_DEVICES=0 \
  bash "${REPO_ROOT}/experiments/uniss_phase3_runtime_parity_streaming_v2/scripts/evaluate_overfit1.sh"
