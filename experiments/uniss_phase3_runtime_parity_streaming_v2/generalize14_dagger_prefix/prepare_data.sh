#!/usr/bin/env bash
set -euo pipefail

V14_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${V14_DIR}/config_canary.env}"
# shellcheck source=/dev/null
source "${CONFIG}"

source_iteration="$(printf 'iter_%07d' "${V13_BASE_ITERATION}")"
source_checkpoint="${V13_CHECKPOINT_ROOT}/${source_iteration}"
[[ -f "${source_checkpoint}/.metadata" ]] || {
  echo "Missing Generalize13 base checkpoint: ${source_checkpoint}" >&2
  exit 1
}

mkdir -p "${PHASE3_NATIVE_CHECKPOINT}"
pinned_checkpoint="${PHASE3_NATIVE_CHECKPOINT}/${source_iteration}"
if [[ -L "${pinned_checkpoint}" ]]; then
  [[ "$(readlink -f "${pinned_checkpoint}")" == "$(readlink -f "${source_checkpoint}")" ]] || {
    echo "Pinned Generalize13 checkpoint points to a different source" >&2
    exit 1
  }
elif [[ -e "${pinned_checkpoint}" ]]; then
  echo "Refusing to replace non-symlink pinned checkpoint: ${pinned_checkpoint}" >&2
  exit 1
else
  ln -s "${source_checkpoint}" "${pinned_checkpoint}"
fi
printf '%s\n' "${V13_BASE_ITERATION}" > "${PHASE3_NATIVE_CHECKPOINT}/latest_checkpointed_iteration.txt"

exec "${PYTHON}" -m experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_training_manifest \
  --trajectory-packed "${TRAJECTORY_PACKED}" \
  --replay-packed "${PHASE3_REPLAY_PACKED}" \
  --replay-offsets "${PHASE3_REPLAY_OFFSETS}" \
  --output-root "${TRAINING_ROOT}" \
  --coverage-epochs "${COVERAGE_EPOCHS}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --data-parallel-size 8 \
  --replay-fraction "${REPLAY_FRACTION}" \
  --seed "${SEED}"
