#!/usr/bin/env bash
set -euo pipefail

V13_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${V13_DIR}/config_canary.env"

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
