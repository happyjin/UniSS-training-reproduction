#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config.env"
cd "${REPO_ROOT}"

"${PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.build_multifile_manifest \
  --parts-root data/processed/uniss_phase3_dense_aligned_streaming_pilot15_v1/pack_parts \
  --output "${TRAJECTORY_MANIFEST}" --split train --expected-parts 32
"${PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.build_multifile_manifest \
  --parts-root data/processed/uniss_phase3_dense_aligned_streaming_pilot15_v1/valid_pack_parts \
  --output "${VALID_TRAJECTORY_MANIFEST}" --split valid --expected-parts 4
"${PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.build_training_manifest \
  --trajectory-manifest "${TRAJECTORY_MANIFEST}" \
  --valid-trajectory-manifest "${VALID_TRAJECTORY_MANIFEST}" \
  --replay-packed "${PHASE3_REPLAY_PACKED}" \
  --replay-offsets "${PHASE3_REPLAY_OFFSETS}" \
  --data-audit "${DATA_AUDIT}" --output-root "${TRAINING_ROOT}" \
  --coverage-epochs "${COVERAGE_EPOCHS}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" --data-parallel-size 8 \
  --replay-fraction "${REPLAY_FRACTION}" --seed "${SEED}" "$@"

