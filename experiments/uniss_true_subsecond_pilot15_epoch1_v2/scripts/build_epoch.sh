#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"
REPLAY_ROOT="${REPO_ROOT}/data/megatron/uniss_true_subsecond_pilot15_v1"

"${PYTHON}" -m experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.build_epoch \
  --trajectory-packed "${PACKED_ROOT}/packed_trajectory.jsonl" \
  --trajectory-offsets "${PACKED_ROOT}/packed_trajectory.offsets.u64" \
  --replay-packed "${REPLAY_ROOT}/packed_replay.jsonl" \
  --replay-offsets "${REPLAY_ROOT}/packed_replay.offsets.u64" \
  --audit "${REPORT_ROOT}/data_audit_v2.json" \
  --output-root "${PACKED_ROOT}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --data-parallel-microbatch "$(( 8 * MICRO_BATCH_SIZE ))" \
  --seed "${SEED}" | tee "${LOG_ROOT}/epoch_manifest.log"
