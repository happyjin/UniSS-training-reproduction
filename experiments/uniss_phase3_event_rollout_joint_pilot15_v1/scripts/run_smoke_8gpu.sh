#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config.env"
cd "${REPO_ROOT}"

SMOKE_NAME="${SMOKE_NAME:-uniss_phase3_event_rollout_joint_pilot15_smoke8_v1}"
SMOKE_ROOT="${REPO_ROOT}/data/megatron/uniss_phase3_event_rollout_joint_pilot15_v1/smoke8_v1"
SMOKE_TRAIN="${SMOKE_ROOT}/train_parts_manifest.json"
SMOKE_VALID="${SMOKE_ROOT}/valid_parts_manifest.json"
SMOKE_TRAINING="${SMOKE_ROOT}/training"
SMOKE_AUDIT="${REPO_ROOT}/reports/uniss_phase3_event_rollout_joint_pilot15_v1/data_audit_sampled.json"

mkdir -p "${SMOKE_ROOT}"
"${PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.build_multifile_manifest \
  --parts-root data/processed/uniss_phase3_dense_aligned_streaming_pilot15_v1/pack_parts \
  --output "${SMOKE_TRAIN}" --split train --expected-parts 32 --records-per-part 1 >/dev/null
"${PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.build_multifile_manifest \
  --parts-root data/processed/uniss_phase3_dense_aligned_streaming_pilot15_v1/valid_pack_parts \
  --output "${SMOKE_VALID}" --split valid --expected-parts 4 --records-per-part 1 >/dev/null
"${PYTHON}" -m experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.build_training_manifest \
  --trajectory-manifest "${SMOKE_TRAIN}" --valid-trajectory-manifest "${SMOKE_VALID}" \
  --replay-packed "${PHASE3_REPLAY_PACKED}" --replay-offsets "${PHASE3_REPLAY_OFFSETS}" \
  --data-audit "${SMOKE_AUDIT}" --output-root "${SMOKE_TRAINING}" \
  --coverage-epochs 3 --micro-batch-size 2 --global-batch-size 128 \
  --data-parallel-size 8 --replay-fraction 0.35 --seed "${SEED}" \
  --allow-sampled-audit >/dev/null

export EXPERIMENT_NAME="${SMOKE_NAME}"
export TRAJECTORY_MANIFEST="${SMOKE_TRAIN}"
export VALID_TRAJECTORY_MANIFEST="${SMOKE_VALID}"
export DATA_AUDIT="${SMOKE_AUDIT}"
export TRAINING_ROOT="${SMOKE_TRAINING}"
export TRAINING_MANIFEST="${SMOKE_TRAINING}/training_manifest.json"
export REPLAY_SUBSET_OFFSETS="${SMOKE_TRAINING}/replay_subset.u64"
export SAVE_DIR="${REPO_ROOT}/checkpoints/${SMOKE_NAME}"
export RUN_DIR="${REPO_ROOT}/runs/${SMOKE_NAME}"
export TB_DIR="${RUN_DIR}/tensorboard"
export LOG_PATH="${REPO_ROOT}/logs/${SMOKE_NAME}.log"
export REPORT_ROOT="${REPO_ROOT}/reports/${SMOKE_NAME}"
export COVERAGE_EPOCHS=3
export SAVE_INTERVAL=1 EVAL_INTERVAL=1 EVAL_ITERS=1 LOG_INTERVAL=1
export RUN_SMOKE=1 ALLOW_SAMPLED_AUDIT=1 EVENT_ROLLOUT_FORCE_ROLLIN=1
export RUN_EXIT_INTERVAL="${SMOKE_EXIT_INTERVAL:-}"

arguments=()
[[ "${DRY_RUN}" == 1 ]] && arguments+=(--dry-run)
exec bash "${HERE}/scripts/run_megatron_8gpu.sh" "${arguments[@]}" "$@"

