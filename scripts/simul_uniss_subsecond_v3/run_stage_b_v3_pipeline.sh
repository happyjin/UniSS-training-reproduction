#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_B_V3_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"

mkdir -p "${V3_LOG_ROOT}"
exec > >(tee -a "${V3_LOG_ROOT}/pipeline.log") 2>&1

echo "[$(date -Is)] Stage-B-v3 pipeline started"
if [[ ! -s "${V3_MIXED_TRAIN_MANIFEST}.complete.json" || \
      ! -s "${V3_MIXED_VALID_MANIFEST}.complete.json" ]]; then
  bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v3/prepare_stage_b_v3_data.sh"
else
  echo "[$(date -Is)] Reusing complete Stage-B-v3 data manifests"
fi

bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v3/start_tensorboard.sh"

if [[ ! -s "${V3_CHECKPOINT_ROOT}/TRAINING_COMPLETE.json" ]]; then
  if [[ -d "${V3_CHECKPOINT_ROOT}" ]] && \
      find "${V3_CHECKPOINT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite an incomplete formal checkpoint directory:" >&2
    echo "${V3_CHECKPOINT_ROOT}" >&2
    exit 1
  fi
  bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v3/train_stage_b_v3.sh"
else
  echo "[$(date -Is)] Reusing completed Stage-B-v3 training"
fi

if [[ ! -s "${V3_CHECKPOINT_ROOT}/JOINT_SELECTION.json" ]]; then
  bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v3/select_joint_checkpoint.sh"
else
  echo "[$(date -Is)] Reusing completed joint selection"
fi
echo "[$(date -Is)] Stage-B-v3 pipeline complete"
