#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_C_V3_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v3/stage_c_after_v3_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

mkdir -p "${STAGE_C_V3_LOG_DIR}" "${STAGE_C_V3_REPORT_DIR}"
exec > >(tee -a "${STAGE_C_V3_LOG_DIR}/pipeline.log") 2>&1
echo "[$(date -Is)] Stage-C-after-v3 pipeline started"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v3/start_stage_c_after_v3_tensorboard.sh"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v3/train_stage_c_after_v3.sh" formal
python -m training.simul_uniss.subsecond_v2.validate_stage_c \
  --calibration "${STAGE_C_V3_OUTPUT_DIR}/calibration.json" \
  --output "${STAGE_C_V3_REPORT_DIR}/STAGE_C_QUALITY_GATE.json"
echo "[$(date -Is)] Stage-C-after-v3 pipeline complete"
