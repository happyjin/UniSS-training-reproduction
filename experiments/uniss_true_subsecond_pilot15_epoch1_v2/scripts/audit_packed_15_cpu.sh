#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"

mkdir -p "${REPORT_ROOT}" "${LOG_ROOT}"
OMP_NUM_THREADS=1 "${PYTHON}" -m \
  experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.audit_packed \
  --root "${DATA_ROOT}/trajectory_packs" \
  --output "${REPORT_ROOT}/packed_causal_parity_v1.json" \
  --workers 15 | tee "${LOG_ROOT}/packed_causal_parity_v1.log"
