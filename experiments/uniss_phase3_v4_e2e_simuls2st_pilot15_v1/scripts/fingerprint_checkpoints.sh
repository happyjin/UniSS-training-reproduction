#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

OUTPUT=${PROCESSED_ROOT}/manifests/checkpoint_fingerprints.json
mkdir -p "$(dirname -- "${OUTPUT}")" "${TMPDIR}"
export PYTHONPATH=${REPO_ROOT}
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "v1=${V1_CHECKPOINT}" \
  --checkpoint "phase3=${PHASE3_CHECKPOINT}" \
  --workers "${CPU_WORKERS:-8}" \
  --output "${OUTPUT}"

echo "fingerprints=${OUTPUT}"
