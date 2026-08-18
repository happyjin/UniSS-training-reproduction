#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

mkdir -p "${PROCESSED_ROOT}/source_events" "${REPORT_ROOT}" "${TMPDIR}"
export PYTHONPATH=${REPO_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

FINGERPRINTS=${PROCESSED_ROOT}/manifests/checkpoint_fingerprints.json
if [[ ! -f "${FINGERPRINTS}" ]]; then
  CPU_WORKERS=${CPU_WORKERS:-8} "${SCRIPT_DIR}/fingerprint_checkpoints.sh"
fi
V1_SHA=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"]["v1"]["sha256"])' "${FINGERPRINTS}")
PHASE3_SHA=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"]["phase3"]["sha256"])' "${FINGERPRINTS}")
OUTPUT=${PROCESSED_ROOT}/source_events/train_gold_trajectories.jsonl
AUDIT=${REPORT_ROOT}/train_gold_trajectory_audit.json

if [[ -e "${OUTPUT}" || -e "${AUDIT}" ]]; then
  echo "refusing to overwrite smoke data; select a new DATA_RUN_ID" >&2
  exit 2
fi

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.build_gold_trajectories \
  --manifest "${STAGE_A_SOURCE_TRAIN}" \
  --output "${OUTPUT}" \
  --split train \
  --workers "${CPU_WORKERS:-8}" \
  --limit "${SMOKE_RECORDS:-32}" \
  --hash-audio \
  --v1-checkpoint-sha256 "${V1_SHA}" \
  --phase3-teacher-sha256 "${PHASE3_SHA}"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.audit_trajectories \
  --input "${OUTPUT}" \
  --output "${AUDIT}" \
  --require-audio-hash

echo "trajectory=${OUTPUT}"
echo "audit=${AUDIT}"
