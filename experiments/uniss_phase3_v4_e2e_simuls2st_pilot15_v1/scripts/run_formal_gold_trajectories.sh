#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

if [[ "${DATA_RUN_ID}" == "gold_smoke_v1" ]]; then
  echo "formal build requires an explicit versioned DATA_RUN_ID" >&2
  exit 2
fi
mkdir -p \
  "${PROCESSED_ROOT}/manifests" \
  "${PROCESSED_ROOT}/source_events" \
  "${REPORT_ROOT}" \
  "${LOG_ROOT}/gold_trajectories" \
  "${TMPDIR}"
export PYTHONPATH=${REPO_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

FINGERPRINTS=${PROCESSED_ROOT}/manifests/checkpoint_fingerprints.json
FROZEN_SPLIT=${PROCESSED_ROOT}/manifests/frozen_split.json
TRAIN_OUTPUT=${PROCESSED_ROOT}/source_events/train_gold_trajectories.jsonl
VALID_OUTPUT=${PROCESSED_ROOT}/source_events/valid_gold_trajectories.jsonl
TRAIN_AUDIT=${REPORT_ROOT}/train_gold_trajectory_audit.json
VALID_AUDIT=${REPORT_ROOT}/valid_gold_trajectory_audit.json
GATE=${REPORT_ROOT}/GOLD_TRAJECTORY_GATE.json

for path in "${FINGERPRINTS}" "${FROZEN_SPLIT}" "${TRAIN_OUTPUT}" "${VALID_OUTPUT}" "${TRAIN_AUDIT}" "${VALID_AUDIT}" "${GATE}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite formal asset: ${path}" >&2
    exit 2
  fi
done

CPU_WORKERS=${FINGERPRINT_WORKERS:-16} "${SCRIPT_DIR}/fingerprint_checkpoints.sh" \
  > "${LOG_ROOT}/gold_trajectories/fingerprint.log" 2>&1

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.freeze_split \
  --source-snapshot "${STAGE_A_SOURCE_SNAPSHOT}" \
  --stage-a-audit "${STAGE_A_DATA_AUDIT}" \
  --checkpoint-fingerprints "${FINGERPRINTS}" \
  --output "${FROZEN_SPLIT}" \
  > "${LOG_ROOT}/gold_trajectories/freeze_split.log" 2>&1

V1_SHA=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"]["v1"]["sha256"])' "${FINGERPRINTS}")
PHASE3_SHA=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"]["phase3"]["sha256"])' "${FINGERPRINTS}")

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.build_gold_trajectories \
  --manifest "${STAGE_A_SOURCE_TRAIN}" \
  --output "${TRAIN_OUTPUT}" \
  --split train \
  --workers "${TRAIN_WORKERS:-56}" \
  --hash-audio \
  --audit-audio \
  --v1-checkpoint-sha256 "${V1_SHA}" \
  --phase3-teacher-sha256 "${PHASE3_SHA}" \
  > "${LOG_ROOT}/gold_trajectories/train_build.log" 2>&1 &
train_pid=$!

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.build_gold_trajectories \
  --manifest "${STAGE_A_SOURCE_VALID}" \
  --output "${VALID_OUTPUT}" \
  --split valid \
  --workers "${VALID_WORKERS:-8}" \
  --hash-audio \
  --audit-audio \
  --v1-checkpoint-sha256 "${V1_SHA}" \
  --phase3-teacher-sha256 "${PHASE3_SHA}" \
  > "${LOG_ROOT}/gold_trajectories/valid_build.log" 2>&1 &
valid_pid=$!

status=0
wait "${train_pid}" || status=$?
wait "${valid_pid}" || status=$?
if (( status != 0 )); then
  echo "formal trajectory build failed; inspect ${LOG_ROOT}/gold_trajectories" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.audit_trajectories \
  --input "${TRAIN_OUTPUT}" \
  --output "${TRAIN_AUDIT}" \
  --require-audio-hash \
  --require-audio-audit \
  > "${LOG_ROOT}/gold_trajectories/train_audit.log" 2>&1 &
train_audit_pid=$!

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.audit_trajectories \
  --input "${VALID_OUTPUT}" \
  --output "${VALID_AUDIT}" \
  --require-audio-hash \
  --require-audio-audit \
  > "${LOG_ROOT}/gold_trajectories/valid_audit.log" 2>&1 &
valid_audit_pid=$!

status=0
wait "${train_audit_pid}" || status=$?
wait "${valid_audit_pid}" || status=$?
if (( status != 0 )); then
  echo "formal trajectory audit failed; inspect ${LOG_ROOT}/gold_trajectories" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.finalize_gold_gate \
  --frozen-split "${FROZEN_SPLIT}" \
  --train-build "${TRAIN_OUTPUT}.build.json" \
  --valid-build "${VALID_OUTPUT}.build.json" \
  --train-audit "${TRAIN_AUDIT}" \
  --valid-audit "${VALID_AUDIT}" \
  --output "${GATE}" \
  > "${LOG_ROOT}/gold_trajectories/finalize.log" 2>&1

echo "gate=${GATE}"
echo "train=${TRAIN_OUTPUT}"
echo "valid=${VALID_OUTPUT}"
