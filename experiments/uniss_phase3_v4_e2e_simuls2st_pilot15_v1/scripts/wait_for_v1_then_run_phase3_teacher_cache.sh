#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
: "${V1_FORMAL_RUN_ID:?set V1_FORMAL_RUN_ID without the train/valid suffix}"
: "${FORMAL_RUN_ID:?set the immutable Phase3 teacher formal run ID}"

ROLLOUT_SESSION=${ROLLOUT_SESSION:-uniss_e2e_v1_rollout_formal}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-30}
TRAIN_AUDIT=${REPORT_ROOT}/v1_rollouts/${V1_FORMAL_RUN_ID}_train/AUDIT.json
VALID_AUDIT=${REPORT_ROOT}/v1_rollouts/${V1_FORMAL_RUN_ID}_valid/AUDIT.json
TRAIN_QUALITY=${REPORT_ROOT}/v1_rollouts/${V1_FORMAL_RUN_ID}_train/QUALITY_GATE.json
VALID_QUALITY=${REPORT_ROOT}/v1_rollouts/${V1_FORMAL_RUN_ID}_valid/QUALITY_GATE.json

if (( WAIT_INTERVAL_SECONDS < 1 )); then
  echo "WAIT_INTERVAL_SECONDS must be positive" >&2
  exit 2
fi

while tmux has-session -t "${ROLLOUT_SESSION}" 2>/dev/null; do
  printf '%s waiting for V1 train/valid rollout sequence to close\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep "${WAIT_INTERVAL_SECONDS}"
done

for path in "${TRAIN_AUDIT}" "${VALID_AUDIT}"; do
  [[ -f "${path}" ]] || {
    echo "V1 rollout session ended before an audit was produced: ${path}" >&2
    exit 3
  }
done

for split in train valid; do
  DATA_RUN_ID=${DATA_RUN_ID} \
  ROLLOUT_RUN_ID=${V1_FORMAL_RUN_ID}_${split} \
  ROLLOUT_SPLIT=${split} \
  QUALITY_AUDIT_WORKERS=${QUALITY_AUDIT_WORKERS:-64} \
  "${SCRIPT_DIR}/run_rollout_quality_gate.sh"
done

"${PYTHON_BIN}" - <<'PY' "${TRAIN_AUDIT}" "${VALID_AUDIT}" "${TRAIN_QUALITY}" "${VALID_QUALITY}"
import json
import pathlib
import sys

for value in sys.argv[1:]:
    path = pathlib.Path(value)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise SystemExit(f"V1 rollout gate did not pass: {path}")
PY

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
  printf '%s waiting for GPUs to become free after V1 rollout\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep "${WAIT_INTERVAL_SECONDS}"
done

printf '%s starting V1 and Phase3 teacher-cache smoke/formal sequence\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec env \
  DATA_RUN_ID="${DATA_RUN_ID}" \
  V1_FORMAL_RUN_ID="${V1_FORMAL_RUN_ID}" \
  FORMAL_RUN_ID="${FORMAL_RUN_ID}" \
  NUM_GPUS="${NUM_GPUS:-8}" \
  PHASE3_PROCESSES_PER_GPU="${PHASE3_PROCESSES_PER_GPU:-1}" \
  V1_PROCESSES_PER_GPU="${V1_PROCESSES_PER_GPU:-1}" \
  PHASE3_SMOKE_LIMIT="${PHASE3_SMOKE_LIMIT:-256}" \
  V1_SMOKE_LIMIT="${V1_SMOKE_LIMIT:-64}" \
  "${SCRIPT_DIR}/run_all_teacher_caches_formal_sequence.sh"
