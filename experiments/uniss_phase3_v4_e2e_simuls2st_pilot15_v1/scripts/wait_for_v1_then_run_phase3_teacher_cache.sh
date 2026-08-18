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

if (( WAIT_INTERVAL_SECONDS < 1 )); then
  echo "WAIT_INTERVAL_SECONDS must be positive" >&2
  exit 2
fi

while [[ ! -f "${TRAIN_AUDIT}" || ! -f "${VALID_AUDIT}" ]]; do
  if ! tmux has-session -t "${ROLLOUT_SESSION}" 2>/dev/null; then
    echo "V1 rollout session ended before both audits were produced" >&2
    echo "train_audit=${TRAIN_AUDIT}" >&2
    echo "valid_audit=${VALID_AUDIT}" >&2
    exit 3
  fi
  printf '%s waiting for V1 train/valid audits\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep "${WAIT_INTERVAL_SECONDS}"
done

"${PYTHON_BIN}" - <<'PY' "${TRAIN_AUDIT}" "${VALID_AUDIT}"
import json
import pathlib
import sys

for value in sys.argv[1:]:
    path = pathlib.Path(value)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise SystemExit(f"V1 rollout audit did not pass: {path}")
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
