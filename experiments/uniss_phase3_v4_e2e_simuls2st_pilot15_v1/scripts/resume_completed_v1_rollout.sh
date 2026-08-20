#!/usr/bin/env bash
set -euo pipefail

# Resume only the CPU post-processing of a V1 rollout whose GPU workers have
# already completed. This script never regenerates rollout parts and refuses
# ambiguous partial outputs instead of overwriting them.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${ROLLOUT_RUN_ID:?set the immutable rollout run ID with its split suffix}"
: "${ROLLOUT_SPLIT:?set ROLLOUT_SPLIT to train or valid}"
if [[ "${ROLLOUT_SPLIT}" != "train" && "${ROLLOUT_SPLIT}" != "valid" ]]; then
  echo "ROLLOUT_SPLIT must be train or valid" >&2
  exit 2
fi

QUALITY_AUDIT_WORKERS=${QUALITY_AUDIT_WORKERS:-64}
if (( QUALITY_AUDIT_WORKERS < 1 )); then
  echo "QUALITY_AUDIT_WORKERS must be positive" >&2
  exit 2
fi

ROLLOUT_ROOT=${PROCESSED_ROOT}/v1_rollouts/${ROLLOUT_RUN_ID}
REPORT_DIR=${REPORT_ROOT}/v1_rollouts/${ROLLOUT_RUN_ID}
LOG_DIR=${LOG_ROOT}/v1_rollouts/${ROLLOUT_RUN_ID}
PART_ROOT=${ROLLOUT_ROOT}/parts
WORKER_REPORT_ROOT=${REPORT_DIR}/workers
GOLD=${PROCESSED_ROOT}/source_events/${ROLLOUT_SPLIT}_gold_trajectories.jsonl
MERGED=${ROLLOUT_ROOT}/${ROLLOUT_SPLIT}_v1_rollouts.jsonl
MERGE_REPORT=${REPORT_DIR}/MERGE.json
AUDIT_JSON=${REPORT_DIR}/AUDIT.json
AUDIT_MD=${REPORT_DIR}/AUDIT.md
QUALITY_GATE_JSON=${REPORT_DIR}/QUALITY_GATE.json

for path in "${GOLD}" "${PART_ROOT}" "${WORKER_REPORT_ROOT}"; do
  [[ -e "${path}" ]] || { echo "missing completed-rollout input: ${path}" >&2; exit 3; }
done

mapfile -t worker_reports < <(find "${WORKER_REPORT_ROOT}" -maxdepth 1 -type f -name 'rank*.json' | sort)
if (( ${#worker_reports[@]} == 0 )); then
  echo "no rollout worker reports found in ${WORKER_REPORT_ROOT}" >&2
  exit 3
fi

expected_workers=$(jq -r '.num_workers' "${worker_reports[0]}")
if ! [[ "${expected_workers}" =~ ^[0-9]+$ ]] || (( expected_workers < 1 )); then
  echo "invalid num_workers in ${worker_reports[0]}" >&2
  exit 3
fi
if (( ${#worker_reports[@]} != expected_workers )); then
  echo "worker report count ${#worker_reports[@]} differs from expected ${expected_workers}" >&2
  exit 3
fi

bad_reports=$(jq -r 'select(.status != "complete") | input_filename' "${worker_reports[@]}" | wc -l)
if (( bad_reports != 0 )); then
  echo "${bad_reports} rollout worker reports are not complete" >&2
  exit 3
fi

for ((worker=0; worker<expected_workers; worker++)); do
  report=$(printf '%s/rank%03d.json' "${WORKER_REPORT_ROOT}" "${worker}")
  part=$(printf '%s/rank%03d.jsonl' "${PART_ROOT}" "${worker}")
  [[ -f "${report}" ]] || { echo "missing worker report: ${report}" >&2; exit 3; }
  [[ -f "${part}" ]] || { echo "missing rollout part: ${part}" >&2; exit 3; }
  [[ -f "${part}.offsets.bin" ]] || { echo "missing rollout part index: ${part}.offsets.bin" >&2; exit 3; }
done

mkdir -p "${LOG_DIR}"
export PYTHONPATH=${REPO_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

if [[ ! -e "${MERGED}" && ! -e "${MERGE_REPORT}" ]]; then
  merge_args=()
  for report in "${worker_reports[@]}"; do
    merge_args+=(--part-report "${report}")
  done
  "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.merge_parts \
    "${merge_args[@]}" \
    --output "${MERGED}" \
    --report "${MERGE_REPORT}" \
    > "${LOG_DIR}/merge.log" 2>&1
elif [[ ! -f "${MERGED}" || ! -f "${MERGED}.offsets.bin" || ! -f "${MERGE_REPORT}" ]]; then
  echo "ambiguous partial merge exists; refusing to overwrite ${ROLLOUT_ROOT}" >&2
  exit 4
fi
jq -e '.status == "complete"' "${MERGE_REPORT}" >/dev/null

if [[ ! -e "${AUDIT_JSON}" && ! -e "${AUDIT_MD}" ]]; then
  "${PYTHON_BIN}" -m \
    experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.audit_rollouts \
    --gold "${GOLD}" \
    --rollouts "${MERGED}" \
    --merge-report "${MERGE_REPORT}" \
    --output-json "${AUDIT_JSON}" \
    --output-md "${AUDIT_MD}" \
    > "${LOG_DIR}/audit.log" 2>&1
elif [[ ! -f "${AUDIT_JSON}" || ! -f "${AUDIT_MD}" ]]; then
  echo "ambiguous partial audit exists; refusing to overwrite ${REPORT_DIR}" >&2
  exit 4
fi
jq -e '.status == "passed"' "${AUDIT_JSON}" >/dev/null

DATA_RUN_ID="${DATA_RUN_ID}" \
ROLLOUT_RUN_ID="${ROLLOUT_RUN_ID}" \
ROLLOUT_SPLIT="${ROLLOUT_SPLIT}" \
QUALITY_AUDIT_WORKERS="${QUALITY_AUDIT_WORKERS}" \
  "${SCRIPT_DIR}/run_rollout_quality_gate.sh"
jq -e '.status == "passed"' "${QUALITY_GATE_JSON}" >/dev/null

echo "rollout=${MERGED}"
echo "merge_report=${MERGE_REPORT}"
echo "audit=${AUDIT_JSON}"
echo "quality_gate=${QUALITY_GATE_JSON}"
echo "quality_audit_workers=${QUALITY_AUDIT_WORKERS}"
