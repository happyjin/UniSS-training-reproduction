#!/usr/bin/env bash
set -euo pipefail

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
ROLLOUT_ROOT=${PROCESSED_ROOT}/v1_rollouts/${ROLLOUT_RUN_ID}
REPORT_DIR=${REPORT_ROOT}/v1_rollouts/${ROLLOUT_RUN_ID}
LOG_DIR=${LOG_ROOT}/v1_rollouts/${ROLLOUT_RUN_ID}
GOLD=${PROCESSED_ROOT}/source_events/${ROLLOUT_SPLIT}_gold_trajectories.jsonl
ROLLOUT=${ROLLOUT_ROOT}/${ROLLOUT_SPLIT}_v1_rollouts.jsonl
MERGE_REPORT=${REPORT_DIR}/MERGE.json
AUDIT=${REPORT_DIR}/AUDIT.json
STRATA=${ROLLOUT_ROOT}/${ROLLOUT_SPLIT}_quality_strata.jsonl
PARTS=${ROLLOUT_ROOT}/quality_strata_parts
QUALITY_JSON=${REPORT_DIR}/QUALITY_GATE.json
QUALITY_MD=${REPORT_DIR}/QUALITY_GATE.md

if [[ -f "${QUALITY_JSON}" ]]; then
  jq -e '.status == "passed"' "${QUALITY_JSON}" >/dev/null || {
    echo "existing rollout quality gate did not pass: ${QUALITY_JSON}" >&2
    exit 4
  }
  echo "quality_gate=${QUALITY_JSON}"
  exit 0
fi
for path in "${GOLD}" "${ROLLOUT}" "${ROLLOUT}.offsets.bin" "${MERGE_REPORT}" "${AUDIT}"; do
  [[ -f "${path}" ]] || { echo "missing rollout quality input: ${path}" >&2; exit 3; }
done
jq -e '.status == "passed"' "${AUDIT}" >/dev/null || {
  echo "base rollout audit did not pass: ${AUDIT}" >&2
  exit 3
}
mkdir -p "${LOG_DIR}"
export PYTHONPATH=${REPO_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.stratify_rollouts \
  --gold "${GOLD}" \
  --rollouts "${ROLLOUT}" \
  --merge-report "${MERGE_REPORT}" \
  --output-manifest "${STRATA}" \
  --output-json "${QUALITY_JSON}" \
  --output-md "${QUALITY_MD}" \
  --parts-root "${PARTS}" \
  --workers "${QUALITY_AUDIT_WORKERS}" \
  --english-clean-wer "${ENGLISH_CLEAN_WER:-0.30}" \
  --chinese-clean-cer "${CHINESE_CLEAN_CER:-0.20}" \
  --maximum-quarantine-rate "${MAXIMUM_QUARANTINE_RATE:-0.40}" \
  --minimum-accepted-rate "${MINIMUM_ACCEPTED_RATE:-0.60}" \
  --minimum-final-eos-rate "${MINIMUM_FINAL_EOS_RATE:-0.99}" \
  > "${LOG_DIR}/quality_gate.log" 2>&1

echo "strata_manifest=${STRATA}"
echo "quality_gate=${QUALITY_JSON}"
echo "quality_report=${QUALITY_MD}"

