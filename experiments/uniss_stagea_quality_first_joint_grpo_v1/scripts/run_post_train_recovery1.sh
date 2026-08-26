#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../config.env"

POST_ID=${POST_ID:-formal_complete_v1}
OUTPUT_ROOT=${EVAL_ROOT}/${POST_ID}
FINAL_REPORT_ROOT=${REPORT_ROOT}/${POST_ID}
RECOVERY_LOG=${LOG_ROOT}/${POST_ID}.post_train_recovery1.log
BEST_ARM=$(tr -d '[:space:]' < "${FINAL_REPORT_ROOT}/BEST_ARM.txt")
[[ "${BEST_ARM}" == a3_g8_full_recovery1 ]] || {
  echo "unexpected quality-first arm: ${BEST_ARM}" >&2
  exit 3
}

ROUTED_ROOT=${OUTPUT_ROOT}/routed64_e2e16
SHORT_ROOT=${OUTPUT_ROOT}/short_audio_multichunk
PREFIX_ROOT=${OUTPUT_ROOT}/long_audio4_prefix60_multichunk
LONG_ROOT=${OUTPUT_ROOT}/bounded_longform_chunk640_recovery1
BEST_LONG=${LONG_ROOT}/${BEST_ARM}
STAGE_A_LONG=${LONG_ROOT}/stage_a_iter381
FINAL_REPORT=${FINAL_REPORT_ROOT}/REPORT.zh-CN.md

for path in \
  "${FINAL_REPORT_ROOT}/TRAINING_AUDIT.json" \
  "${FINAL_REPORT_ROOT}/COMPARISON.json" \
  "${CHECKPOINT_ROOT}/${BEST_ARM}/iter_0002510/.metadata" \
  "${EXPERIMENT_ROOT}/scripts/run_bounded_longform_4gpu.sh" \
  "${EXPERIMENT_ROOT}/evaluation/write_report.py"; do
  [[ -f "${path}" ]] || { echo "missing prerequisite: ${path}" >&2; exit 3; }
done
for arm in \
  a1_sft_full_recovery1 a2_g4_full_recovery1 \
  a3_g8_full_recovery1 a4_g8_seed2_full_recovery1; do
  [[ -f "${ROUTED_ROOT}/${arm}/SUMMARY.json" ]] || exit 3
  [[ -f "${SHORT_ROOT}/${arm}/SUMMARY.json" ]] || exit 3
  [[ -f "${PREFIX_ROOT}/${arm}/SUMMARY.json" ]] || exit 3
done
[[ ! -e "${LONG_ROOT}" ]] || {
  echo "refusing to overwrite recovery output: ${LONG_ROOT}" >&2
  exit 4
}
[[ ! -e "${FINAL_REPORT}" ]] || {
  echo "refusing to overwrite final report: ${FINAL_REPORT}" >&2
  exit 4
}

mkdir -p "${LONG_ROOT}" "${FINAL_REPORT_ROOT}" "${USER_ROOT}/tmp"
exec 9>"${USER_ROOT}/tmp/${EXPERIMENT_NAME}_${POST_ID}_recovery1.lock"
flock -n 9 || { echo "post-train recovery is already running" >&2; exit 4; }
exec > >(tee -a "${RECOVERY_LOG}") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting bounded long-form recovery"
"${EXPERIMENT_ROOT}/scripts/run_bounded_longform_4gpu.sh" \
  "${BEST_ARM}" "${CHECKPOINT_ROOT}/${BEST_ARM}/iter_0002510" \
  "${BEST_LONG}" 0 1 2 3 640 & p0=$!
"${EXPERIMENT_ROOT}/scripts/run_bounded_longform_4gpu.sh" \
  stage_a_iter381 NONE "${STAGE_A_LONG}" 4 5 6 7 640 & p1=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
[[ "${status}" -eq 0 ]] || {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] bounded recovery failed"
  exit 1
}

"${PYTHON_BIN}" "${EXPERIMENT_ROOT}/evaluation/write_report.py" \
  --evaluation-root "${ROUTED_ROOT}" \
  --training-audit "${FINAL_REPORT_ROOT}/TRAINING_AUDIT.json" \
  --comparison "${FINAL_REPORT_ROOT}/COMPARISON.json" \
  --short-root "${SHORT_ROOT}" \
  --long-prefix-root "${PREFIX_ROOT}" \
  --best-longform "${BEST_LONG}" \
  --stage-a-longform "${STAGE_A_LONG}" \
  --output "${FINAL_REPORT}"

echo complete > "${FINAL_REPORT_ROOT}/PIPELINE_COMPLETE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] recovery complete: ${FINAL_REPORT}"
