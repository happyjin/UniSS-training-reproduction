#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"

for label in r0_e3_v1_bias r1_rebalanced_coverage r2_explicit_latency r3_bilingual_adaptive; do
  marker="${TEST_EVAL_ROOT}/${label}/${FULL_RUN_ID}/COMPLETE"
  while [[ ! -f "${marker}" ]]; do sleep 30; done
done

"${EVAL_ENV}/bin/python" "${SCRIPT_DIR}/compare.py" \
  --root "${TEST_EVAL_ROOT}" --run-id "${FULL_RUN_ID}" \
  --dev-comparison "${DEV_COMPARE_ROOT}/comparison.json" \
  --output-json "${TEST_EVAL_ROOT}/comparison.json" \
  --report "${TEST_EVAL_ROOT}/reward_v2_four_way_full_test_report.md" \
  --continuity-report "${CONTINUITY_REPORT}"
touch "${TEST_EVAL_ROOT}/COMPLETE"
echo "REPORT=${TEST_EVAL_ROOT}/reward_v2_four_way_full_test_report.md"
echo "CONTINUITY_REPORT=${CONTINUITY_REPORT}"

