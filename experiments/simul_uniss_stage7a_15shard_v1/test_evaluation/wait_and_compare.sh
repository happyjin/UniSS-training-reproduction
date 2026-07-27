#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"
for label in e0_stage6 e1_continued_sft e2_grpo_g4 e3_grpo_g8; do
  marker="${TEST_EVAL_ROOT}/${label}/${FULL_RUN_ID}/COMPLETE"
  while [[ ! -f "${marker}" ]]; do sleep 30; done
done
"${EVAL_ENV}/bin/python" "${SCRIPT_DIR}/compare.py" \
  --root "${TEST_EVAL_ROOT}" --run-id "${FULL_RUN_ID}" \
  --output-json "${TEST_EVAL_ROOT}/comparison.json" \
  --report "${TEST_EVAL_ROOT}/stage7a_four_way_full_test_report.md"
touch "${TEST_EVAL_ROOT}/COMPLETE"
echo "REPORT=${TEST_EVAL_ROOT}/stage7a_four_way_full_test_report.md"
