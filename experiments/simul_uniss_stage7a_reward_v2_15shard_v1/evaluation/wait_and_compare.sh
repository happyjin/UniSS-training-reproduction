#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/experiment.env"
DEV_ROOT="${EVAL_ROOT}/full_dev_e2e_v1"
for label in r0_e3_v1_bias r1_rebalanced_coverage r2_explicit_latency r3_bilingual_adaptive; do
  while [[ ! -f "${DEV_ROOT}/${label}/COMPLETE" ]]; do sleep 30; done
done
"${EVAL_ENV}/bin/python" "${ROOT}/evaluation/compare.py" \
  --root "${DEV_ROOT}" --output-json "${DEV_ROOT}/comparison.json" \
  --report "${DEV_ROOT}/reward_v2_four_way_full_dev_report.md"
touch "${DEV_ROOT}/COMPLETE"
echo "REPORT=${DEV_ROOT}/reward_v2_four_way_full_dev_report.md"

