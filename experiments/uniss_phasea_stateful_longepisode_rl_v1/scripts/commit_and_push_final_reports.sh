#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
TRAIN_REPORT=${REPO_ROOT}/reports/uniss_phasea_stateful_longepisode_rl_v1/rollouts/formal_train64_g4_v1/REPORT.zh-CN.md
FINAL_REPORT_ROOT=${REPO_ROOT}/reports/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1
FINAL_REPORT=${FINAL_REPORT_ROOT}/REPORT.zh-CN.md

for path in "${TRAIN_REPORT}" "${FINAL_REPORT}"; do
  [[ -f "${path}" ]] || { echo "missing final report artifact: ${path}" >&2; exit 2; }
done

cd "${REPO_ROOT}"
git add \
  reports/uniss_phasea_stateful_longepisode_rl_v1/rollouts/formal_train64_g4_v1/REPORT.zh-CN.md \
  reports/uniss_phasea_stateful_longepisode_rl_v1/final_comparison_v1
git diff --cached --check
if ! git diff --cached --quiet; then
  git commit -m "report final long-episode RL comparison"
fi
git push private master:main
echo "FINAL_REPORT_COMMITTED=${FINAL_REPORT}"
