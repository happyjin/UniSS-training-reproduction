#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"
SMOKE="${1:-}"
if [[ -n "${SMOKE}" && "${SMOKE}" != "--smoke" ]]; then
  echo "Usage: $0 [--smoke]" >&2
  exit 2
fi

if [[ -z "${SMOKE}" && ! -f "${DEV_COMPARE_ROOT}/COMPLETE" ]]; then
  echo "Refusing to start test before the frozen four-way dev comparison is complete" >&2
  exit 1
fi

labels=(r0_e3_v1_bias r1_rebalanced_coverage r2_explicit_latency r3_bilingual_adaptive)
sessions=(stage7a_reward_v2_test_r0 stage7a_reward_v2_test_r1 stage7a_reward_v2_test_r2 stage7a_reward_v2_test_r3)
for index in "${!labels[@]}"; do
  label="${labels[${index}]}"
  session="${sessions[${index}]}"
  marker="${TEST_EVAL_ROOT}/${label}/${FULL_RUN_ID}/COMPLETE"
  if [[ -f "${marker}" ]]; then
    echo "already complete: ${label}"
  elif tmux has-session -t "${session}" 2>/dev/null; then
    echo "already running: ${session}"
  else
    tmux new-session -d -s "${session}" \
      "cd '${REPO_ROOT}' && exec '${SCRIPT_DIR}/run_one_2gpu.sh' '${label}' '${SMOKE}'"
    echo "started ${session}: ${label}"
  fi
done

if [[ "${SMOKE}" == "--smoke" ]]; then
  echo "smoke runs launched; no full-test comparison watcher started"
  exit 0
fi

compare_session=stage7a_reward_v2_test_compare
if [[ -f "${TEST_EVAL_ROOT}/COMPLETE" ]]; then
  echo "test comparison already complete"
elif tmux has-session -t "${compare_session}" 2>/dev/null; then
  echo "already running: ${compare_session}"
else
  tmux new-session -d -s "${compare_session}" \
    "cd '${REPO_ROOT}' && exec '${SCRIPT_DIR}/wait_and_compare.sh'"
  echo "started ${compare_session}"
fi
