#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"
STAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
TEST_RUN_ID="full_test_${STAMP}"
DEV_RUN_ID="full_dev_${STAMP}"
TEST_SESSION="uniss_stage6_test_${STAMP}"
DEV_SESSION="uniss_stage6_dev_wait_${STAMP}"

if tmux has-session -t "${TEST_SESSION}" 2>/dev/null || tmux has-session -t "${DEV_SESSION}" 2>/dev/null; then
  echo "Refusing to reuse Stage6 tmux session for ${STAMP}" >&2
  exit 1
fi
tmux new-session -d -s "${TEST_SESSION}" \
  "cd '${REPO_ROOT}' && '${SCRIPT_DIR}/run_logged_split.sh' test '${TEST_RUN_ID}'"
tmux new-session -d -s "${DEV_SESSION}" \
  "cd '${REPO_ROOT}' && '${SCRIPT_DIR}/wait_then_run_dev.sh' '${DEV_RUN_ID}'"

echo "TEST_SESSION=${TEST_SESSION}"
echo "TEST_RUN_DIR=${OUTPUT_ROOT}/${TEST_RUN_ID}"
echo "DEV_WAIT_SESSION=${DEV_SESSION}"
echo "DEV_RUN_DIR=${OUTPUT_ROOT}/${DEV_RUN_ID}"
