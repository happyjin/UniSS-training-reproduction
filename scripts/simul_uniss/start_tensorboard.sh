#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_v1/bootstrap_15shard.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

mkdir -p "${TENSORBOARD_DIR}" "${LOG_DIR}"
if tmux has-session -t "${TENSORBOARD_SESSION}" 2>/dev/null; then
  echo "TensorBoard session already exists: ${TENSORBOARD_SESSION}"
  exit 0
fi

command="env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy tensorboard --logdir $(printf '%q' "${TENSORBOARD_DIR}") --host 0.0.0.0 --port $(printf '%q' "${TENSORBOARD_PORT}") 2>&1 | tee -a $(printf '%q' "${LOG_DIR}/tensorboard.log")"
tmux new-session -d -s "${TENSORBOARD_SESSION}" "bash -lc $(printf '%q' "${command}")"
echo "TensorBoard started in tmux session ${TENSORBOARD_SESSION} on port ${TENSORBOARD_PORT}"
