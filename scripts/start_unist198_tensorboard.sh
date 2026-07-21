#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
CONFIG_FILE=""
HOST="${TENSORBOARD_HOST:-0.0.0.0}"
PORT="${TENSORBOARD_PORT:-6006}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/uniss_qwen0p5b_unist198_full_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

cmd=(tensorboard --logdir "${TENSORBOARD_ROOT}" --host "${HOST}" --port "${PORT}")
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -f "${ACTIVATE_SCRIPT}" ]] || { echo "Missing activation script: ${ACTIVATE_SCRIPT}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
mkdir -p "${TENSORBOARD_ROOT}"
exec "${cmd[@]}"
