#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
PORT="${UNISS_E2_PORT:-7863}"
DEVICE="${UNISS_E2_DEVICE:-cuda:0}"
SCRIPT_DIR="${REPO_ROOT}/web_demo/subsecond_e2_v1"

[[ -x "${DEMO_ENV}/bin/python" ]] || { echo "Missing demo environment: ${DEMO_ENV}" >&2; exit 1; }
mkdir -p "${SCRIPT_DIR}/runtime_logs"
cd "${REPO_ROOT}"
exec "${DEMO_ENV}/bin/python" -m web_demo.subsecond_e2_v1.app_gradio \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --share
