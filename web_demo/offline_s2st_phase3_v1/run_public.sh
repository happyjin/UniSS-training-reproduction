#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/offline_s2st_phase3_v1"
LOG_DIR="${SCRIPT_DIR}/runtime_logs"
PORT="${UNISS_DEMO_PORT:-7861}"

[[ -x "${DEMO_ENV}/bin/python" ]] || { echo "Missing demo environment: ${DEMO_ENV}" >&2; exit 1; }
[[ -n "${UNISS_DEMO_AUTH_PASSWORD:-}" ]] || { echo "UNISS_DEMO_AUTH_PASSWORD is required" >&2; exit 1; }
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/public_server.log") 2>&1
cd "${REPO_ROOT}"
exec "${DEMO_ENV}/bin/python" -m web_demo.offline_s2st_phase3_v1.app_gradio \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --share \
  --auth-user "${UNISS_DEMO_AUTH_USER:-uniss}" \
  --auth-password "${UNISS_DEMO_AUTH_PASSWORD}"
