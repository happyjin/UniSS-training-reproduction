#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/runtime_logs"
RESTART_DELAY="${UNISS_STUDENT_V2_RESTART_DELAY_SECONDS:-30}"
mkdir -p "${LOG_DIR}"
while true; do
  rm -f "${SCRIPT_DIR}/public_url.txt" "${SCRIPT_DIR}/access_info.json"
  echo "[$(date -Is)] starting Student-v2 streaming stereo demo" | tee -a "${LOG_DIR}/watchdog.log"
  "${SCRIPT_DIR}/run_public.sh" 2>&1 | tee -a "${LOG_DIR}/public_server.log"
  exit_code="${PIPESTATUS[0]}"
  echo "[$(date -Is)] server exited code=${exit_code}; retry in ${RESTART_DELAY}s" \
    | tee -a "${LOG_DIR}/watchdog.log"
  sleep "${RESTART_DELAY}"
done
