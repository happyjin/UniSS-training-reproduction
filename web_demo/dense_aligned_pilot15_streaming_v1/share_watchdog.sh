#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELAY="${UNISS_DENSE_STREAMING_RESTART_DELAY_SECONDS:-30}"
mkdir -p "${SCRIPT_DIR}/runtime_logs"
while true; do
  rm -f "${SCRIPT_DIR}/public_url.txt" "${SCRIPT_DIR}/access_info.json"
  echo "[$(date -Is)] starting dense-aligned streaming demo" | tee -a "${SCRIPT_DIR}/runtime_logs/watchdog.log"
  "${SCRIPT_DIR}/run_public.sh" 2>&1 | tee -a "${SCRIPT_DIR}/runtime_logs/public_server.log"
  code="${PIPESTATUS[0]}"
  echo "[$(date -Is)] server exited code=${code}; retry in ${DELAY}s" | tee -a "${SCRIPT_DIR}/runtime_logs/watchdog.log"
  sleep "${DELAY}"
done

