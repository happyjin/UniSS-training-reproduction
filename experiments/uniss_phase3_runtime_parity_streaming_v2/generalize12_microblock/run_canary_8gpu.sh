#!/usr/bin/env bash
set -euo pipefail
V12_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${V12_DIR}/config_canary.env"
exec bash "${V12_DIR}/run_8gpu.sh" "$@"
