#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cache_8gpu.sh"
bash "${SCRIPT_DIR}/run_pack_15_cpu.sh"
bash "${SCRIPT_DIR}/build_epoch.sh"
bash "${SCRIPT_DIR}/run_train_8gpu.sh"
