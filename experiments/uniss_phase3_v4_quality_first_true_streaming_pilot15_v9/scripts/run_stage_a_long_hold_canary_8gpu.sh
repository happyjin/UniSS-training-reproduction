#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Compatibility alias for the name used by the first committed V9 draft.
exec bash "${SCRIPT_DIR}/run_stage_a_bridge_freeze_canary_8gpu.sh" "$@"
