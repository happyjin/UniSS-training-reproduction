#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
suffix=()
[[ "${DRY_RUN}" == "0" ]] || suffix+=(--dry-run)
"${EXPERIMENT_DIR}/data_preparation/prepare_shards.sh" "${suffix[@]}"
"${EXPERIMENT_DIR}/data_preparation/pack_shards.sh" "${suffix[@]}"
"${EXPERIMENT_DIR}/data_preparation/assemble_full198.sh" "${suffix[@]}"
