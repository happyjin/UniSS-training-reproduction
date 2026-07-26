#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  [[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
  "${EXPERIMENT_DIR}/data_preparation/pack_shards.sh" --dry-run
  "${EXPERIMENT_DIR}/data_preparation/assemble.sh" --dry-run
  exit 0
fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
"${EXPERIMENT_DIR}/data_preparation/pack_shards.sh"
"${EXPERIMENT_DIR}/data_preparation/assemble.sh"
