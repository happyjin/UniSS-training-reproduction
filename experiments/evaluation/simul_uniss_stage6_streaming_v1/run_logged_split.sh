#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != "dev" && "$1" != "test" ) ]]; then
  echo "Usage: $0 dev|test RUN_ID" >&2
  exit 2
fi
SPLIT="$1"
RUN_ID="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/experiment.env"
LOG_DIR="${REPO_ROOT}/logs/evaluation/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/${RUN_ID}.log") 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting Stage6 ${SPLIT}: ${RUN_ID}"
"${SCRIPT_DIR}/run_full_split_4gpu.sh" "${SPLIT}" "${RUN_ID}"
