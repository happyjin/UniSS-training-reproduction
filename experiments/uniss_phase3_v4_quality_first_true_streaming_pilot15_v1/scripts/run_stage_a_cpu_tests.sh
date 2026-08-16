#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"
mkdir -p "${TMPDIR}" "${LOG_ROOT}/stage_a"

"${PYTHON_BIN}" -m pytest -q \
  "${EXPERIMENT_DIR}/tests" \
  | tee "${LOG_ROOT}/stage_a/cpu_tests.log"

