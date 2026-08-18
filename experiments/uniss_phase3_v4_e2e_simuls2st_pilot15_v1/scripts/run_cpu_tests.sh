#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

export PYTHONPATH=${REPO_ROOT}
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON_BIN}" -m pytest -q "${EXPERIMENT_DIR}/tests"
