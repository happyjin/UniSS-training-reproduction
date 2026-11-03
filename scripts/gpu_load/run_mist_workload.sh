#!/usr/bin/env bash
set -euo pipefail
LOAD_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOAD_PYTHON="${TRAIN_LOAD_PYTHON:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python}"
exec "${LOAD_PYTHON}" -B -u "${LOAD_REPO_ROOT}/scripts/gpu_load/train_mist_workload.py" "$@"
