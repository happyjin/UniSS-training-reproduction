#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
ENV_ROOT="${ENV_ROOT:-${USER_ROOT}/conda_envs/uniss-train}"

export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/cache/pip}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-Bridge/src:${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${REPO_ROOT}/checkpoints"

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/convert_uniss_checkpoint.sh {import|export} [args...]" >&2
  exit 2
fi

DIRECTION="$1"
shift

"${ENV_ROOT}/bin/python" "${REPO_ROOT}/training/convert_uniss_checkpoint.py" "${DIRECTION}" "$@"
