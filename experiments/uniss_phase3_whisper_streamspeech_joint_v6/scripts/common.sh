#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_ROOT}/experiment.env"

export PATH="${USER_ROOT}/conda_envs/uniss-train/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/cache/pip}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${USER_ROOT}/cache/torch_kernel}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" "${PYTORCH_KERNEL_CACHE_PATH}" "${TMPDIR}" \
  "${REPO_ROOT}/logs/uniss_phase3_whisper_streamspeech_joint_v6"

SITE_PACKAGES="$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_LIBRARY_DIRS=()
shopt -s nullglob
for directory in "${SITE_PACKAGES}"/nvidia/*/lib; do
  [[ -d "${directory}" ]] && NVIDIA_LIBRARY_DIRS+=("${directory}")
done
shopt -u nullglob
if (( ${#NVIDIA_LIBRARY_DIRS[@]} > 0 )); then
  NVIDIA_LIBRARY_PATH="$(IFS=:; echo "${NVIDIA_LIBRARY_DIRS[*]}")"
  export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }; }
refuse_existing() {
  for path in "$@"; do
    [[ ! -e "${path}" ]] || { echo "Refusing to overwrite: ${path}" >&2; exit 1; }
  done
}
