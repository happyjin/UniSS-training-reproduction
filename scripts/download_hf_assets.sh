#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

TARGET="${1:-uniss}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
ENV_ROOT="${ENV_ROOT:-${USER_ROOT}/conda_envs/uniss-train}"
HF_CLI="${HF_CLI:-${ENV_ROOT}/bin/huggingface-cli}"

export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${TMPDIR}" "${REPO_ROOT}/pretrained_models" "${REPO_ROOT}/data/raw"

if [[ ! -x "${HF_CLI}" ]]; then
  HF_CLI="$(command -v huggingface-cli)"
fi

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

download_model() {
  local repo_id="$1"
  local local_dir="$2"
  run_cmd "${HF_CLI}" download "${repo_id}" \
    --repo-type model \
    --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
    --local-dir "${local_dir}" \
    --max-workers "${HF_MAX_WORKERS:-4}"
}

download_dataset() {
  local repo_id="$1"
  local local_dir="$2"
  run_cmd "${HF_CLI}" download "${repo_id}" \
    --repo-type dataset \
    --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
    --local-dir "${local_dir}" \
    --max-workers "${HF_MAX_WORKERS:-4}"
}

case "${TARGET}" in
  uniss)
    download_model "cmots/UniSS" "${REPO_ROOT}/pretrained_models/UniSS"
    ;;
  qwen)
    download_model "Qwen/Qwen2.5-1.5B-Instruct" "${REPO_ROOT}/pretrained_models/Qwen2.5-1.5B-Instruct"
    ;;
  qwen0p5b)
    download_model "Qwen/Qwen2.5-0.5B-Instruct" "${REPO_ROOT}/pretrained_models/Qwen2.5-0.5B-Instruct"
    ;;
  unist)
    download_dataset "cmots/UniST" "${REPO_ROOT}/data/raw/UniST"
    ;;
  all)
    download_model "cmots/UniSS" "${REPO_ROOT}/pretrained_models/UniSS"
    download_model "Qwen/Qwen2.5-1.5B-Instruct" "${REPO_ROOT}/pretrained_models/Qwen2.5-1.5B-Instruct"
    download_dataset "cmots/UniST" "${REPO_ROOT}/data/raw/UniST"
    ;;
  *)
    echo "Usage: $0 [--dry-run] {uniss|qwen|qwen0p5b|unist|all}" >&2
    exit 2
    ;;
esac
