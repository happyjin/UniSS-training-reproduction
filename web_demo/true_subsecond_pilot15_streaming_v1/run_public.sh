#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/true_subsecond_pilot15_streaming_v1"
PORT="${UNISS_TRUE_STREAMING_PORT:-7868}"
GPU_CACHE="${UNISS_TRUE_STREAMING_CACHE:-${USER_ROOT}/cache/uniss_true_streaming_demo}"
FFMPEG_BIN_DIR="${REPO_ROOT}/web_demo/streaming_s2st_r2_v1/bin"

[[ -x "${DEMO_ENV}/bin/python" ]] || { echo "Missing reused Phase3 demo environment: ${DEMO_ENV}" >&2; exit 1; }
[[ -x "${FFMPEG_BIN_DIR}/ffmpeg" && -x "${FFMPEG_BIN_DIR}/ffprobe" ]] || {
  echo "Missing repository-local ffmpeg wrappers: ${FFMPEG_BIN_DIR}" >&2
  exit 1
}
mkdir -p "${SCRIPT_DIR}/runtime_logs" "${GPU_CACHE}/gradio" "${GPU_CACHE}/torch"
export PATH="${FFMPEG_BIN_DIR}:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${GPU_CACHE}/huggingface"
export XDG_CACHE_HOME="${GPU_CACHE}/xdg"
export GRADIO_TEMP_DIR="${GPU_CACHE}/gradio"
export PYTORCH_KERNEL_CACHE_PATH="${GPU_CACHE}/torch"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export NO_PROXY="127.0.0.1,localhost"
cd "${REPO_ROOT}"
exec "${DEMO_ENV}/bin/python" -m web_demo.true_subsecond_pilot15_streaming_v1.app_gradio \
  --host 0.0.0.0 --port "${PORT}" --share
