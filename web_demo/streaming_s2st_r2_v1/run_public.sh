#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/streaming_s2st_r2_v1"
PORT="${UNISS_STREAMING_PORT:-7862}"
FFMPEG_BIN_DIR="${UNISS_STREAMING_FFMPEG_BIN_DIR:-${SCRIPT_DIR}/bin}"
KERNEL_CACHE="${UNISS_STREAMING_KERNEL_CACHE:-${USER_ROOT}/torch_kernel_cache}"

[[ -x "${DEMO_ENV}/bin/python" ]] || { echo "Missing demo environment: ${DEMO_ENV}" >&2; exit 1; }
[[ -x "${FFMPEG_BIN_DIR}/ffmpeg" && -x "${FFMPEG_BIN_DIR}/ffprobe" ]] || {
  echo "Missing isolated ffmpeg/ffprobe wrappers below ${FFMPEG_BIN_DIR}" >&2
  exit 1
}
mkdir -p "${SCRIPT_DIR}/runtime_logs"
mkdir -p "${KERNEL_CACHE}"
export PATH="${FFMPEG_BIN_DIR}:${PATH}"
export PYTORCH_KERNEL_CACHE_PATH="${KERNEL_CACHE}"
export HOME="${HOME:-${USER_ROOT}}"
cd "${REPO_ROOT}"
exec "${DEMO_ENV}/bin/python" -m web_demo.streaming_s2st_r2_v1.app_gradio \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --share
