#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/uniss_phase3_prefix_streaming_v3_stereo_v1"
PORT="${UNISS_PREFIX_STREAMING_PORT:-7865}"
KERNEL_CACHE="${UNISS_PREFIX_STREAMING_KERNEL_CACHE:-${USER_ROOT}/torch_kernel_cache}"
FFMPEG_BIN_DIR="${UNISS_PREFIX_STREAMING_FFMPEG_BIN_DIR:-${REPO_ROOT}/web_demo/streaming_s2st_r2_v1/bin}"

[[ -x "${DEMO_ENV}/bin/python" ]] || { echo "Missing environment: ${DEMO_ENV}" >&2; exit 1; }
[[ -x "${FFMPEG_BIN_DIR}/ffmpeg" && -x "${FFMPEG_BIN_DIR}/ffprobe" ]] || {
  echo "Missing repository-local ffmpeg/ffprobe wrappers: ${FFMPEG_BIN_DIR}" >&2
  exit 1
}
mkdir -p "${SCRIPT_DIR}/runtime_logs" "${KERNEL_CACHE}"
export PATH="${FFMPEG_BIN_DIR}:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_KERNEL_CACHE_PATH="${KERNEL_CACHE}"
cd "${REPO_ROOT}"
exec "${DEMO_ENV}/bin/python" -m web_demo.uniss_phase3_prefix_streaming_v3_stereo_v1.app_gradio \
  --host 0.0.0.0 --port "${PORT}" --share
