#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/web_demo/stage_b_v2_streaming_stereo_v1"
PORT="${UNISS_STUDENT_V2_DEMO_PORT:-7864}"
FFMPEG_BIN_DIR="${UNISS_STUDENT_V2_FFMPEG_BIN_DIR:-${REPO_ROOT}/web_demo/streaming_s2st_r2_v1/bin}"
KERNEL_CACHE="${UNISS_STUDENT_V2_KERNEL_CACHE:-${USER_ROOT}/torch_kernel_cache}"
CACHE_ROOT="${UNISS_STUDENT_V2_CACHE_ROOT:-${USER_ROOT}/.cache}"
TMP_ROOT="${UNISS_STUDENT_V2_TMP_ROOT:-${USER_ROOT}/tmp}"

[[ -x "${DEMO_ENV}/bin/python" ]] || { echo "Missing demo environment: ${DEMO_ENV}" >&2; exit 1; }
[[ -x "${FFMPEG_BIN_DIR}/ffmpeg" && -x "${FFMPEG_BIN_DIR}/ffprobe" ]] || {
  echo "Missing isolated ffmpeg/ffprobe wrappers: ${FFMPEG_BIN_DIR}" >&2
  exit 1
}
mkdir -p \
  "${SCRIPT_DIR}/runtime_logs" \
  "${KERNEL_CACHE}" \
  "${CACHE_ROOT}/huggingface" \
  "${CACHE_ROOT}/torch" \
  "${CACHE_ROOT}/gradio" \
  "${TMP_ROOT}"
export PATH="${FFMPEG_BIN_DIR}:${PATH}"
export PYTORCH_KERNEL_CACHE_PATH="${KERNEL_CACHE}"
export HOME="${USER_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export GRADIO_TEMP_DIR="${CACHE_ROOT}/gradio"
export TMPDIR="${TMP_ROOT}"
cd "${REPO_ROOT}"
exec "${DEMO_ENV}/bin/python" -m web_demo.stage_b_v2_streaming_stereo_v1.app_gradio \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --share
