#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_BIN_DIR="${UNISS_STREAMING_FFMPEG_BIN_DIR:-${USER_ROOT}/conda_pkgs_recovered/ffmpeg-8.1.2-gpl_h1bf8424_901/bin}"

[[ -x "${DEMO_ENV}/bin/python" ]] || {
  echo "Missing reusable demo environment: ${DEMO_ENV}" >&2
  echo "Repair/install environments only below ${USER_ROOT}; do not modify the system Python." >&2
  exit 1
}

"${DEMO_ENV}/bin/python" - <<'PY'
import gradio, librosa, numpy, soundfile, torch, torchaudio, transformers
expected = "5.49.1"
if gradio.__version__ != expected:
    raise SystemExit(f"Expected gradio {expected}, found {gradio.__version__}")
print("environment-ok")
print("gradio", gradio.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY

[[ -x "${FFMPEG_BIN_DIR}/ffmpeg" && -x "${FFMPEG_BIN_DIR}/ffprobe" ]] || {
  echo "Missing reusable ffmpeg/ffprobe pair: ${FFMPEG_BIN_DIR}" >&2
  exit 1
}
"${FFMPEG_BIN_DIR}/ffmpeg" -version | head -n 1
"${FFMPEG_BIN_DIR}/ffprobe" -version | head -n 1

echo "No packages were installed; the existing isolated environment is compatible."
echo "Requirements reference: ${SCRIPT_DIR}/requirements-demo.txt"
