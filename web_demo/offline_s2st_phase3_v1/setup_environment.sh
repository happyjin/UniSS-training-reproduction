#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
BASE_ENV="${BASE_ENV:-${USER_ROOT}/conda_envs/uniss-train}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
PIP_CACHE="${PIP_CACHE:-${USER_ROOT}/pip_cache}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -x "${BASE_ENV}/bin/python" ]] || { echo "Missing base environment: ${BASE_ENV}" >&2; exit 1; }
if [[ ! -x "${DEMO_ENV}/bin/python" ]]; then
  "${BASE_ENV}/bin/python" -m venv --system-site-packages "${DEMO_ENV}"
fi
"${DEMO_ENV}/bin/python" -m pip install \
  --cache-dir "${PIP_CACHE}" \
  --disable-pip-version-check \
  -r "${SCRIPT_DIR}/requirements-demo.txt"
"${DEMO_ENV}/bin/python" - <<'PY'
import gradio
import imageio_ffmpeg
import torch
import transformers
print({
    "gradio": gradio.__version__,
    "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
})
PY
