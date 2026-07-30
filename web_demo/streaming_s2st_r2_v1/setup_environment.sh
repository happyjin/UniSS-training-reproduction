#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
DEMO_ENV="${DEMO_ENV:-${USER_ROOT}/conda_envs/uniss-offline-demo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

echo "No packages were installed; the existing isolated environment is compatible."
echo "Requirements reference: ${SCRIPT_DIR}/requirements-demo.txt"
