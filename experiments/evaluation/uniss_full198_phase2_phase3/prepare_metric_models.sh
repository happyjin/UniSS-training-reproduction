#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
ENV_ROOT="${ENV_ROOT:-${USER_ROOT}/conda_envs/uniss-eval}"
MODEL_ROOT="${MODEL_ROOT:-${USER_ROOT}/evaluation_models}"
HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-${USER_ROOT}/cache/torch}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${USER_ROOT}/cache/modelscope}"

AUTOPCP_URL="https://dl.fbaipublicfiles.com/speech_expressivity_evaluation/AutoPCP-multilingual-v2.zip"
AUTOPCP_ARCHIVE="${MODEL_ROOT}/AutoPCP-multilingual-v2.zip"
AUTOPCP_DIR="${MODEL_ROOT}/AutoPCP-multilingual-v2"

mkdir -p "${MODEL_ROOT}" "${HF_HOME}" "${TORCH_HOME}" "${MODELSCOPE_CACHE}"
if [[ ! -f "${AUTOPCP_ARCHIVE}" ]]; then
  curl -fL --retry 5 --retry-delay 5 --continue-at - -o "${AUTOPCP_ARCHIVE}" "${AUTOPCP_URL}"
fi
if [[ ! -d "${AUTOPCP_DIR}" ]]; then
  mkdir "${AUTOPCP_DIR}"
  unzip -q "${AUTOPCP_ARCHIVE}" -d "${AUTOPCP_DIR}"
fi

HF_HOME="${HF_HOME}" "${ENV_ROOT}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for model in ("openai/whisper-large-v3", "facebook/wav2vec2-large-xlsr-53"):
    print(snapshot_download(model))
PY

TORCH_HOME="${TORCH_HOME}" "${ENV_ROOT}/bin/python" - <<'PY'
import torch
model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
print(type(model).__name__)
PY

MODELSCOPE_CACHE="${MODELSCOPE_CACHE}" "${ENV_ROOT}/bin/python" - <<'PY'
from modelscope import snapshot_download
print(snapshot_download("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"))
PY

sha256sum "${AUTOPCP_ARCHIVE}" "${AUTOPCP_DIR}/model.config" "${AUTOPCP_DIR}/model.pt"
