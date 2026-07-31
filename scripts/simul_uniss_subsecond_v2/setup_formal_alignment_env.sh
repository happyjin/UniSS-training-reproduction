#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
BASE_PYTHON="${BASE_PYTHON:-${USER_ROOT}/conda_envs/uniss-train/bin/python3}"
ENV_ROOT="${FORMAL_ALIGN_ENV_ROOT:-${USER_ROOT}/venvs/uniss-formal-align-v1}"
HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/pip_cache}"

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${ENV_ROOT}"
fi
PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${ENV_ROOT}/bin/pip" install \
  'qwen-asr==0.0.6' \
  'simalign==0.4' \
  'jieba==0.42.1'

attempt=0
until HF_HOME="${HF_HOME}" HF_HUB_DISABLE_XET=1 "${ENV_ROOT}/bin/python" - <<'PY'
from transformers import AutoModel, AutoTokenizer

name = "bert-base-multilingual-cased"
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print(f"formal alignment model ready: {name}")
PY
do
  attempt=$((attempt + 1))
  if [[ "${attempt}" -ge 20 ]]; then
    echo "failed to prepare multilingual alignment model after ${attempt} attempts" >&2
    exit 1
  fi
  echo "alignment model download failed; retry ${attempt}/20 in 30 seconds" >&2
  sleep 30
done

